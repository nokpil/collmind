# %%
import pandas as pd
import numpy as np
from os.path import join
import pickle
from typing import Tuple, List, Union

from tqdm import tqdm
import pymongo
import torch
from dateutil.relativedelta import relativedelta
import bertopic
from sklearn.preprocessing import normalize
from scipy import sparse

from proj_utils import *


# %%
def topics_over_time(topic_model,
                        docs: List[str],
                        topics: List[int],
                        timestamps: Union[List[str], List[int]],
                        nr_bins: int = None,
                        datetime_format: str = None,
                        evolution_tuning: bool = True,
                        global_tuning: bool = True) -> pd.DataFrame:
    """ Adaption from https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py to return also c-tfidf vectors
    """
    bertopic._utils.check_is_fitted(topic_model)
    bertopic._utils.check_documents_type(docs)
    documents = pd.DataFrame({"Document": docs, "Topic": topics, "Timestamps": timestamps})
    global_c_tf_idf = normalize(topic_model.c_tf_idf_, axis=1, norm='l1', copy=False)

    all_topics = sorted(list(documents.Topic.unique())) 
    all_topics_indices = {topic: index for index, topic in enumerate(all_topics)}

    if isinstance(timestamps[0], str):
        infer_datetime_format = True if not datetime_format else False
        documents["Timestamps"] = pd.to_datetime(documents["Timestamps"],
                                                    infer_datetime_format=infer_datetime_format,
                                                    format=datetime_format)

    if nr_bins:
        documents["Bins"] = pd.cut(documents.Timestamps, bins=nr_bins)
        documents["Timestamps"] = documents.apply(lambda row: row.Bins.left, 1)

    # Sort documents in chronological order
    documents = documents.sort_values("Timestamps")
    timestamps = documents.Timestamps.unique()
    if len(timestamps) > 100:
        bertopic._utils.warnings.warn(f"There are more than 100 unique timestamps (i.e., {len(timestamps)}) "
                        "which significantly slows down the application. Consider setting `nr_bins` "
                        "to a value lower than 100 to speed up calculation. ")

    # For each unique timestamp, create topic representations
    topics_over_time = []
    c_tf_idf_dict = {}
    for index, timestamp in tqdm(enumerate(timestamps), total=len(timestamps), disable=not topic_model.verbose):

        # Calculate c-TF-IDF representation for a specific timestamp
        selection = documents.loc[documents.Timestamps == timestamp, :]
        documents_per_topic = selection.groupby(['Topic'], as_index=False).agg({'Document': ' '.join,
                                                                                "Timestamps": "count"}) 
    
        c_tf_idf, words = topic_model._c_tf_idf(documents_per_topic, fit=False)

        if global_tuning or evolution_tuning:
            c_tf_idf = normalize(c_tf_idf, axis=1, norm='l1', copy=False)

        # Fine-tune the c-TF-IDF matrix at timestamp t by averaging it with the c-TF-IDF
        # matrix at timestamp t-1
        if evolution_tuning and index != 0:
            current_topics = sorted(list(documents_per_topic.Topic.values))
            overlapping_topics = sorted(list(set(previous_topics).intersection(set(current_topics))))

            current_overlap_idx = [current_topics.index(topic) for topic in overlapping_topics]
            previous_overlap_idx = [previous_topics.index(topic) for topic in overlapping_topics]

            c_tf_idf.tolil()[current_overlap_idx] = ((c_tf_idf[current_overlap_idx] +
                                                        previous_c_tf_idf[previous_overlap_idx]) / 2.0).tolil()

        # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
        # by simply taking the average of the two
        if global_tuning:
            selected_topics = [all_topics_indices[topic] for topic in documents_per_topic.Topic.values]
            c_tf_idf = (global_c_tf_idf[selected_topics] + c_tf_idf) / 2.0 
            

        # Extract the words per topic
        words_per_topic = topic_model._extract_words_per_topic(words, selection, c_tf_idf, calculate_aspects=False)
        topic_frequency = pd.Series(documents_per_topic.Timestamps.values,
                                    index=documents_per_topic.Topic).to_dict()

        # Fill dataframe with results
        topics_at_timestamp = [(topic,
                                ", ".join([words[0] for words in values][:10]),
                                topic_frequency[topic],
                                timestamp) for topic, values in words_per_topic.items()]
        topics_over_time.extend(topics_at_timestamp)

        if evolution_tuning:
            previous_topics = sorted(list(documents_per_topic.Topic.values))
            previous_c_tf_idf = c_tf_idf.copy()

        timestamp_str = np.datetime_as_string(timestamp, unit="s", timezone="UTC")
        c_tf_idf_dict[timestamp_str] = c_tf_idf

    return pd.DataFrame(topics_over_time, columns=["Topic", "Words", "Frequency", "Timestamp"]), c_tf_idf_dict


# %%

collection_name = "Motherjones"
topic_model = BERTopic.load('/data/comments/valentin/topic-modeling-new-fit/' + collection_name.lower() + '/model', embedding_model="all-MiniLM-L6-v2")
topic_model.verbose = True
topic_model._create_topic_vectors()


# %%
start_date, end_date = DATE_RANGES[collection_name]
time_period = relativedelta(end_date, start_date)
messages, topics, timestamps = [], [], []
for embeddings_month, current_date in tqdm(gen_sent_embeddings(collection_name, start_date, end_date), total=12 * time_period.years + time_period.months):
    comments_df = query_comments_by_id("Comments", collection_name, embeddings_month["_id"],  select_columns=["_id", "raw_message", "createdAt"]) \
        .set_index("_id").loc[embeddings_month["_id"]]
    messages += comments_df["raw_message"].tolist()
    timestamps += comments_df["createdAt"].dt.strftime("%Y-%m-%d").tolist()
    batch = pd.read_feather(join('/data/comments/valentin/topic-modeling-new-transform/', collection_name.lower(), f"batch-{current_date.strftime('%m%y')}.arrow"))
    assert batch["id"].tolist() == embeddings_month["_id"]
    topics.extend(batch["topic"])


# %%
bins = [start_date + relativedelta(months=i, days=-1) for i in range(12 * time_period.years + time_period.months)] + [end_date]
tops_over_time, c_tf_idf_dict = topics_over_time(topic_model, topics=topics, docs=messages, timestamps=timestamps, nr_bins=bins, datetime_format="%Y-%m-%d", evolution_tuning=True, global_tuning=True)


# %%
tops_over_time["Month"] = tops_over_time["Timestamp"] + timedelta(days=1)
ctfidf_per_month = {(pd.to_datetime(date_str) + timedelta(days=1)).strftime("%m%y"): c_tf_idf_csr for date_str, c_tf_idf_csr in c_tf_idf_dict.items()}
tops_over_time.head()


# %%
results_dir = join('/data/comments/valentin/', "ctfidf_new", collection_name.lower())
os.makedirs(results_dir, exist_ok=True)
tops_over_time.drop(columns=["Timestamp"]).to_csv(join(results_dir, "topics_per_month.csv"), index=False)
for month_year, c_tf_idf_csr in tqdm(ctfidf_per_month.items(), total=len(ctfidf_per_month.keys())):
    c_tf_idf_coo = c_tf_idf_csr.tocoo()
    pd.DataFrame({"row": c_tf_idf_coo.row, "col": c_tf_idf_coo.col, "x": c_tf_idf_coo.data}) \
        .to_feather(join(results_dir, f"ctfidf-{month_year}.arrow"))
