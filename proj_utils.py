from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from pymongo import MongoClient
from pymongo.read_preferences import ReadPreference
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
#from cuml.cluster import HDBSCAN
#from cuml.manifold import UMAP
#from cuml.preprocessing import normalize
import pandas as pd
import numpy as np
import torch
from multiprocessing.managers import SyncManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
#from octis.evaluation_metrics.diversity_metrics import TopicDiversity
import nltk
# if punkit is not downloaded
if not nltk.data.find('tokenizers/punkt'):
    nltk.download("punkt")
from nltk.tokenize import word_tokenize

import re
import os
import time
import pickle

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple
from contextlib import nullcontext
from os.path import join
from collections import defaultdict, Counter
from operator import itemgetter
from copy import deepcopy

#############
# Constants #
#############

DATA_DIR = "/data/comments/valentin"

COLLECTION_NAMES = ["Gatewaypundit", "Breitbart", "Thehill", "Atlantic", "Motherjones"]  # ordered from right to left
COLLECTION_NAMES_SHORT = ["GP", "BB", "TH", "AT", "MJ"]  # ordered from right to left

DATE_RANGES_OLD = {
    "Atlantic": (datetime(2012, 6, 1), datetime(2018, 5, 1)),
    "Breitbart": (datetime(2012, 6, 1), datetime(2021, 6, 1)),
    "Gatewaypundit": (datetime(2015, 1, 1), datetime(2021, 6, 1)),
    "Motherjones": (datetime(2012, 6, 1), datetime(2019, 9, 1)),
    "Thehill": (datetime(2012, 6, 1), datetime(2021, 6, 1))
    
}

DATE_RANGES = {
    "Atlantic": (datetime(2012, 6, 1), datetime(2018, 5, 1)),
    "Breitbart": (datetime(2012, 6, 1), datetime(2023, 4, 1)),
    "Gatewaypundit": (datetime(2015, 1, 1), datetime(2023, 4, 1)),
    "Motherjones": (datetime(2012, 6, 1), datetime(2019, 9, 1)),
    "Thehill": (datetime(2012, 6, 1), datetime(2022, 3, 1))
}

MODEL_NAMES = {
    "Atlantic": 'atlantic_new_s4_r19_h200_u80_t10',
    "Breitbart": 'breitbart_new_s3_r19_h225_u20_t10',
    "Gatewaypundit": 'gatewaypundit_new_s4_r19_h400_u30_t10',
    "Motherjones": 'motherjones_new_s5_r19_h425_u90_t10',
    "Thehill": 'thehill_new_s2_r19_h300_u80_t10',
    "global": 'global_new_s1_r19_h325_u20_t10',
    "title": 'title_new_s2_r19_h35_u50_t10'
}

NUM_TOPICS = {
    "Atlantic": 219,
    "Breitbart": 287,
    "Gatewaypundit": 251,
    "Motherjones": 120,
    "Thehill": 257,
    "global": 228
}  # excluding -1

MONTH_OFFSET = {"Atlantic": 0,
    "Breitbart": 0,
    "Gatewaypundit": 31,
    "Motherjones": 0,
    "Thehill": 0,
    "global": 0}

# string s is consist of 'MMYY'. Return the text string one month after.
def next_month(s):
    month = int(s[:2])
    year = int(s[2:])
    if month == 12:
        return '01' + str(year + 1)
    else:
        return str(month + 1).zfill(2) + str(year)
    

#################
# MongoDB Utils #
#################

def query_comments_by_createdAt(db_name: str, collection_name: str, start_date: datetime = None, end_date: datetime = None, select_columns: List[str] = None) -> pd.DataFrame:
    """Query comments inside a given period

    Args:
        db_name (str): Name of the DB, e.g. "Comments"
        collection_name (str): Name of the collection, e.g. "Breitbart"
        start_date (datetime, optional): Start date of the period (inclusive). If None, include the earliers comment. Defaults to None.
        end_date (datetime, optional): End date of the period (inclusive). If None, include the latest comments. Defaults to None.
        select_columns (List[str], optional): JSON keys of the documents to return from the query. If None, return all JSON keys. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe containing the comments as rows with their keys as columns
    """
    try:
        mongo_client, collection = _init_mongo_collection(db_name, collection_name)
        query = {'raw_message': {"$not": {"$regex": '.*https?:\/\/.*[\r\n]*'}}}     # store only comments that do not have urls 
        if start_date or end_date:
            query['createdAt'] = {}
        if start_date:
            query['createdAt']["$gte"] = start_date
        if end_date:
            query['createdAt']["$lt"] = end_date

        search_comments = collection.find(query, projection=select_columns)
        total_comments = collection.count_documents(query)

        comments = []

        for comm in tqdm(search_comments, total=total_comments, desc="Query Comments"):

            proc_message = _process_raw_message(comm.get('raw_message'))

            if len(proc_message.split()) > 1:
                # filter data based on commment length; take only comments with more than one word
                comm["raw_message"] = proc_message
                comments.append(comm)
    except Exception as e:
        print(e)
        comments = []
    finally:
        if mongo_client:
            mongo_client.close()
    
    return pd.DataFrame(comments)

def query_comments_by_id(db_name: str, collection_name: str, ids: List[str], select_columns: List[str] = None, host="localhost", pbar=None, lock=None) -> pd.DataFrame:
    """Query comments by their ID

    Args:
        db_name (str): Name of the DB, e.g. "Comments"
        collection_name (str): Name of the collection, e.g. "Breitbart"
        ids (List[str]): List of the IDs of the desired comments
        select_columns (List[str], optional): JSON keys of the documents to return from the query. If None, return all JSON keys. Defaults to None.
        host (str, optional): Host address of the MongoDB host server. Defaults to "localhost".
        pbar (tqdm.pbar, optional): Progress bar to track query progress. If None, a new progress bar will be created. Defaults to None.
        lock (threading.Lock, optional): Multithreading lock to prevent concurrent updates on progress bar. If None no Lock will be used. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe containing the comments as rows with their keys as columns
    """
    if not pbar:
        pbar = tqdm(total=len(ids))
    if not lock:
        lock = nullcontext()

    try:
        mongo_client, collection = _init_mongo_collection(db_name, collection_name, host)
        
        comments = []
        i = 0
        step = 100000
        while i < len(ids):
            query = {'_id': {"$in": ids[i:i+step]}} 
            search_comments = collection.find(query, projection=select_columns)
            
            for j, comm in enumerate(search_comments):
                if not select_columns or "raw_message" in select_columns:
                    comm["raw_message"] = _process_raw_message(comm.get("raw_message"))
                
                comments.append(comm)
                if j % 10000 == 0:
                    with lock:
                        pbar.update(10000)

            i += step
    except Exception as e:
        print(e)
        comments = []
    finally:
        if mongo_client:
            mongo_client.close()
    
    return pd.DataFrame(comments)

def query_comments_by_id_conc(db_name: str, collection_name: str, ids: List[str], select_columns: List[str] = None) -> pd.DataFrame:
    """Query comments by their ID using multiple DBs concurrently

    Args:
        db_name (str): Name of the DB, e.g. "Comments"
        collection_name (str): Name of the collection, e.g. "Breitbart"
        ids (List[str]): List of the IDs of the desired comments
        select_columns (List[str], optional): JSON keys of the documents to return from the query. If None, return all JSON keys. Defaults to None.
        
    Returns:
        pd.DataFrame: A dataframe containing the comments as rows with their keys as columns
    """
    hosts = ["localhost", "vader.santafe.edu", "maul.santafe.edu"]
    #"kiribati.santafe.edu", 
    id_chunks = np.array_split(ids, len(hosts))
    with ProcessPoolExecutor(max_workers=len(hosts)) as e:
        SyncManager.register('tqdm', tqdm)
        m = SyncManager()
        m.start()
        pbar = m.tqdm(total=len(ids), desc="Query IDs")
        lock = m.Lock()

        futures = []
        for host, id_chunk in zip(hosts, id_chunks):
            futures.append(e.submit(query_comments_by_id, db_name, collection_name, id_chunk.tolist(), select_columns, host, pbar, lock))
        
        comment_lists = []
        status = []
        for f in as_completed(futures):
            comment_lists.append(f.result())
            status.append(f.done())

        assert all(status)

    return pd.concat(comment_lists)

def _init_mongo_collection(db_name: str, collection_name: str, host="localhost"):
    """Creates a connection to a MongoDB server. If the specified host is not reachable, a fallback connection is returned.

    Args:
        db_name (str): Name of the DB, e.g. "Comments"
        collection_name (str): Name of the collection, e.g. "Breitbart"
        host (str, optional): Host address of the MongoDB host server. Defaults to "localhost".

    Returns:
        Tuple: A pair with the mongo client and collection objects
    """
    mongo_client = MongoClient(host=host, maxPoolSize=32)
    db_conn = mongo_client[db_name]
    collection = db_conn[collection_name]
    try:
        cursor = collection.find(limit=1)
        next(cursor)
    except:
        mongo_client.close()

        mongo_client = MongoClient(host="vader.santafe.edu", maxPoolSize=32, replicaSet="rs0", read_preference=ReadPreference.SECONDARY)
        db_conn = mongo_client[db_name]
        collection = db_conn[collection_name]

    return mongo_client, collection

def _process_raw_message(raw_message: str) -> str:
    """Preprocessing of raw messages. Removing HTML tags and other rare characters to keep only alphanumeric chars, ', and spaces ' '

    Args:
        raw_message (str): The raw message from MongoDB

    Returns:
        str: Preprocessed message
    """
    raw_message = re.sub(r'<.+?>', ' ', str(raw_message))   # remove HTML tags
    raw_message = re.sub(r'[^A-Za-z \']', ' ', str(raw_message), flags=re.MULTILINE).lower() # remove everything except alpha chars, ', and spaces
    raw_message = re.sub(r"\s+", " ", str(raw_message), flags=re.MULTILINE).strip() # remove multiple spaces
    return raw_message

##################
# Bertopic Utils #
##################

def init_bertopic(umap_n_components: int, umap_n_neighbors: int, hdbscan_min_cluster_size: int, calculate_probabilities=False, rm_stopwords=True, random_state=19, init='spectral'):
    """Creates HDBSCAN, UMAP, and BERTopic models using specified parameters.

    Args:
        umap_n_components (int): Number of components (target dimension) for UMAP dimensionality reduction
        umap_n_neighbors (int): Number of neighbors for UMAP
        hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN
        calculate_probabilities (bool, optional): Whether to calculate all probabilities for a comment for all possible topics. Defaults to False.
        rm_stopwords (bool, optional): Whether to remove stopwords from topic keywords. Defaults to True.

    Returns:
        Tuple: Returns the topic model, HDBSCAN, and UMAP model
    """
    hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, metric='euclidean',        #min_cluster size is the most important parameter here, it is set high to limit number of the cluster and get more global topic representation
                            cluster_selection_method='eom', gen_min_span_tree=True, prediction_data=True)

    umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, metric='cosine', random_state=random_state, init=init)  #n_components is the most critical paremeter, but other can be changed as well 
    #umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, metric='cosine', random_state=random_state) # for cuml, init state not implemented
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, nr_topics='None', 
        calculate_probabilities=calculate_probabilities, vectorizer_model=(CountVectorizer(stop_words="english") if rm_stopwords else None), 
        ctfidf_model=ClassTfidfTransformer()) #nr_topics = auto merges very similar topics based on cosine similarity together
    return topic_model, hdbscan_model, umap_model

def embedd_comments(comments: List[str]) -> torch.tensor:
    """Embedds a list of text.

    Args:
        comments (List[str]): Comments to embedd

    Returns:
        torch.tensor: The embeddings of the comments with shape (len(comments), embeddings_dim)
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return sentence_model.encode(comments, show_progress_bar=True, convert_to_tensor=True)

def load_model_result(trials_dir: str, loss_rank: int):
    """Loads the topic model by its rank of the BO runs

    Args:
        trials_dir (str): Absolute path to directory containing the "trials.pkl" of BO runs
        loss_rank (int): Loss rank of the model

    Returns:
        Tuple: The topic model and the metrices of that model
    """
    with open(join(trials_dir, "trials.pkl"), "rb") as f:
        trials = pickle.load(f)
    results = sorted(trials.results, key=lambda r: r["loss"])
    result = results[loss_rank - 1]
    filename = result["bertopic_path"].split("/")[-1]
    topic_model = BERTopic.load(join(trials_dir, "bertopic-models", filename), embedding_model="all-MiniLM-L6-v2")
    topic_model._create_topic_vectors()
    return topic_model, result

def fit_and_score_model(topic_model, hdbscan_model, messages: List[str], embeddings: torch.tensor):
    """Fit the topic model and compute default scores for it.

    Args:
        topic_model: The topic model
        hdbscan_model: The HDBSCAN model used by the topic model
        messages (List[str]): A list of text documents to fit the topic model
        embeddings (torch.tensor): The pre-computed sentence embeddings of the text documents. Having shape (len(messages), embedding_dim).

    Returns:
        dict: A dictionary with metric scores
    """
    start_time = time.time()
    topics, probs = topic_model.fit_transform(messages, embeddings=embeddings.numpy())  #around 2.5M comments; it can be reduced if memory consumption is too high
    end_time = time.time()
    
    metrics = score_model(topic_model, hdbscan_model, messages, topics, probs)
    metrics["runtime_secs"] = end_time - start_time
    return metrics

def score_model(topic_model, hdbscan_model, messages: List[str], topics: List[int], probs: np.ndarray, count_rare_words=True) -> dict:
    """Score model

    Args:
        topic_model: The topic model
        hdbscan_model: The HDBSCAN model used by the topic model
        messages (List[str]): A list of text documents to fit the topic model
        topics (List[int]): List of topics
        probs (np.ndarray): Array of probabilities, having shape (len(messages), #unique_topics)
        token_doc_counts (Dict[str, int], optional): Number of unique documents each token appears in. If None, it is computed on the fly. Defaults to None.

    Returns:
        dict: A dictionary with metric scores
    """
    keywords = {topic_id: list(map(itemgetter(0), tuple_list)) for topic_id, tuple_list in topic_model.get_topics().items()}
    n_outliers = (np.array(topics) == -1).sum()
    n_topic0 = (np.array(topics) == 0).sum()
    n_topics = len(set(topics))
    metrics = {}
    if count_rare_words:
        token_doc_counts = get_token_document_counts(messages)
        metrics.update({
            'rare_words_topics': count_topics_with_rare_words(keywords, token_doc_counts, k=10) / n_topics,
            'rare_words': count_rare_words(keywords, token_doc_counts, k=10) / (n_topics * len(keywords[-1]))
        })
    
    metrics.update({
        'dbcv': hdbscan_model.relative_validity_, 
        'n_outliers': n_outliers,
        'n_outliers_-1_0': n_outliers + n_topic0,
        'outlier_ratio': n_outliers / len(topics),
        'outlier_-1_0_ratio': (n_outliers + n_topic0) / len(topics),
        'n_topics': n_topics,
        'topic_diversity' : -1,
        #'topic_diversity': TopicDiversity(topk=10).score({"topics": list(keywords.values())}),
        'n_comments': len(topics),
        'topics': topics,
        'probs': probs,
        'keywords': keywords
    })
    return metrics

#####################
# Aggregation Utils #
#####################

def filter_comments(article, aggregate_day, comment_date_cutoff, is_global, include_embeddings):
    filtered_index = np.where((pd.to_datetime(article.comment_createdAt) - article.createdAt).days < comment_date_cutoff)[0]
    
    if is_global:
        comment_topics = [article.comment_topics_global[i] for i in filtered_index]
    else:
        comment_topics = [article.comment_topics[i] for i in filtered_index]
    
    comment_embeddings = []
    if include_embeddings:
        comment_embeddings = [article.comment_embeddings[i] for i in filtered_index]
    
    return pd.Series([comment_topics, comment_embeddings])

def tmp_df_generator(collection_name, aggregate_day, include_embeddings=True, is_global=False):
    
    comment_threshold = 10
    model_name = MODEL_NAMES[collection_name]
    aggregate_start_date = datetime(2012, 5, 28, 0, 0)  ## Monday
    add_dict = {'3d':relativedelta(days=3), 'week':relativedelta(weeks=1), 'month':relativedelta(months=1)}
    comment_date_cutoff_dict = {'3d': 1, 'week' : 2, 'month': 7}

    file_names = os.listdir(join('article', collection_name.lower(), model_name, 'articles_by_day'))
    keys = sorted([file_name.split('.')[0] for file_name in file_names], key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    start = datetime.strptime(keys[0], '%Y-%m-%d')
    
    start_end_dict = {}
    tmp_list_df = {aggregate_day : []}

    start_date = deepcopy(aggregate_start_date)
    if aggregate_day == 'month':
        start_date = start.replace(day=1)
        end_date = (start_date + relativedelta(months=1)).replace(day=1)
    else:
        while True:
            if start_date + add_dict[aggregate_day] > start:
                break
            else:
                start_date += add_dict[aggregate_day]
        
        end_date = start_date + add_dict[aggregate_day] 
    start_end_dict[aggregate_day] = (start_date, end_date)
        
    for key in keys:
        #print(key)
        articles = pd.read_parquet(join('article', collection_name.lower(), model_name, 'articles_by_day', key +'.parquet'))
        articles = articles[articles['comment_id'].apply(len) > comment_threshold]

        if datetime.strptime(key, '%Y-%m-%d') >= start_end_dict[aggregate_day][1]:
            if tmp_list_df[aggregate_day]:
                tmp_df = pd.concat(tmp_list_df[aggregate_day], ignore_index=True)
                tmp_list_df[aggregate_day].clear()
                
                tmp_df[['comment_topics', 'comment_embeddings']] = tmp_df.apply(
                    lambda row: filter_comments(row, aggregate_day, comment_date_cutoff_dict[aggregate_day], is_global, include_embeddings),
                    axis=1
                )

                if include_embeddings:
                    if is_global:
                        tmp_df = tmp_df[['_id', 'topic_num_global', 'comment_topics', 'comment_embeddings']]
                    else:
                        tmp_df = tmp_df[['_id', 'topic_num', 'comment_topics', 'comment_embeddings']]
                else:
                    if is_global:
                        tmp_df = tmp_df[['_id', 'topic_num_global', 'comment_topics']]
                    else:
                        tmp_df = tmp_df[['_id', 'topic_num', 'comment_topics']]

                print(f'{aggregate_day}, {start_end_dict[aggregate_day][0].strftime("%Y-%m-%d")}')
                
                yield tmp_df
                
            start_end_dict[aggregate_day] = (start_end_dict[aggregate_day][1], start_end_dict[aggregate_day][1] + add_dict[aggregate_day])
            
            
        tmp_list_df[aggregate_day].append(articles)
        
        del articles

    # process leftovers
    print(f'{aggregate_day}, leftovers')
    tmp_df = pd.concat(tmp_list_df[aggregate_day])

    if len(tmp_df) > 0:
        
        tmp_df[['comment_topics', 'comment_embeddings']] = tmp_df.apply(lambda row: filter_comments(row, aggregate_day, comment_date_cutoff_dict[aggregate_day], is_global, include_embeddings), axis=1)

        if include_embeddings:
            if is_global:
                tmp_df = tmp_df[['_id', 'topic_num_global', 'comment_topics', 'comment_embeddings']]
            else:
                tmp_df = tmp_df[['_id', 'topic_num', 'comment_topics', 'comment_embeddings']]
        else:
            if is_global:
                tmp_df = tmp_df[['_id', 'topic_num_global', 'comment_topics']]
            else:
                tmp_df = tmp_df[['_id', 'topic_num', 'comment_topics']]
                
        yield tmp_df

    tmp_list_df[aggregate_day] = []



##############
# MISC Utils #
##############

def reassign_outliers(topics, probs, probability_threshold=0.01):
    """
    Reassign outliers to their most likely topics
    """

    return [np.argmax(prob) if (max(prob) >= probability_threshold and topic == -1) else topic for topic, prob in zip(topics, probs)]

def collect_sampled_data(collection_name: str, size: int, seed: int, start_date=datetime(2016, 4, 1), end_date=datetime(2017, 4, 1), select_columns=None, mode='new'):
    """Samples data, collects the respective sentence embeddings, and queries additional information from the database.

    Args:
        collection_name (str): Name of the collection, e.g. "Breitbart"
        size (int): Sampling size
        seed (int): Random seed to create the sample, if it wasn't created before. Note that a different seed will not return a different sample, if an earlier sample of this collection with the specified size already exists.
        start_date (datetime, optional): Lower bound of the time period to sample from (inclusive). Defaults to datetime(2016, 4, 1).
        end_date (datetime, optional): Upper bound of the time period to sample from (inclusive). Defaults to datetime(2017, 4, 1).
        select_columns (List[str], optional): JSON keys of the documents to return from the query. If None, return all JSON keys. Defaults to None.

    Returns:
        dict: A dictionary with keys ["_id", "embeddings"] + select_columns. Each value is a list of the same length, except for "embeddings", which is a tensor shape[0] equals to the list lengths.
    """

    # Sample data

    if mode=='new':

        collection_dir = join(DATA_DIR, f"new_ids/{collection_name.lower()}") 
        sample_filename = f"s{size}_{start_date.strftime('%m%y')}_{end_date.strftime('%m%y')}_sd{seed}.csv"
        all_ids_df = pd.read_csv(join(collection_dir, f"all.csv"), dtype={"id": str})
        filenames_mask = _filter_filenames_between(all_ids_df["filename"].unique(), start_date, end_date)
        filtered_ids_df = all_ids_df[all_ids_df["filename"].isin(filenames_mask)]
        
        sampled_ids_df = filtered_ids_df.sample(n=size, random_state=seed)
        #sampled_ids_df.to_csv(join(collection_dir, sample_filename), index=False)
        
    elif mode=='old':
        collection_dir = join(DATA_DIR, f"ids/{collection_name.lower()}")
        sample_filename = f"s{size}_{start_date.strftime('%m%y')}_{end_date.strftime('%m%y')}.csv"
        all_ids_df = pd.read_csv(join(collection_dir, f"all.csv"), dtype={"ids": str})
        filenames_mask = _filter_filenames_between(all_ids_df["filename"].unique(), start_date, end_date)
        filtered_ids_df = all_ids_df[all_ids_df["filename"].isin(filenames_mask)]
        
        sampled_ids_df = filtered_ids_df.sample(n=size, random_state=seed)
        #sampled_ids_df.to_csv(join(collection_dir, sample_filename), index=False)
    else:
        print('error')

    # Read sentence embeddins for sampled data
    sampled_ids_df = sampled_ids_df.set_index("filename")
    embeddings_dict = defaultdict(list)
    for filename in tqdm(sampled_ids_df.index.unique(), desc="Collect sampled data"):
        if mode=='new':
            embeddings_month = torch.load(
            join(f"/data/comments/valentin/sentence-embeddings/{collection_name.lower()}", filename),
            map_location=torch.device("cpu"))
            sampled_ids = sampled_ids_df.loc[filename]["id"].tolist()
        elif mode=='old':
            embeddings_month = torch.load(
            join(f"/local/vrupp/sent-embeddings/{collection_name.lower()}", filename), 
            map_location=torch.device("cpu"))
            sampled_ids = sampled_ids_df.loc[filename]["ids"].tolist()
        else:
            print('error')

        sampled_ids_set = set(sampled_ids)
        indices = [i for i, id in enumerate(embeddings_month["_id"]) if id in sampled_ids_set]
        for k, v in embeddings_month.items():
            if k == "embeddings":
                embeddings_dict[k].append(v[indices])
            elif k == "_id":
                embeddings_dict[k].extend(np.array(v)[indices])
    embeddings_dict = dict(embeddings_dict)
    embeddings_dict["embeddings"] = torch.cat(embeddings_dict["embeddings"], dim=0)

    # Query additional information from MongoDB
    if select_columns and len(set(select_columns).difference(embeddings_dict.keys())) > 0:
        comments_df = query_comments_by_id("Comments", collection_name, embeddings_dict["_id"], select_columns=select_columns) \
            .set_index("_id") \
            .loc[embeddings_dict["_id"]]

        comments_dict = comments_df.to_dict(orient="list")
        for k, v in comments_dict.items():
            if k not in embeddings_dict:
                embeddings_dict[k] = v

    return embeddings_dict

def get_threshold(freq, sampling_size):
    
    '''calculate k for stratified samplings'''

    sum_freq = sum(freq)
    cumsum_freq = np.cumsum(freq)
    cumsum_i_freq = np.cumsum((freq * (np.arange(len(freq))+1)))
    for k in range(1, len(freq)+1):
        if k * (sum_freq - cumsum_freq[k-1]) + cumsum_i_freq[k-1] > sampling_size:
            break
        
    return k-1

def collect_sampled_data_new(collection_name: str, size: int, seed: int, start_date=datetime(2016, 4, 1), end_date=datetime(2017, 4, 1), select_columns=None, mode='new', threshold=10):
    """Samples data, collects the respective sentence embeddings, and queries additional information from the database.
    NOW SAMPLING DATA BY ARTICLE BASE

    Args:
        collection_name (str): Name of the collection, e.g. "Breitbart"
        size (int): Sampling size
        seed (int): Random seed to create the sample, if it wasn't created before. Note that a different seed will not return a different sample, if an earlier sample of this collection with the specified size already exists.
        start_date (datetime, optional): Lower bound of the time period to sample from (inclusive). Defaults to datetime(2016, 4, 1).
        end_date (datetime, optional): Upper bound of the time period to sample from (inclusive). Defaults to datetime(2017, 4, 1).
        select_columns (List[str], optional): JSON keys of the documents to return from the query. If None, return all JSON keys. Defaults to None.

    Returns:
        dict: A dictionary with keys ["_id", "embeddings"] + select_columns. Each value is a list of the same length, except for "embeddings", which is a tensor shape[0] equals to the list lengths.
    """

    # Sample data
    
    print(mode)

    if mode=='new':
        collection_dir = join(DATA_DIR, f"new_ids/{collection_name.lower()}") 
        sample_filename = f"s{size}_{start_date.strftime('%m%y')}_{end_date.strftime('%m%y')}_sd{seed}.csv"
        all_ids_df = pd.read_csv(join(collection_dir, f"all.csv"), dtype={"id": str})
        filenames_mask = _filter_filenames_between(all_ids_df["filename"].unique(), start_date, end_date)
        filtered_ids_df = all_ids_df[all_ids_df["filename"].isin(filenames_mask)]
        
        
    elif mode=='old':
        collection_dir = join(DATA_DIR, f"ids/{collection_name.lower()}")
        sample_filename = f"s{size}_{start_date.strftime('%m%y')}_{end_date.strftime('%m%y')}.csv"
        all_ids_df = pd.read_csv(join(collection_dir, f"all.csv"), dtype={"ids": str})
        filenames_mask = _filter_filenames_between(all_ids_df["filename"].unique(), start_date, end_date)
        filtered_ids_df = all_ids_df[all_ids_df["filename"].isin(filenames_mask)]
        
    else:
        print('error')
        
    # Stratified sampling
    time_period = relativedelta(end_date, start_date)
    messages, topics, timestamps = [], [], []
    art_id_counter = Counter()
    art_id_dict = defaultdict(list)
    for embeddings_month, current_date in tqdm(gen_sent_embeddings(collection_name, start_date, end_date, mode='new'), total=12 * time_period.years + time_period.months):
        comments_df = query_comments_by_id("Comments", collection_name, embeddings_month["_id"],  select_columns=["_id", "raw_message", "createdAt", "art_id"]) \
            .set_index("_id").loc[embeddings_month["_id"]]
        temp = Counter(comments_df['art_id'].values)
        art_id_counter += temp
        
        # make a dictionary from comments_df, where the key is art_id and value is the lis of _id shares the same art_id
        for i, row in comments_df.iterrows():
            art_id_dict[row['art_id']].append(i)
    
    art_id_counter_filtered = {k: v for k, v in art_id_counter.items() if v >= threshold}
    
    freq, bins = np.histogram(list(art_id_counter_filtered.values()), bins=np.arange(max(art_id_counter_filtered.values())+1))
    threshold = get_threshold(freq, size)

    # add ids into id_list by sampling from each article
    id_list = []
    np.random.seed(seed)  # Fix the seed
    for key in art_id_counter_filtered:
        if art_id_counter[key] <= threshold:
            # add all comments
            id_list += art_id_dict[key]
        else:
            id_list += list(np.random.choice(art_id_dict[key], threshold, replace=False))
            
    sampled_ids_df = filtered_ids_df[filtered_ids_df['id'].isin(id_list)]

    # Read sentence embeddins for sampled data
    sampled_ids_df = sampled_ids_df.set_index("filename")
    embeddings_dict = defaultdict(list)
    for filename in tqdm(sampled_ids_df.index.unique(), desc="Collect sampled data"):
        if mode=='new':
            embeddings_month = torch.load(
            join(f"/data/comments/valentin/sentence-embeddings/{collection_name.lower()}", filename),
            map_location=torch.device("cpu"))
            sampled_ids = sampled_ids_df.loc[filename]["id"].tolist()
        elif mode=='old':
            embeddings_month = torch.load(
            join(f"/local/vrupp/sent-embeddings/{collection_name.lower()}", filename), 
            map_location=torch.device("cpu"))
            sampled_ids = sampled_ids_df.loc[filename]["ids"].tolist()
        else:
            print('error')

        sampled_ids_set = set(sampled_ids)
        indices = [i for i, id in enumerate(embeddings_month["_id"]) if id in sampled_ids_set]
        for k, v in embeddings_month.items():
            if k == "embeddings":
                embeddings_dict[k].append(v[indices])
            elif k == "_id":
                embeddings_dict[k].extend(np.array(v)[indices])
    embeddings_dict = dict(embeddings_dict)
    embeddings_dict["embeddings"] = torch.cat(embeddings_dict["embeddings"], dim=0)

    # Query additional information from MongoDB
    if select_columns and len(set(select_columns).difference(embeddings_dict.keys())) > 0:
        comments_df = query_comments_by_id("Comments", collection_name, embeddings_dict["_id"], select_columns=select_columns) \
            .set_index("_id") \
            .loc[embeddings_dict["_id"]]

        comments_dict = comments_df.to_dict(orient="list")
        for k, v in comments_dict.items():
            if k not in embeddings_dict:
                embeddings_dict[k] = v

    return embeddings_dict

def _filter_filenames_between(filenames: List[str], start_date: datetime, end_date: datetime):
    """Filter a list of filenames, which contain batches in a given time period

    Args:
        filenames (List[str]): Filenames to filter
        start_date (datetime): Lower bound of the time period (inclusive)
        end_date (datetime): Upper bound of the time period (inclusive)

    Returns:
        List[str]: Filtered list of filenames
    """
    filtered_filenames = []
    for filename in filenames:
        file_date_start, file_date_end = filename.replace(".pt", "").split("-")[-2:]
        file_date_start = datetime.strptime(file_date_start, "%m%y")
        file_date_end = datetime.strptime(file_date_end, "%m%y")
        if (start_date <= file_date_start <= end_date) and (start_date <= file_date_end <= end_date):
            filtered_filenames.append(filename)
    return filtered_filenames

def gen_sent_embeddings(collection_name: str, start_date=datetime(2016, 4, 1), end_date=datetime(2017, 4, 1), mode='new') -> Tuple[dict, datetime]:
    """Generate monthly stored sentence embeddings

    Args:
        collection_name (str): _description_
        start_date (datetime, optional): Lower bound of the time period to read embeddings from (inclusive). Defaults to datetime(2016, 4, 1).
        end_date (datetime, optional): Upper bound of the time period to read embeddings from (inclusive). Defaults to datetime(2017, 4, 1).

    Yields:
        Iterator[Tuple[dict, datetime]]: A dictionary with keys ["_id", "embeddings"] and the month of the embeddings
    """
    next_date = start_date + relativedelta(months=1)
    while next_date <= end_date:
        if mode == 'new':
            embeddings_month = torch.load(
            f"/data/comments/valentin/sentence-embeddings/{collection_name.lower()}/bert-emb-{start_date.strftime('%m%y')}-{next_date.strftime('%m%y')}.pt", 
            map_location=torch.device("cpu"))
            yield embeddings_month, start_date
        elif mode == 'old':
            embeddings_month = torch.load(
            f"/local/vrupp/sent-embeddings/{collection_name.lower()}/bert-emb-{start_date.strftime('%m%y')}-{next_date.strftime('%m%y')}.pt", 
            map_location=torch.device("cpu"))
            yield embeddings_month, start_date
        else:
            print('error')
        
        start_date += relativedelta(months=1)
        next_date += relativedelta(months=1)

def get_token_document_counts(docs: List[str]) -> Counter:
    """Count unique tokens in a text corpus (i.e. list of documents)
    """
    tokens = []
    for raw_message in tqdm(docs):
        tokens.extend(set(word_tokenize(raw_message)))
    return Counter(tokens)

def count_topics_with_rare_words(keywords: Dict[int, List[str]], token_doc_counts: Dict[str, int], k=10):
    """Count the number of topics that contain a rare keyword

    Args:
        keywords (Dict[int, List[str]]): Topic to keyword mapping
        token_doc_counts (Dict[str, int]): Number of unique documents each token appears in
        k (int, optional): Minimum token count to not be a rare word, i.e. tokens with < k appearence are rare tokens. Defaults to 10.

    Returns:
        int: Number of topics containing rare keywords
    """
    rare_words = set([word for word, doc_count in token_doc_counts.items() if doc_count < k])
    topic_count = 0
    for topic_id, words in keywords.items():
        if len(set(words).intersection(rare_words)) > 0:
            topic_count += 1
    return topic_count

def count_rare_words(keywords: Dict[int, List[str]], token_doc_counts: Dict[str, int], k=10):
    """Count the total number of rare words over all keywords of all topics.

    Args:
        keywords (Dict[int, List[str]]): Topic to keyword mapping
        token_doc_counts (Dict[str, int]): Number of unique documents each token appears in
        k (int, optional): Minimum token count to not be a rare word, i.e. tokens with < k appearence are rare tokens. Defaults to 10.

    Returns:
        int: Total number of rare words.
    """
    rare_words = set([word for word, doc_count in token_doc_counts.items() if doc_count < k])
    rare_words_count = 0
    for topic_id, words in keywords.items():
        rare_words_count += len(set(words).intersection(rare_words))
    return rare_words_count


## Inter-model topic matching methods

def cos_sim_matching(dense_list, name_list, mode):
    nameset_list = [set(name) for name in name_list]
    cos_sim_list = []
    matching_list = []
    for i in range(len(dense_list)):
        cos_sim_list.append([])
        matching_list.append([])
        for j in range(len(dense_list)):
            if i > j and i != j:
                matching, score_list = cos_sim_between_models(dense_list, name_list, nameset_list, mode, i, j)
                cos_sim_list[-1].append(sorted(score_list, reverse=True))
                matching_list[-1].append(matching)
            else:
                cos_sim_list[-1].append(0)
                matching_list[-1].append([])
    
    return cos_sim_list, matching_list
    
def cos_sim_between_models(dense_list, name_list, nameset_list, mode, i, j):
    
    if mode == 'ctfidf':
    
        shared_sorted = sorted(nameset_list[i].intersection(nameset_list[j]))

        counter = 0
        shared_index_ij = []
        for word in shared_sorted:
            while True:
                if word == name_list[i][counter]:
                    shared_index_ij.append(counter)
                    break
                else:
                    counter += 1
                    
        counter = 0
        shared_index_ji = []
        for word in shared_sorted:
            while True:
                if word == name_list[j][counter]:
                    shared_index_ji.append(counter)
                    break
                else:
                    counter += 1
            
        dense_i_shared = dense_list[i][:, shared_index_ij]
        dense_j_shared = dense_list[j][:, shared_index_ji]
    
        cos_sim_matrix = cosine_similarity(dense_i_shared, dense_j_shared)

    elif mode == 'embeddings':
        cos_sim_matrix = cosine_similarity(dense_list[i], dense_list[j])
    else:
        raise NotImplementedError

    row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix)

    matching = []
    for i, j in zip(row_ind, col_ind):
        matching.append((i, j))
    
    return matching, cos_sim_matrix[row_ind, col_ind]


###

