# %%
from os.path import join
import time
import json
import pickle
import os
import sys
from argparse import ArgumentParser
from typing import List, Tuple
from glob import glob
from collections import defaultdict

import pymongo
import pandas as pd
import numpy as np
from tqdm import tqdm

from proj_utils import *


# %%
# Function definitions

def fit(collection_names: List[str], sample_sizes: List[int], params: List[List[int]]):
    """
    Args:
        collection_names (List[str]): Collection names on which to train a model on
        sample_sizes (List[int]): Sample sizes for different fits
        params (List[List[int]]): List of pairs [hdbscan_min_cluster_size, umap_n_neighbors]
    """
    with tqdm(total=len(collection_names) * len(sample_sizes) * len(params), desc="Fit Models") as pbar:
        for collection_name in collection_names:
            start_date, end_date = DATE_RANGES[collection_name]
            
            for sampling_size in sample_sizes:
                for hdb_min_cluster_size, umap_n_neighbors in params:
                    sampled_embeddings_dict = collect_sampled_data(collection_name, sampling_size, seed=19, start_date=start_date, end_date=end_date, 
                    select_columns=["_id", "raw_message"])  
               
                    topic_model, hdbscan_model, umap_model = init_bertopic(umap_n_components=2, umap_n_neighbors=umap_n_neighbors,
                    hdbscan_min_cluster_size=hdb_min_cluster_size, calculate_probabilities=False)

                    topic_model.fit(documents = sampled_embeddings_dict["raw_message"], embeddings = sampled_embeddings_dict["embeddings"].numpy())

                    topic_model.save('/data/comments/valentin/topic-modeling-new-fit/' + collection_name.lower() + '/model' , save_embedding_model=False)


def transform(model_subpath):
    topic_model = BERTopic.load(model_subpath, embedding_model="all-MiniLM-L6-v2")
    topic_model.calculate_probabilities = False
    topic_model._create_topic_vectors()

    collection_name = 'Motherjones'
    start_date, end_date = DATE_RANGES[collection_name]
    time_period = relativedelta(end_date, start_date)
    results_dir = ("/data/comments/valentin/topic-modeling-new-transform/" + collection_name.lower())
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for embeddings_month, current_date in tqdm(gen_sent_embeddings(collection_name, start_date, end_date), total=12 * time_period.years + time_period.months):
        batch_path = join(results_dir, f"batch-{current_date.strftime('%m%y')}.arrow")
        if not os.path.exists(batch_path):
            comments_df = query_comments_by_id("Comments", collection_name, embeddings_month["_id"], select_columns=["_id", "raw_message"]) \
                .set_index("_id")
            messages = comments_df.loc[embeddings_month["_id"], "raw_message"].tolist()
            
            topics, probs = topic_model.transform(messages, embeddings=embeddings_month["embeddings"].numpy())

            batch = {"id": embeddings_month["_id"], "topic": topics}
            pd.DataFrame(batch).to_feather(batch_path)


# %%
parser = ArgumentParser()
parser.add_argument("operation", choices=["fit", "transform"])
parser.add_argument("--collection_names", nargs="+")
parser.add_argument("--sample_sizes", nargs="+", type=int)
parser.add_argument("--params")
parser.add_argument("--model_subpath")
args = vars(parser.parse_args(sys.argv[1:]))
print(args)
if args["operation"] == "fit":
    fit(args["collection_names"], args["sample_sizes"], json.loads(args["params"]))
elif args["operation"] == "transform":
    transform(args["model_subpath"])
else:
    print(sys.argv)
    print("Usage")
    print("python fit-final-model.py fit")
    print("python fit-final-model.py transform <db_collection_name> <model_path>")

exit(0)

# %%
# fit --collection_names Thehill Motherjones Gatewaypundit --sample_sizes 200000 1000000 --params "[[159, 38]]"
# fit --collection_names Motherjones --sample_sizes 200000 1000000 --params "[[159, 38]]"
# nohup python fit-final-models.py transform --model_subpath breitbart/s2000000/bertopic_nneigh-21_minsize-517 > transform-b.out &
# nohup python fit-final-models.py transform --model_subpath thehill/s2000000/bertopic_nneigh-69_minsize-396 > transform-th.out &


