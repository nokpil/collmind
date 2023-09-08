from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from pymongo import MongoClient
from pymongo.read_preferences import ReadPreference
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP 
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
import numpy as np
import torch
from multiprocessing.managers import SyncManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.feature_extraction.text import CountVectorizer


import re
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple
from contextlib import nullcontext
import os
from os.path import join
import pickle
from collections import defaultdict, Counter
from operator import itemgetter
import time


#############
# Constants #
#############

DATA_DIR = "/data/comments/valentin"
DATE_RANGES = {
    "Breitbart": (datetime(2021, 6, 1), datetime(2023, 7, 1)),
    "Atlantic": (datetime(2012, 6, 1), datetime(2018, 5, 1)),
    "Thehill": (datetime(2021, 6, 1), datetime(2023, 7, 1)),
    "Motherjones": (datetime(2012, 1, 1), datetime(2019, 9, 1)),
    "Gatewaypundit": (datetime(2016, 6, 1), datetime(2023, 7, 1))
}

#################
# MongoDB Utils #
#################

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

def init_bertopic(umap_n_components: int, umap_n_neighbors: int, hdbscan_min_cluster_size: int, calculate_probabilities=False, rm_stopwords=True):
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

    umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, metric='cosine', random_state=19)  #n_components is the most critical paremeter, but other can be changed as well 

    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, nr_topics='None', 
        calculate_probabilities=calculate_probabilities, vectorizer_model=(CountVectorizer(stop_words="english") if rm_stopwords else None), 
        ctfidf_model=ClassTfidfTransformer())
    return topic_model, hdbscan_model, umap_model



##############
# MISC Utils #
##############


def collect_sampled_data(collection_name: str, size: int, seed: int, start_date=datetime(2016, 4, 1), end_date=datetime(2017, 4, 1), select_columns=None):
    """Samples data, collects the respective sentence embeddings, and queries additional information from the database.

    Args:
        collection_name (str): Name of the collection, e.g. "Breitbart"
        size (int): Sampling size
        seed (int): Random seed to create the sample
        start_date (datetime, optional): Lower bound of the time period to sample from (inclusive). Defaults to datetime(2016, 4, 1).
        end_date (datetime, optional): Upper bound of the time period to sample from (inclusive). Defaults to datetime(2017, 4, 1).
        select_columns (List[str], optional): JSON keys of the documents to return from the query. If None, return all JSON keys. Defaults to None.

    Returns:
        dict: A dictionary with keys ["_id", "embeddings"] + select_columns. Each value is a list of the same length, except for "embeddings", which is a tensor shape[0] equals to the list lengths.
    """

    # Sample data
    collection_dir = join(DATA_DIR, f"new_ids/{collection_name.lower()}") 
    sample_filename = f"s{size}_{start_date.strftime('%m%y')}_{end_date.strftime('%m%y')}.csv"
    all_ids_df = pd.read_csv(join(collection_dir, f"all.csv"), dtype={"id": str})
    filenames_mask = _filter_filenames_between(all_ids_df["filename"].unique(), start_date, end_date)
    filtered_ids_df = all_ids_df[all_ids_df["filename"].isin(filenames_mask)]

    sampled_ids_df = filtered_ids_df.sample(n=size, random_state=seed)
    sampled_ids_df.to_csv(join(collection_dir, sample_filename), index=False)
    
    # Read sentence embeddins for sampled data
    sampled_ids_df = sampled_ids_df.set_index("filename")
    embeddings_dict = defaultdict(list)
    for filename in tqdm(sampled_ids_df.index.unique(), desc="Collect sampled data"):
        embeddings_month = torch.load(
            join(f"/data/comments/valentin/sentence-embeddings/{collection_name.lower()}", filename), 
            map_location=torch.device("cpu"))
        sampled_ids = sampled_ids_df.loc[filename]["id"].tolist()
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

def gen_sent_embeddings(collection_name: str, start_date=datetime(2016, 4, 1), end_date=datetime(2017, 4, 1)) -> Tuple[dict, datetime]:
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
        embeddings_month = torch.load(
            f"/data/comments/valentin/sentence-embeddings/{collection_name.lower()}/bert-emb-{start_date.strftime('%m%y')}-{next_date.strftime('%m%y')}.pt", 
            map_location=torch.device("cpu"))
        yield embeddings_month, start_date

        start_date += relativedelta(months=1)
        next_date += relativedelta(months=1)






