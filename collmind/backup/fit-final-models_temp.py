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
from sklearn.decomposition import PCA

from proj_utils import *


# %%
# Function definitions


def fit(collection_name: str, sample_size: int, param: List[int], seed: int, random_state: int, mode: str, threshold: int, text: str):

    start_date, end_date = DATE_RANGES[collection_name]
    hdb_min_cluster_size, umap_n_neighbors = param
    
    print(f'hdb_min_cluster_size: {hdb_min_cluster_size}, umap_n_neighbors: {umap_n_neighbors}, collection_name: {collection_name}, sampling_size: {sample_size}')
    print(f'seed: {seed}, random_state: {random_state}, mode: {mode}, threshold: {threshold}, text: {text}')

    with open(join('result', collection_name.lower(), 'sampled_embeddings_dict', f'sampled_embeddings_dict_{seed}.pkl'), 'rb') as f:
        sampled_embeddings_dict = pickle.load(f)

    print('data sampling finished')
    
    init_embeddings = sampled_embeddings_dict["embeddings"].numpy()
    pca_embeddings = PCA(n_components=2, svd_solver='full').fit_transform(init_embeddings)
    pca_embeddings = np.array(pca_embeddings, copy=True)
    pca_embeddings /= np.std(pca_embeddings[:, 0]) * 10000

    topic_model, hdbscan_model, umap_model = init_bertopic(umap_n_components=2, umap_n_neighbors=umap_n_neighbors,
    hdbscan_min_cluster_size=hdb_min_cluster_size, calculate_probabilities=False, random_state=int(random_state), init=pca_embeddings)

    topic_model.fit(documents = sampled_embeddings_dict["raw_message"], embeddings = sampled_embeddings_dict["embeddings"].numpy())
    
    print('fit done')
    topic_model.save(f'model/{collection_name.lower()}/{collection_name.lower()}_new_s{seed}_r{random_state}_h{hdb_min_cluster_size}_u{umap_n_neighbors}_t{threshold}{"_" + text if text else ""}', save_embedding_model=False)

def transform(model_subpath):
    
    collection_name = model_subpath.split("/")[0]
    collection_name = collection_name[0].upper() + collection_name[1:]
    print(collection_name)

    topic_model = BERTopic.load(join('model', model_subpath), embedding_model="all-MiniLM-L6-v2")
    topic_model.calculate_probabilities = False
    topic_model._create_topic_vectors()
    
    start_date, end_date = DATE_RANGES[collection_name]
    time_period = relativedelta(end_date, start_date)
    print(time_period)
    results_dir = 'transform/' + model_subpath
    article_dir = 'article/' + model_subpath
    
    article_dict = {}
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(article_dir):
        os.makedirs(article_dir)
        
    for embeddings_month, current_date in tqdm(gen_sent_embeddings(collection_name, start_date, end_date), total=12 * time_period.years + time_period.months):
        batch_path = join(results_dir, f"batch-{current_date.strftime('%m%y')}.arrow")
        # if current date is before 21/01/01, skip
        if current_date < datetime(2022, 3, 1):
            print(current_date)
            comments_df = query_comments_by_id("Comments", collection_name, embeddings_month["_id"], select_columns=["_id", "raw_message", "art_id", "createdAt"])
            comments_df = comments_df.set_index("_id")
            messages = comments_df.loc[embeddings_month["_id"], "raw_message"].tolist()
            
            batch = pd.read_feather(batch_path)
            topics = batch['topic']
            
            comments_df.reset_index(inplace=True)
            comments_df['topics'] = topics
            art_id_df = comments_df.groupby('art_id').agg({'_id': lambda x: x.tolist(), 'topics': lambda x: x.tolist(), 'createdAt': lambda x: x.tolist()})

            # for each rows in art_id_df, if art_id is not in article_dict, make new entry with empty list for id and topics. else, append id and topics to existing list
            for i in range(len(art_id_df)):
                if art_id_df.index[i] not in article_dict:
                    article_dict[art_id_df.index[i]] = {'id': [], 'topics': [], 'createdAt': []}
                article_dict[art_id_df.index[i]]['id'] += art_id_df['_id'].values[i]
                article_dict[art_id_df.index[i]]['topics'] += art_id_df['topics'].values[i]
                article_dict[art_id_df.index[i]]['createdAt'] += art_id_df['createdAt'].values[i]
                
            with open(join(article_dir, 'article_dict.pkl'), 'wb') as handle:
                pickle.dump(article_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("temporary job finished")

# %%
parser = ArgumentParser()
parser.add_argument("operation", choices=["fit", "transform"])
parser.add_argument("--collection_name")
parser.add_argument("--sample_size")
parser.add_argument("--param")
parser.add_argument("--model_subpath")
parser.add_argument("--seed")
parser.add_argument("--random_state")
parser.add_argument("--mode", default='new')
parser.add_argument("--threshold")
parser.add_argument("--text", default='')
args = vars(parser.parse_args(sys.argv[1:]))
print(args)

# get current time for recording
now = datetime.now()


if args["operation"] == "fit":
    fit(args['collection_name'], args["sample_size"], json.loads(args["param"]), args["seed"], args["random_state"], args["mode"], args["threshold"], args["text"])
elif args["operation"] == "transform":
    transform(args["model_subpath"])
else:
    print(sys.argv)
    print("Usage")
    print("python fit-final-model.py fit")
    print("python fit-final-model.py transform <db_collection_name> <model_path>")

# get elapsed time
elapsed_time = datetime.now() - now
print('Time elapsed (hh:mm:ss.ms) {}'.format(elapsed_time))

exit(0)

# %%
# nohup python fit-final-models.py fit --collection_name Atlantic --sample_size 2000000 --param "[200, 80]" --seed 4 --random_state 19 --mode new --threshold 10 > fit_model_atlantic.out &
# nohup python fit-final-models.py fit --collection_name Breitbart --sample_size 2000000 --param "[225, 20]" --seed 3 --random_state 19 --mode new --threshold 10 > fit_model_Breitbart.out &
# nohup python fit-final-models.py fit --collection_name Thehill --sample_size 2000000 --param "[300, 80]" --seed 2 --random_state 19 --mode new --threshold 10 > fit_model_atlantic.out &
# nohup python fit-final-models.py fit --collection_name Gatewaypundit --sample_size 2000000 --param "[425, 30]" --seed 4 --random_state 19 --mode new --threshold 10 > fit_model_gwp.out &

# nohup python fit-final-models.py transform --model_subpath breitbart/breitbart_new_s3_r19_h225_u20_t10 > transform_breitbart.out &
# nohup python fit-final-models.py transform --model_subpath thehill/thehill_new_s2_r19_h300_u80_t10 > transform_thehill.out &
# nohup python fit-final-models.py transform --model_subpath atlantic/atlantic_new_s4_r19_h200_u80_t10 > transform_atlantic.out &

# nohup python fit-final-models.py transform --model_subpath motherjones/motherjones_new_s5_r19_h425_u90_t10 > transform_mj.out &
# nohup python fit-final-models.py transform --model_subpath gatewaypundit/gatewaypundit_new_s4_r19_h425_u30_t10 > transform_gwp.out &

# 15 02:00  : 77/130(br), 68/117 (th)
# 16 08:30  : 90/130(br), 74/117 (th)

# nohup python fit-final-models_temp.py transform --model_subpath breitbart/breitbart_new_s3_r19_h225_u20_t10 > transform_breitbart_temp.out &