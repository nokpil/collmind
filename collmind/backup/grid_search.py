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
import gc

import pymongo
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

from proj_utils import *


def fit(collection_name: str, sampling_size: int, params: List[int], seed: int, random_state: int, mode: str, threshold: int, text: str):

    pca = True
    start_time = time.time()

    start_date, end_date = DATE_RANGES_NEW[collection_name]
    hdb_min_cluster_size, umap_n_neighbors = params
    
    # check whether there are sampled_embeddings_dict for given seed
    if os.path.exists(join('result', collection_name.lower(), 'sampled_embeddings_dict', f'sampled_embeddings_dict_{seed}.pkl')):
        print(f'sampled_embeddings_dict_{seed}.pkl exists')
        with open(join('result', collection_name.lower(), 'sampled_embeddings_dict', f'sampled_embeddings_dict_{seed}.pkl'), 'rb') as f:
            sampled_embeddings_dict = pickle.load(f)
    else:
        print('no sampled_embeddings_dict found, sampling now')
        print(start_date, end_date, mode)
        sampled_embeddings_dict = collect_sampled_data_new(collection_name, sampling_size, seed=seed, start_date=start_date, end_date=end_date, 
    select_columns=["_id", "raw_message"], mode=mode, threshold=int(threshold))
        
        with open(f'{join("result", collection_name.lower(), "sampled_embeddings_dict", f"sampled_embeddings_dict_{seed}")}.pkl', 'wb') as f:
            pickle.dump(sampled_embeddings_dict, f)
    
    end_time = time.time()
    print(f"Time elapsed for sampling (and saving): {end_time - start_time} seconds")
    start_time = end_time
    
    init_embeddings = sampled_embeddings_dict["embeddings"].numpy()
    pca_embeddings = PCA(n_components=2, svd_solver='full').fit_transform(init_embeddings)
    pca_embeddings = np.array(pca_embeddings, copy=True)
    pca_embeddings /= np.std(pca_embeddings[:, 0]) * 10000
    
    topic_model, hdbscan_model, umap_model = init_bertopic(umap_n_components=2, umap_n_neighbors=umap_n_neighbors,
    hdbscan_min_cluster_size=hdb_min_cluster_size, calculate_probabilities=False, random_state=int(random_state), init=pca_embeddings)

    
    start_time = time.time()
    topic_model.fit(documents = sampled_embeddings_dict["raw_message"], embeddings = sampled_embeddings_dict["embeddings"].numpy())
    
    end_time = time.time()
    print(f"Time elapsed for fitting: {end_time - start_time} seconds")
    start_time = end_time
    
    model_subpath = f'{collection_name.lower()}_new_s{seed}_r{random_state}_h{hdb_min_cluster_size}_u{umap_n_neighbors}_t{threshold}{"_" + text if text else ""}'
    
    topics = topic_model.get_topic_info()
    count_sum = topics['Count'].sum()
    
    n_topics = len(topics)
    outlier_ratio_1 = topics.iloc[0]['Count'] / count_sum
    outlier_ratio_10 = (topics.iloc[0]['Count'] + topics.iloc[1]['Count']) / count_sum
    dbcv = topic_model.hdbscan_model.relative_validity_
    
    ctfidf_vectors = np.array(topic_model.c_tf_idf_.todense())
    topic_embeddings = np.array(topic_model.topic_embeddings_)
    names = topic_model.vectorizer_model.get_feature_names_out()

    end_time = time.time()
    print(f"Time elapsed for finishing: {end_time - start_time} seconds")
    start_time = end_time

    return topic_model, ctfidf_vectors, topic_embeddings, names, n_topics, outlier_ratio_1, outlier_ratio_10, dbcv

# %%
parser = ArgumentParser()
parser.add_argument("--collection_name")
parser.add_argument("--hlist")
parser.add_argument("--ulist")
parser.add_argument("--slist")
parser.add_argument("--rlist", default="[19]")
args = vars(parser.parse_args(sys.argv[1:]))
print(args)

# get current time for recording

## namespace main

from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

if __name__ == "__main__":

    sample_sizes = 2000000
    random_state = 19
    threshold = 10
    text = ''
    
    collection_name = args['collection_name']
    hlist = json.loads(args["hlist"])
    ulist = json.loads(args["ulist"])
    slist = json.loads(args["slist"])
    rlist = json.loads(args["rlist"])
    
    elapsed_time_list = np.zeros((len(hlist), len(ulist)))

    for i, h in enumerate(hlist):
        for j, u in enumerate(ulist):
            
            ctfidf_vectors_list = []
            topic_embeddings_list = []
            names_list = []
            
            for k, s in enumerate(slist):
                for l, r in enumerate(rlist):
                    
                    model_subpath = f'{collection_name.lower()}_new_s{s}_r{r}_h{h}_u{u}_t{threshold}{"_" + text if text else ""}'
                    
                    # check whether there is a file under the model_subpath
                    if os.path.exists(join('result', collection_name.lower(), model_subpath + '_fit.csv')):
                        print(f'{model_subpath} exists')
                        with open(f'{join("result", collection_name.lower(), model_subpath)}_output.pkl', 'rb') as f:
                            output = pickle.load(f)
                        ctfidf_vectors_list.append(output['ctfidf_vectors'])
                        topic_embeddings_list.append(output['topic_embeddings'])
                        names_list.append(output['names'])
                        print(f"n_topics : {output['n_topics']}, outlier_ratio : {output['outlier_ratio_1']}, {output['outlier_ratio_10']}, dbcv: {output['dbcv']}")
                        continue

                    now = datetime.now()
                    output = fit(collection_name, sample_sizes, [h, u], s, random_state, 'new', threshold, text)
                    elapsed_time = datetime.now() - now
                    
                    topic_model, ctfidf_vectors, topic_embeddings, names, n_topics, outlier_ratio_1, outlier_ratio_10, dbcv = output
                    topic_model.get_topic_info().to_csv(f'{join("result", collection_name.lower(), model_subpath)}_fit.csv')
                    
                    ctfidf_vectors_list.append(ctfidf_vectors)
                    topic_embeddings_list.append(topic_embeddings)
                    names_list.append(names)
                    
                    with open(f'{join("result", collection_name.lower(), model_subpath)}_output.pkl', 'wb') as f:
                        pickle.dump({'ctfidf_vectors': ctfidf_vectors, 'topic_embeddings': topic_embeddings, 'names': names, 'n_topics': n_topics, 'outlier_ratio_1': outlier_ratio_1, 'outlier_ratio_10': outlier_ratio_10, 'dbcv':dbcv}, f)
                    
                    print(f'[{h}, {u}, {s}] {datetime.now()} : ' + 'Time elapsed (hh:mm:ss.ms) {}'.format(elapsed_time))
                    print(f'n_topics : {n_topics}, outlier_ratio : {outlier_ratio_1}, {outlier_ratio_10}, dbcv: {dbcv}')
                    
                    del topic_model, ctfidf_vectors, topic_embeddings, names, n_topics, outlier_ratio_1, outlier_ratio_10, dbcv, output
                    gc.collect()
                    
            # Finished (h, u) pairs for all seeds, do matching!
            if len(slist)>1:
                cos_sim_list_ctfidf, matching_list_ctfidf = cos_sim_matching(ctfidf_vectors_list, names_list, mode='ctfidf')
                cos_sim_list_embeddings, matching_list_embeddings = cos_sim_matching(topic_embeddings_list, names_list, mode='embeddings')
                matching_dict_name = f"{collection_name.lower()}_new_s{'_'.join([str(seed) for seed in sorted(slist)])}_r{random_state}_h{h}_u{u}"

                with open(f'{join("result", collection_name.lower(), matching_dict_name)}_matching.pkl', 'wb') as f:
                    pickle.dump({'cos_sim_list_ctfidf': cos_sim_list_ctfidf, 'matching_list_ctfidf': matching_list_ctfidf, 'cos_sim_list_embeddings': cos_sim_list_embeddings, 'matching_list_embeddings': matching_list_embeddings}, f)

    exit(0)