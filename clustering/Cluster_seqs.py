"""
This is the module to perform hieararchical clustering of LLPS-positive sequences based on pair-wise Jaccard similarity.
"""
import random
from math import sqrt

import os
import sys
import json
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import kneighbors_graph

# Data directory configuration
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_dir)
sys.path.append(parent_dir+'/src')
# print(sys.path)

from src.Config import original_data_dir

data_path = original_data_dir + '/clustering/clustering_data'

jaccard_file = data_path + '/SeqDistances.mult.CompleteSequences.20191213.txt'


def jaccard_distance_matrix(jaccard_similarity_file, jaccard_similarity_kind='smallest'):
    """
    convert jaccard_similarity datafile to jaccard distance matrix.
    reads: location of jaccard similarity file generated from Dipeptide.py
    returns: jaccard distance matrix (n x n array), matrix index (list of n sequence names)
    :param jaccard_similarity_file: The Jaccard file generated from Dipeptide.py
    :param jaccard_similarity_kind: The kind of jaccard similarity to use. Default jsimilarity from smallest window.
    :return: jaccard distance matrix (n x n array); index (sequence names) of the jaccard distance matrix (list of n
     sequence names)
    """

    df = pd.read_csv(jaccard_file, sep='\s+', names=['jall', 'jsmall', 'name1', 'name2'])

    # make two columns of "jaccard distance" jaccard distance = (1 - jaccard similarity)
    df['jdall'] = 1.0 - df['jall']
    df['jdsmall'] = 1.0 - df['jsmall']

    # sorted name list
    name_order = df.name1.unique().tolist()

    # add same-protein pair jaccard distances as 0.
    dict_toapp = {'jall': [], 'jsmall': [], 'name1': [], 'name2': [], 'jdall': [], 'jdsmall': []}
    for name in name_order:
        dict_toapp['jall'].append(1)
        dict_toapp['jsmall'].append(1)
        dict_toapp['name1'].append(name)
        dict_toapp['name2'].append(name)
        dict_toapp['jdall'].append(0)
        dict_toapp['jdsmall'].append(0)
    df_toapp = pd.DataFrame(dict_toapp)
    df = pd.concat((df, df_toapp), axis=0)

    # sort jaccard distance values by protein pair names
    df = df.sort_values(by=['name1', 'name2']).reset_index(drop=True)
    assert df.name1.drop_duplicates().tolist() == df.name1.tolist()[
                                                  ::len(
                                                      name_order)], "The protein names in pairs sholud be in same order"
    assert df.name2.drop_duplicates().tolist() == df.name2.tolist()[
                                                  :len(
                                                      name_order)], "The protein names in pairs sholud be in same order"
    assert df.name1.drop_duplicates().tolist() == df.name2.drop_duplicates().tolist(), "The protein names in pairs " \
                                                                                       "sholud be in same order "

    # use sorted protein names as data index.
    data_index = df.name1.drop_duplicates().tolist()

    # choose "smallest/all comparison jaccard distance" as the data matrix.
    if jaccard_similarity_kind == 'smallest':
        data_matrix = df.jdsmall.values
    elif jaccard_similarity_kind == "all":
        data_matrix = df.jdall.values
    elif jaccard_similarity_kind == "average":
        data_matrix = df[['jdsmall', 'jdall']].mean(axis=1).values
    data_dim = sqrt(data_matrix.shape[0])
    assert data_dim == int(data_dim), "data dimension should be integar"
    data_matrix = data_matrix.reshape(int(data_dim), int(data_dim))
    return data_matrix, data_index


def connectivity_matrix(jdist_matrix, n_neighbors=20):
    """
    calculate a connectivity (constraint) matrix from distance matraix.
    :param jdist_matrix: jaccard distance matrix (nxn array)
    :return: connectivity matrix (nxn array)

    """
    neigh = kneighbors_graph(jdist_matrix, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    return neigh


def cluster_by_jdist(jdist_matrix, jmatrix_index, connectivity, linkage_method='complete', jsim_threshold=0.4, ):
    """
    cluster protein sequences by the jaccard distance matrix from jaccard_distance_matrix(jaccard_file).
    :param jdist_matrix: jaccard distance matrix (n x n array)
    :param jmatrix_index: index (sequence names) of the jaccard distance matrix (list of n sequence names)
    :param linkage_method: which linkage criterion to use. "complete", "average", "single"
    :param jsim_threshold: the jaccard similarity threshold to cluster on (above which, clusters will be merged) 
    :return: A list of (seq_name, cluster_label); A dendrogram plot figure showing the clustering result.
    """

    jdist_threshold = 1.0 - jsim_threshold

    data_dists = squareform(jdist_matrix)

    # plot dendrogram to choose threshold for the hierarchical clustering
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    linkage_matrix = linkage(data_dists, method=linkage_method)
    dn = dendrogram(linkage_matrix, ax=ax, orientation='top', labels=jmatrix_index, distance_sort='descending',
                    show_leaf_counts=True)
    ax.set_title('hierarchical clustering dedrogram of all protein sequences')
    ax.set_xticklabels(labels=jmatrix_index, fontsize=8, rotation=45)
    ax.set_ylabel('Jaccard Distance (1 - Jaccard Similarity)')
    # plt.show()

    # do hierarchical clustering
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, connectivity=connectivity,
                                    linkage=linkage_method, distance_threshold=jdist_threshold, ).fit(jdist_matrix)
    # print(model.n_clusters_, model.n_connected_components_, model.children_)

    clustered_index = list(zip(jmatrix_index, model.labels_))
    return clustered_index, fig


def cluster_DBSCAN(jdist_matrix, jmatrix_index, jsim_threshold=0.4, min_samples=1):
    jdist_threshold = 1.0 - jsim_threshold

    # do DBSCAN clustering model
    model = DBSCAN(eps=jdist_threshold, min_samples=min_samples, metric='precomputed', ).fit(jdist_matrix)
    clustered_index = list(zip(jmatrix_index, model.labels_))
    return clustered_index


def select_val_set(cluster_result_file, sequence_file, percent=0.2):
    """
    select validation set of sequences based on the sequence clustering result.
    :param cluster_result_file: The clustering result file. Each line has two columns: (cluster_name, seq_name)
    :param sequence_file: The file storing all sequences. dictionary of (seq_name: sequence)
    :param percent: percentage of the sequence names chosen to be validation set.
    :return: two dataframes - training & validation set, (cluster_name, seq_name, sequence)
    """
    assert os.path.exists(cluster_result_file)
    assert os.path.exists(sequence_file)

    # construct a dataframe for all sequences with three columns: cluster_name, sequence_name, sequence.
    cluster_result = pd.read_csv(cluster_result_file, )  # cluster_result in dataframe format.
    with open(sequence_file, 'r') as fp:    # sequence dictionary: {sequence name: sequence}
        sequence_dict = json.load(fp)
    cluster_result['sequence'] = cluster_result['name1'].map(sequence_dict) # add sequence column to the cluster_result.

    assert cluster_result.notna().all().all()   # no missing values.

    # collect train and validation set in two lists.
    total_set = cluster_result['name1'].unique().tolist()  # all protein names
    all_clusters = cluster_result['clunum1'].unique().tolist()  # all cluster names
    total_size = len(total_set)  # total set size
    val_size = int(total_size * percent)  # validation set size
    val_set = []
    train_set = deepcopy(total_set)
    while len(val_set) < val_size:
        val_cluster = random.choice(all_clusters)
        all_clusters.remove(val_cluster)
        add_seq_names = cluster_result.loc[cluster_result.clunum1==val_cluster, 'name1'].tolist()
        val_set.extend(add_seq_names)   # extend validation set each time during iteration over random cluster number.
        train_set = list(set(train_set) - set(add_seq_names))   # reduce training set each time during
                                                                # iteration over random cluster number.
    # return the results in dataframe format.
    val_df = cluster_result.loc[cluster_result.name1.isin(val_set)]
    train_df = cluster_result.loc[cluster_result.name1.isin(train_set)]
    return val_df, train_df

if __name__ == "__main__":
    jdist_matrix, index = jaccard_distance_matrix(jaccard_similarity_file=jaccard_file, jaccard_similarity_kind='average')
    # connect_matrix = connectivity_matrix(jdist_matrix=jdist_matrix, n_neighbors=300)
    # connect_matrix = pd.read_csv(data_path + '/connectivity.tpseqs.mat', index_col=0).fillna(0).values

    clustered_index, fig = cluster_by_jdist(jdist_matrix=jdist_matrix, jmatrix_index=index, connectivity=None,
                                            linkage_method='single', jsim_threshold=0.5, )
    # clustered_index = cluster_DBSCAN(jdist_matrix=jdist_matrix, jmatrix_index=index, jsim_threshold=0.6, min_samples=2)
    # validation_set = select_val_set(clustered_index=clustered_index, percent=0.2)

    # jsimilarity dataframe.
    jsim_df = pd.DataFrame((1 - jdist_matrix), index=index, columns=index)
    # jsim_df.to_csv(data_path+'/TRUE_POSITIVE_SEQUENCES_JSIMARITY.txt',)

    # cluster dictionary.
    cluster_d = {}
    for seq, clu_n in clustered_index:
        cluster_d.setdefault(clu_n, []).append(seq)
    # for key in sorted(cluster_d):
    #     print(key, len(cluster_d[key]))
    print(cluster_d)

    # with open(data_path + '/clustering_result.txt', 'w+') as fp:
    #     for clu_n in sorted(list(cluster_d.keys())):
    #         cluster = cluster_d[clu_n]
    #         for clu_n1 in sorted(list(cluster_d.keys())):
    #             cluster1 = cluster_d[clu_n1]
    #             print(len(cluster), len(cluster1))
    
    #             for seq1 in cluster:
    #                 for seq2 in cluster1:
    #                     sim = jsim_df.at[seq1, seq2]
    #                     fp.write("{:<10} {:<10} {:35} {:35} {:3.6f} \n".format(clu_n, clu_n1, seq1, seq2, sim))
    #                     print("{:<10} {:<10} {:35} {:35} {:8.5} \n".format(clu_n, clu_n1, seq1, seq2, sim))