"""
This is the module to visualize all clustered sequences.
"""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
# import pyparsing
import matplotlib.pyplot as plt
import json
import sys

# Data directory configuration
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_dir)
sys.path.append(parent_dir+'/src')
# print(sys.path)

from src.Config import original_data_dir

data_path = original_data_dir + '/clustering/clustering_data'

file_path = data_path + '/clustering_result.txt' # clustering result file.
sequences = json.load(open(data_path + '/COMPLETE_SEQUENCES_UNTAGGED_20191126.json', 'r')) # {name:sequence} dict.

def is_related(name1: str, name2: str):
    """
    Function to see if two protein sequence names are related (coming from same mother protein).
    :param name1: First protein sequence's name
    :param name2: Second protein sequence's name
    :return: True or False
    """
    # This is a list of words to rule out that might cause confusion.
    flag_words = ['GFP', 'YFP', 'SNAP', 'IDR', 'PTB', 'His6', 'LC', 'PBM', 'mCherry', 'LCD', 'Baldwin2',
                  'LCR', 'ABD', 'YtoF', 'RGG', 'Fawzi', 'FL',
                  'NTD', 'CTD', 'pep', 'MBP', 'delHRD', 'WT', 'Hyman', 'RBD', 'PLD', 'IDR1', 'Chilkoti', ]
    l1 = name1.split('_')
    l2 = name2.split('_')
    common = list(set(l1).intersection(l2) - set(flag_words))
    if len(common) > 0:
        return True
    else:
        return False

def cluster_df(cluster_file):
    """
    generate cluster dataframe from the clustering result file.
    :param cluster_file: file path
    :return: dataframe with two columns: cluster_num, seq_name
    """
    df = pd.read_csv(cluster_file, sep='\s+', names=['clunum1', 'clunum2', 'name1', 'name2', 'jsimilarity'], )
    cluster_df = df[['clunum1', 'name1']].drop_duplicates().sort_values(by=['clunum1', 'name1']).copy()
    return cluster_df

def cluster_dict(cluster_file):
    """
    generate cluster dictionary from the clustering result file.
    :param cluster_file: file path
    :return: dictionary {cluster_num: [seq_name1, seq_name2, ...]}.
    """
    cls_df = cluster_df(cluster_file)
    cluster_dict = cls_df.groupby('clunum1')['name1'].apply(list).to_dict()
    return cluster_dict

def cluster_sizes(cluster_file):
    """Analyze the sizes of clusters."""
    cls_df = cluster_df(cluster_file=cluster_file)
    cls_sizes = cls_df.clunum1.sort_values(ascending=False).value_counts()
    return cls_sizes

def intercluster_similarity(cluster_file):
    """Returns the similarity values between different cluster's sequences."""
    sim_df = pd.read_csv(cluster_file, sep='\s+', names=['clunum1', 'clunum2', 'name1', 'name2', 'jsimilarity'], )
    inter_cls_sim_df = sim_df[(sim_df.clunum1 != sim_df.clunum2)].sort_values(by='jsimilarity', ascending=False).copy()
    return inter_cls_sim_df

def intracluster_similarity(cluster_file):
    """Returns the similarity values between same cluster's sequences."""
    sim_df = pd.read_csv(cluster_file, sep='\s+', names=['clunum1', 'clunum2', 'name1', 'name2', 'jsimilarity'], )
    intra_cls_sim_df = sim_df[(sim_df.clunum1 == sim_df.clunum2)].sort_values(by='jsimilarity', ascending=False).copy()
    return intra_cls_sim_df

def abnormal_intercluster_similarity(cluster_file, similarity_thresh=0.5):
    """
    Return abnormally high simialarity values for intercluster sequence pairs.
    :param cluster_file: file path.
    :param similarity_thresh: 0.0-1.0
    :return: df.
    """
    inter_cls_df = intercluster_similarity(cluster_file=cluster_file)
    abn_df = inter_cls_df.loc[inter_cls_df.jsimilarity>similarity_thresh].sort_values(by='jsimilarity', ascending=False)
    return abn_df

def abnormal_intracluster_similarity(cluster_file, similarity_thresh=0.5):
    """
    Return abnormally high simialarity values for intracluster sequence pairs.
    :param cluster_file: file path.
    :param similarity_thresh: 0.0-1.0
    :return: df.
    """
    intra_cls_df = intracluster_similarity(cluster_file=cluster_file)
    abn_df = intra_cls_df.loc[intra_cls_df.jsimilarity<similarity_thresh].sort_values(by='jsimilarity', ascending=False)
    return abn_df

# reads cluster file. # show the cluster size distribution.
cls_df = cluster_df(cluster_file=file_path)
cls_dict = cluster_dict(cluster_file=file_path)
# cls_df.to_csv(data_path + '/cluster_sequence_20191204.txt', index=False)
# with open(data_path + '/cluster_sequence_20191204.json', 'w+') as fp:
#     json.dump(cls_dict, fp)

# show the cluster size distribution.
cls_sizes = cluster_sizes(cluster_file=file_path)
ax = cls_sizes.plot.pie(autopct=lambda x: "{}".format(int(cls_sizes.sum()*x/100)))
# ax = cls_sizes.plot.bar()
ax.set_title('Sizes of clusters \n(outside are cluster names; inside are cluster sizes)')
plt.show()

# similarity values for all-, inter- and intra- cluster sequence pairs.
all_cls_sim = pd.read_csv(file_path, sep='\s+', names=['clunum1', 'clunum2', 'name1', 'name2', 'jsimilarity'], )
diff_cls_sim = intercluster_similarity(cluster_file=file_path)
same_cls_sim = intracluster_similarity(cluster_file=file_path)

# show the similarity value distribution for differnt groups.
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
all_cls_sim.jsimilarity.hist(bins=100, ax=axes[0])
diff_cls_sim.jsimilarity.hist(bins=100, ax=axes[1])
same_cls_sim.jsimilarity.hist(bins=100, ax=axes[2])
fig.suptitle('jsimilarity value distribution')
axes[0].set_title('all-cluster sequence pairs')
axes[1].set_title('inter-cluster sequence pairs')
axes[2].set_title('intra_cluster sequence pairs')
plt.show()

# abnormal similarity values for inter- and intra- cluster sequence pairs.
diff_cls_high_sim = abnormal_intercluster_similarity(cluster_file=file_path, similarity_thresh=0.5)
same_cls_low_sim = abnormal_intracluster_similarity(cluster_file=file_path, similarity_thresh=0.5)

# for inter-cluster sequence pairs, no high (>0.5) jsimilarity value was found.
print(diff_cls_high_sim)
print(len(diff_cls_high_sim))

# for intra-cluster sequence pairs, we can see that all of the low (<0.5) jsimilarity value actually comes from related
# sequences: They either come from different parts of the same/similar protein (FET protein family, hnRNPA), or are
# different ELP variants.
print(same_cls_low_sim)
# same_cls_low_sim.to_csv(data_path+'/sequences_sameclusterlowjsimilarity.txt', index=False)
# same_cls_low_sim['related'] = same_cls_low_sim.apply(lambda x: is_related(x.name1, x.name2), axis=1)
# print(same_cls_low_sim[~same_cls_low_sim.related])

# look at in which cluster are the abnormal intra-cluster sequence pairs found.
# print(same_cls_low_sim.clunum1.value_counts())
ax = same_cls_low_sim.clunum1.value_counts().plot.pie(autopct=lambda x: "{}".format(int(cls_sizes.sum()*x/100)))
ax.set_title('clusters where abnormally low intra-cluster sequence pairs were found.'
             '\n(outside are cluster names; inside are occurences)')
plt.show()

# the largest three clusters.
largest_cluster_nums = cls_sizes.index[:3]
print(cls_df.loc[cls_df.clunum1.isin(largest_cluster_nums)])
