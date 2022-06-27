"""
This is the module with functions to train the sequence "embedding" - by finding the optimal threshold ("weight") for
each biophysical feature such that it maximize the separation of "grid-to-weight" score between positive and negative
training samples. The optimization process uses genetic algorithm and stochastic optimization; The optimized metric
is ROC AUC score.
"""
import numpy as np
import pickle
import sys
from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool
from random import sample, choice
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path

# home_dir = str(Path.home())
# sys.path.append(home_dir + '/PycharmProjects/Predictor2.0')
from Config import processed_data_dir, model_dir, grid_cache_dir
from Loader import load_grid, seqs2grids
from Model import grid2weight_score_one_feature, init_grid_weight_one_feature, mutate_grid_weight_one_feature
from Model import grid2weight_score_all_feature, init_grid_weight, mutate_grid_weight_all_feature, mutate_grid_weight_16feature
from Model import grid2weight_score_selected_features, mutate_grid_weight_selected_features
from Model import grid2weight_score_sub_feature, mutate_grid_weight_sub_feature
from Names import feature_tagABs


# Load the combined human grid.
combined_human_grid = pickle.load(open(grid_cache_dir + '/combined_human_grid.pkl', 'rb'))

# initialize pool worker.
pool = Pool(14)

####################################
#  Train-test set split functions 
####################################


def train_test_split_(pos_tags, neg_tags, pos_tags_groups, test_size=0.3):
    """The function to do train-test splitting for postitive and negative tags."""
    pos_train, pos_test, neg_train, neg_test = [], [], [], []

    # negative train-test split.
    neg_train, neg_test = train_test_split(neg_tags, test_size=test_size, random_state=2)

    # positive train_test split.
    test_sample_size = int(len(pos_tags)*test_size)
    train_sample_size = len(pos_tags) - test_sample_size 
    while len(pos_tags)>train_sample_size:
        rand_group_name = choice(pos_tags_groups)
        pos_test += [pos_tags[x] for x in range(len(pos_tags)) if pos_tags_groups[x]==rand_group_name]
        pos_tags = [i for i in pos_tags if i not in pos_test]
    pos_train = pos_tags
    return pos_train, pos_test, neg_train, neg_test


def Kfold_split(pos_tags, neg_tags, pos_tags_groups, n_splits=5):
    """The function to do K-fold splitting for positive and negative tags."""
    kf_tag_sets = [] # [([pos_train], [pos_test], [neg_train], [neg_test]), ([], [], [], []), ... ]
    
    # negative Kfold split.
    kf_neg = KFold(n_splits=n_splits, shuffle=True)#, random_state=2)
    neg_split = kf_neg.split(neg_tags)
    neg_trains, neg_tests = [], []
    for train_idx, test_idx in neg_split:
        neg_train, neg_test = [neg_tags[i] for i in train_idx], [neg_tags[j] for j in test_idx]
        neg_trains.append(neg_train)
        neg_tests.append(neg_test)
    
    # positive Kfold split.
    pos_splits = [[] for i in range(n_splits)]
    sample_size = len(pos_tags)
    i = 0
    while len(pos_tags)>0 and i<n_splits:
        if len(pos_splits[i])<sample_size//n_splits:
            rand_group_name = choice(pos_tags_groups)
            pos_splits[i]+=[pos_tags[x] for x in range(len(pos_tags)) if pos_tags_groups[x]==rand_group_name]
            pos_tags = [pos_tags[x] for x in range(len(pos_tags)) if pos_tags_groups[x]!=rand_group_name]
            pos_tags_groups = [group for group in pos_tags_groups if group!=rand_group_name]
        else:
            i += 1
    pos_trains, pos_tests = [], []
    for i in range(n_splits):
        pos_test = pos_splits[i]
        pos_train = []
        for j in range(n_splits):
            if j!=i:
                pos_train+=pos_splits[j]
        pos_trains.append(pos_train)
        pos_tests.append(pos_test)

    # pack train and test tags.
    kf_tag_sets = list(zip(pos_trains, pos_tests, neg_trains, neg_tests))
    return kf_tag_sets


############################
# AUC score calculators.
############################


def cal_auc_score(pos_g2w_scores, neg_g2w_scores):
    """The function to calculates the overall auc score for all positive and negative grid2weight scores."""
    scores = pos_g2w_scores + neg_g2w_scores
    labels = [1 for score in pos_g2w_scores] + [0 for score in neg_g2w_scores]
    auc_score = roc_auc_score(y_true=labels, y_score=scores, average='weighted')
    return auc_score


def pool_auc_score_one_feature(one_feature_pos_grids, one_feature_neg_grids, one_feature_weight):
    """Use pool.starmap to calculate auc scores for positive and negative grids against one-feature weight."""
    # sample neg_grids to reduce class imbalance.
    all_grids = one_feature_pos_grids + sample(one_feature_neg_grids, len(one_feature_neg_grids))

    # use pool.starmap to calculate one-feature grid2weight scores for all grids.
    zlist = list(zip(all_grids, repeat(one_feature_weight)))
    one_feature_g2w_scores = pool.starmap(grid2weight_score_one_feature, zlist)
    one_feature_pos_g2w_scores = one_feature_g2w_scores[:len(one_feature_pos_grids)]
    one_feature_neg_g2w_scores = one_feature_g2w_scores[len(one_feature_pos_grids):]

    # calculate auc_score.
    one_feature_auc_score = cal_auc_score(one_feature_pos_g2w_scores, one_feature_neg_g2w_scores)
    return one_feature_auc_score


def pool_auc_score_sub_feature(one_feature_pos_grids, one_feature_neg_grids, one_feature_weight, sub_feature):
    """Use pool.starmap to calculate auc score for positive and negative grids against sub-feature weight.

    Args:
        one_feature_pos_grids (list): list of one-feature positive grids.
        one_feature_neg_grids (list): list of one-feature negative grids.
        one_feature_weight (dict): dict of one-feature weight.
        sub_feature (int): 0 or 1

    Returns:
        float: AUC score of sub-feature
    """
    # combine positive and negative grids.
    all_grids = one_feature_pos_grids + one_feature_neg_grids

    # use pool.starmap to calculate one-feature grid2weight scores for all grids.
    zlist = list(zip(all_grids, repeat(one_feature_weight), repeat(sub_feature)))
    sub_feature_g2w_scores = pool.starmap(grid2weight_score_sub_feature, zlist)
    sub_feature_pos_g2w_scores = sub_feature_g2w_scores[:len(one_feature_pos_grids)]
    sub_feature_neg_g2w_scores = sub_feature_g2w_scores[len(one_feature_pos_grids):]

    # calculate auc_score.
    sub_feature_auc_score = cal_auc_score(sub_feature_pos_g2w_scores, sub_feature_neg_g2w_scores)
    return sub_feature_auc_score


def pool_auc_score_all_feature(pos_grids, neg_grids, weight):
    """Use pool.starmap to calculate auc scores for positive and negative grids against all-feature weight."""
    # sample neg_grids to reduce class imbalance.
    all_grids = pos_grids + sample(neg_grids, len(neg_grids))

    # use pool.starmap to calculate all-feature grid2weight scores for all grids.
    zlist = list(zip(all_grids, repeat(weight)))
    g2w_scores = pool.starmap(grid2weight_score_all_feature, zlist)
    pos_g2w_scores = g2w_scores[:len(pos_grids)]
    neg_g2w_scores = g2w_scores[len(pos_grids):]

    # calculate auc_score.
    auc_score = cal_auc_score(pos_g2w_scores, neg_g2w_scores)
    return auc_score


def pool_auc_score_selected_features(pos_grids, neg_grids, weight, selected_features):
    """Use pool.startmap to calculate auc scores for positive and negative grids against selected-features weight.

    Args:
        pos_grids (list): list of "grids" for LLPS-positive sequences.
        neg_grids (list): list of "grids" for LLPS-negative sequences.
        weight (dict): weight to calculate against.
        selected_features (list): list of sub-features to calculate on.

    Returns:
        float: auc score (0-1.0)
    """
    # sample neg_grids to reduce class imbalance.
    all_grids = pos_grids + sample(neg_grids, len(neg_grids))

    # use pool.starmap to calculate all-feature grid2weight scores for all grids.
    zlist = list(zip(all_grids, repeat(weight), repeat(selected_features)))
    g2w_scores = pool.starmap(grid2weight_score_selected_features, zlist)
    pos_g2w_scores = g2w_scores[:len(pos_grids)]
    neg_g2w_scores = g2w_scores[len(pos_grids):]

    # calculate auc_score.
    auc_score = cal_auc_score(pos_g2w_scores, neg_g2w_scores)
    return auc_score


###########################
#   One-feature Trainer.  #
###########################


def train_one_feature_model(pos_grids, neg_grids, init_weight, feature='S2.SUMPI', iter_1=100, iter_2=1000, log_filepath=False):
    """This function trains the model and finds the best weight using genetic algorithm."""
    # get one-feature grids and weight.
    one_feature_pos_grids = [g[feature] for g in pos_grids]
    one_feature_neg_grids = [g[feature] for g in neg_grids]
    one_feature_pool_grid = combined_human_grid[feature]

    # store training results.
    best_weight = init_weight[feature]
    best_auc_score = 0.0
    accepts_1 = []
    accepts_2 = []

    # record training acceptance in a file.
    if log_filepath:
        f = open(log_filepath, 'w+')

    # step1: fully random weights.
    for i in range(iter_1):
        # randomize weight.
        one_feature_weight = init_grid_weight_one_feature(one_feature_pool_grid)
        weight = deepcopy(one_feature_weight)

        # use pool_auc_score_one_feature to calculate the auc scores for one-feature pos and neg grids
        one_feature_auc_score = pool_auc_score_one_feature(one_feature_pos_grids,
                                                           one_feature_neg_grids,
                                                           weight)

        # update best_weight, best_auc_score, accepts_1.
        if one_feature_auc_score > best_auc_score:
            best_weight = weight
            best_auc_score = one_feature_auc_score
            accepts_1.append(1)
        else:
            accepts_1.append(0)
        output_str = "Training 1 iter: {:6d}; Current auc: {:20.15f}; Best auc: {:20.15f}; Acceptance: {:20.15f}; Feature: {:25}".\
            format(i, one_feature_auc_score, best_auc_score, np.mean(accepts_1), feature)
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')

    # step2: genetic algorithm.
    for i in range(iter_2):
        # store the copy of best_weight.
        weight = deepcopy(best_weight)

        # mutate weight using genetic algorithm.
        one_feature_weight = mutate_grid_weight_one_feature(weight, one_feature_pool_grid)

        # calculate auc_score.
        one_feature_auc_score = pool_auc_score_one_feature(one_feature_pos_grids,
                                                           one_feature_neg_grids,
                                                           one_feature_weight)

        # update best_weight, best_auc_score, accepts_1.
        if one_feature_auc_score > best_auc_score:
            best_weight = one_feature_weight
            best_auc_score = one_feature_auc_score
            accepts_2.append(1)
        else:
            accepts_2.append(0)
        output_str = "Training 2 iter: {:6d}; Current auc: {:20.15f}; Best auc: {:20.15f}; Acceptance: {:20.15f}; Feature: {:25}".\
            format(i, one_feature_auc_score, best_auc_score, np.mean(accepts_2), feature)
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')
    
    if log_filepath:
        f.close()
    return best_weight


###########################
#   All-feature Trainer.  #
###########################


def train_all_feature_model(pos_grids, neg_grids, init_weight,
                            iter_1=800, iter_2=8000, log_filepath=False):
    """This function trains the model and finds the best weight using genetic algorithm."""
    # store training results.
    best_weight = init_weight
    best_auc_score = 0.0
    accepts_1 = []
    accepts_2 = []

    # record training acceptance in a file.
    if log_filepath:
        f = open(log_filepath, 'w+')

    # step1: fully random weights.
    for i in range(iter_1):
        # randomize weight.
        weight = init_grid_weight(combined_human_grid)
        # weight = deepcopy(weight)

        # use pool_auc_score_one_feature to calculate the auc scores for one-feature pos and neg grids
        all_feature_auc_score = pool_auc_score_all_feature(pos_grids,
                                                           neg_grids,
                                                           weight)

        # update best_weight, best_auc_score, accepts_1.
        if all_feature_auc_score > best_auc_score:
            best_weight = weight
            best_auc_score = all_feature_auc_score
            accepts_1.append(1)
        else:
            accepts_1.append(0)

        output_str = "Training 1 iter: {:6d}; Current auc: {:20.15f}; Best auc: {:20.15f}; Acceptance: {:20.15f}; Feature: {:25}".\
            format(i, all_feature_auc_score, best_auc_score, np.mean(accepts_1), 'All')
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')

    # step2: genetic algorithm.
    for i in range(iter_2):
        # store the copy of best_weight.
        weight = deepcopy(best_weight)

        # mutate weight using genetic algorithm.
        mutated_feature, all_feature_weight = mutate_grid_weight_all_feature(weight, combined_human_grid)

        # calculate auc_score.
        all_feature_auc_score = pool_auc_score_all_feature(pos_grids,
                                                           neg_grids,
                                                           all_feature_weight)

        # update best_weight, best_auc_score, accepts_1.
        if all_feature_auc_score > best_auc_score:
            best_weight = all_feature_weight
            best_auc_score = all_feature_auc_score
            accepts_2.append(1)
        else:
            accepts_2.append(0)

        output_str = "Training 2 iter: {:6d}; Current auc: {:20.15f}; Best auc: {:20.15f}; Acceptance: {:20.15f}; Feature: {:25}".\
            format(i, all_feature_auc_score, best_auc_score, np.mean(accepts_2), mutated_feature)
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')
    
    if log_filepath:
        f.close()
    return best_weight


##############################
#   16 sub-feature Trainer.  #
##############################


def train_16feature_model(pos_grids, neg_grids, init_weight,
                            iter_1=800, iter_2=8000, log_filepath=False):
    """This model trains on 16 features instead of 8 features."""
    best_weight = init_weight
    init_auc_score = pool_auc_score_all_feature(pos_grids=pos_grids,
                                                neg_grids=neg_grids,
                                                weight=init_weight)
    best_auc_score = deepcopy(init_auc_score)
    accepts_1 = []
    accepts_2 = {
        (feature, sub_feature):[] for feature in feature_tagABs for sub_feature in feature_tagABs[feature]
        }

    # record training acceptance in a file.
    if log_filepath:
        f = open(log_filepath, 'w+')

    # step1: fully random weights.
    for i in range(iter_1):
        # randomize weight.
        weight = init_grid_weight(combined_human_grid)

        # use pool_auc_score_one_feature to calculate the auc scores for one-feature pos and neg grids
        all_feature_auc_score = pool_auc_score_all_feature(pos_grids,
                                                           neg_grids,
                                                           weight)

        # update best_weight, best_auc_score, accepts_1.
        if all_feature_auc_score > best_auc_score:
            best_weight = weight
            best_auc_score = all_feature_auc_score
            accepts_1.append(1)
        else:
            accepts_1.append(0)

        output_str = "Training 1 iter: {:6d}; Current auc: {:18.15f}; Best auc: {:18.15f}; Acceptance: {:18.15f}; Feature: {:15}; Sub-feature:{:15}".\
            format(i, all_feature_auc_score, best_auc_score, np.mean(accepts_1), 'All', 'All')
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')

    # step2: genetic algorithm.
    for i in range(iter_2):
        # store the copy of best_weight.
        weight = deepcopy(best_weight)

        # mutate weight using genetic algorithm.
        (mutated_feature, mutated_sub_feature), _16feature_weight = mutate_grid_weight_16feature(weight, combined_human_grid)

        # calculate auc_score.
        all_feature_auc_score = pool_auc_score_all_feature(pos_grids,
                                                           neg_grids,
                                                           _16feature_weight)

        # update best_weight, best_auc_score, accepts_2.
        if all_feature_auc_score > best_auc_score:
            best_weight = _16feature_weight
            best_auc_score = all_feature_auc_score
            accepts_2[(mutated_feature, mutated_sub_feature)].append(1)
        else:
            accepts_2[(mutated_feature, mutated_sub_feature)].append(0)

        output_str = "Training 2 iter: {:6d}; Current auc: {:18.15f}; Best auc: {:18.15f}; Acceptance: {:18.15f}; Feature: {:15}; Sub-feature:{:15}".\
            format(i, all_feature_auc_score, best_auc_score, np.mean(accepts_2[(mutated_feature, mutated_sub_feature)]), mutated_feature, mutated_sub_feature)
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')
    
    if log_filepath:
        f.close()
    return best_weight


#####################################
#   Sub-feature Trainer.  #
#####################################
 

def train_sub_feature_model(pos_grids, neg_grids, init_weight, feature, sub_feature, iter_1=100, iter_2=1000, log_filepath=False):
    """Trains on one sub-feature and return the trained weights.

    Args:
        pos_grids (list): list of positive grids.
        neg_grids (list): list of negative grids.
        init_weight (dict): initial weight.
        feature (str): feature to train on, e.g. "S2.SUMPI".
        sub_feature (int): 0 or 1.
        iter_1 (int, optional): Number of random mutations. Defaults to 100.
        iter_2 (int, optional): Number of genetic mutations. Defaults to 1000.
        log_filepath (bool/str, optional): File path to store training log. Defaults to False.

    Returns:
        dict: trained weight
    """

    # get one-feature grids and weight.
    one_feature_pos_grids = [g[feature] for g in pos_grids]
    one_feature_neg_grids = [g[feature] for g in neg_grids]
    one_feature_pool_grid = combined_human_grid[feature]

    # store training results.
    best_weight = init_weight[feature]
    init_auc = pool_auc_score_sub_feature(one_feature_pos_grids=one_feature_pos_grids,
                                          one_feature_neg_grids=one_feature_neg_grids,
                                          one_feature_weight=best_weight,
                                          sub_feature=sub_feature)
    best_auc = deepcopy(init_auc)
    accepts_1 = []
    accepts_2 = []

    # record training acceptance in a file.
    if log_filepath:
        f = open(log_filepath, 'w+')

    # step1: fully random weights.
    for i in range(iter_1):
        # randomize weight.
        one_feature_weight = init_grid_weight_one_feature(one_feature_pool_grid)

        # use pool_auc_score_one_feature to calculate the auc scores for one-feature pos and neg grids
        sub_feature_auc_score = pool_auc_score_sub_feature(one_feature_pos_grids=one_feature_pos_grids,
                                                           one_feature_neg_grids=one_feature_neg_grids,
                                                           one_feature_weight=one_feature_weight,
                                                           sub_feature=sub_feature)

        # update best_weight, best_auc_score, accepts_1.
        if (init_auc>=0.5 and sub_feature_auc_score>best_auc) or (init_auc<0.5 and sub_feature_auc_score<best_auc):
            best_weight = one_feature_weight
            best_auc = sub_feature_auc_score
            accepts_1.append(1)
        else:
            accepts_1.append(0)
        output_str = "Training 1 iter: {:6d}; Current auc: {:18.15f}; Best auc: {:18.15f}; Acceptance: {:18.15f}; Feature: {:15}; Sub-feature:{:15}".\
            format(i, sub_feature_auc_score, best_auc, np.mean(accepts_1), feature, feature_tagABs[feature][sub_feature])
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')

    # step2: genetic algorithm.
    for i in range(iter_2):
        # store the copy of best_weight.
        weight = deepcopy(best_weight)

        # mutate weight using genetic algorithm.
        one_feature_weight = mutate_grid_weight_sub_feature(weight, one_feature_pool_grid, sub_feature=sub_feature)

        # calculate auc_score.
        sub_feature_auc_score = pool_auc_score_sub_feature(one_feature_pos_grids=one_feature_pos_grids,
                                                           one_feature_neg_grids=one_feature_neg_grids,
                                                           one_feature_weight=one_feature_weight,
                                                           sub_feature=sub_feature)

        # update best_weight, best_auc_score, accepts_1.
        if (init_auc>=0.5 and sub_feature_auc_score>best_auc) or (init_auc<0.5 and sub_feature_auc_score<best_auc):
            best_weight = one_feature_weight
            best_auc = sub_feature_auc_score
            accepts_2.append(1)
        else:
            accepts_2.append(0)
        output_str = "Training 2 iter: {:6d}; Current auc: {:18.15f}; Best auc: {:18.15f}; Acceptance: {:18.15f}; Feature: {:15}; Sub-feature:{:15}".\
            format(i, sub_feature_auc_score, best_auc, np.mean(accepts_2), feature, feature_tagABs[feature][sub_feature])
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')
    
    if log_filepath:
        f.close()
    return best_weight


#####################################
#   Selected sub-features Trainer.  #
#####################################


def train_selected_features_model(pos_grids, neg_grids, init_weight,
                                    selected_features, iter_1=800, iter_2=8000, log_filepath=False):
    """train on selected list of sub-features.

    Args:
        pos_grids (list): positive "grids".
        neg_grids (list): negative "grids".
        init_weight (dict): initial weight.
        selected_features (list): list of selected sub-features.
        iter_1 (int, optional): number of first-round iterations. Defaults to 800.
        iter_2 (int, optional): number of second-round iterations. Defaults to 8000.
        log_filepath (bool, optional): file path to store log file. Defaults to False.

    Returns:
        dict: optimized weight.
    """
    best_weight = init_weight
    best_auc_score = 0.0
    accepts_1 = []
    accepts_2 = []

    # record training acceptance in a file.
    if log_filepath:
        f = open(log_filepath, 'w+')

    # step1: fully random weights.
    for i in range(iter_1):
        # randomize weight.
        weight = init_grid_weight(combined_human_grid)

        # use pool_auc_score_selected_features to calculate the auc scores for one-feature pos and neg grids
        selected_features_auc_score = pool_auc_score_selected_features(pos_grids,
                                                                    neg_grids,
                                                                    weight,
                                                                    selected_features)

        # update best_weight, best_auc_score, accepts_1.
        if selected_features_auc_score > best_auc_score:
            best_weight = weight
            best_auc_score = selected_features_auc_score
            accepts_1.append(1)
        else:
            accepts_1.append(0)

        output_str = "Training 1 iter: {:6d}; Current auc: {:18.15f}; Best auc: {:18.15f}; Acceptance: {:18.15f}; Feature: {:15}; Sub-feature:{:15}".\
            format(i, selected_features_auc_score, best_auc_score, np.mean(accepts_1), 'All', 'All')
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')

    # step2: genetic algorithm.
    for i in range(iter_2):
        # store the copy of best_weight.
        weight = deepcopy(best_weight)

        # mutate weight using genetic algorithm.
        (mutated_feature, mutated_sub_feature), selected_features_weight = mutate_grid_weight_selected_features(weight, combined_human_grid,
                                                                                                                selected_features)

        # calculate auc_score.
        selected_features_auc_score = pool_auc_score_selected_features(pos_grids,
                                                           neg_grids,
                                                           selected_features_weight,
                                                           selected_features)

        # update best_weight, best_auc_score, accepts_2.
        if selected_features_auc_score > best_auc_score:
            best_weight = selected_features_weight
            best_auc_score = selected_features_auc_score
            accepts_2.append(1)
        else:
            accepts_2.append(0)

        output_str = "Training 2 iter: {:6d}; Current auc: {:18.15f}; Best auc: {:18.15f}; Acceptance: {:18.15f}; Feature: {:15}; Sub-feature:{:15}".\
            format(i, selected_features_auc_score, best_auc_score, np.mean(accepts_2), mutated_feature, mutated_sub_feature)
        print(output_str)
        if log_filepath:
            f.write(output_str+'\n')
    
    if log_filepath:
        f.close()
    return best_weight



#####################################
#   Test Trainer Function  #
#####################################

def test():
    """Testing function."""
    features = ["S2.SUMPI", "S3.WATER.V2", "S4.SSPRED", "S5.DISO",
                "S6.CHARGE.V2", "S7.ELECHB.V2", "S8.CationPi.V2", "S9.LARKS.V2"]

    # load grids.
    pos_train_tags = pickle.load(open(processed_data_dir + '/training/tp_set_training_tags.pkl', 'rb'))
    pos_test_tags = pickle.load(open(processed_data_dir + '/test/tp_set_test_tags.pkl', 'rb'))
    neg_train_tags = pickle.load(open(processed_data_dir + '/training/pdb_set_training_tags.pkl', 'rb'))
    neg_test_tags = pickle.load(open(processed_data_dir + '/test/pdb_set_test_tags.pkl', 'rb'))
    pos_train_grids = [load_grid(tag) for tag in pos_train_tags]
    neg_train_grids = [load_grid(tag) for tag in neg_train_tags]
    pos_test_grids = [load_grid(tag) for tag in pos_test_tags] 
    neg_test_grids = [load_grid(tag) for tag in neg_test_tags]

    selected_features = [
        # "pipi (srpipi)",
        "pipi (lrpipi)",
        "water (Water)",
        "water (Carbon)",
        # "sec. structure (ssH)",
        # "sec. structure (ssE)",
        "disorder (disL)",
        "disorder (disS)",
        "charge (srELEC)",
        # "charge (lrELEC)",
        # "hydrogen bond (sr_hb)",
        "hydrogen bond (lr_hb)",
        # "cation pi (srCATPI)",
        # "cation pi (lrCATPI)",
        "K-Beta (larkSIM)",
        # "K-Beta (larkFAR)",
    ]

    # training selected sub-features competitively with signs using entire set.
    init_weight = pickle.load(open(model_dir+"/random_weight.pkl", "rb"))
    weights = train_selected_features_model(pos_grids=pos_train_grids, 
                                            neg_grids=neg_train_grids, 
                                            init_weight=init_weight,
                                            selected_features=selected_features, 
                                            iter_1=10, 
                                            iter_2=100, 
                                            log_filepath=False)
    
    val_str = ("AUC on TEST set: {}".format(pool_auc_score_selected_features(pos_grids=pos_test_grids,
                                                                        neg_grids=neg_test_grids,
                                                                        weight=weights,
                                                                        selected_features=selected_features)))
    print(val_str)


if __name__=='__main__':
    test()
exit()

# Training scripts
if __name__=='__main__':
    # Load positive and negative tags.
    pos_tags = pickle.load(open(processed_data_dir + '/total/tp_set_tags.pkl', "rb"))
    neg_tags = pickle.load(open(processed_data_dir + '/total/pdb_set_tags.pkl', 'rb'))\
            + pickle.load(open(processed_data_dir + '/total/human_set_tags.pkl', 'rb'))
    pos_tags_groups = pickle.load(open(processed_data_dir + '/total/tp_set_groups.pkl', "rb"))

    # train-test split
    pos_train_tags, pos_test_tags, neg_train_tags, neg_test_tags = train_test_split_(pos_tags=pos_tags,
                                                                                     neg_tags=neg_tags,
                                                                                     pos_tags_groups=pos_tags_groups,
                                                                                     test_size=0.3)

    # load positive and negative grids
    pos_grids = [load_grid(tag) for tag in pos_tags]
    neg_grids = [load_grid(tag) for tag in neg_tags]
    pos_train_grids = [load_grid(tag) for tag in pos_train_tags]
    neg_train_grids = [load_grid(tag) for tag in neg_train_tags]
    pos_test_grids = [load_grid(tag) for tag in pos_test_tags] 
    neg_test_grids = [load_grid(tag) for tag in neg_test_tags]

    # load initial weight
    random_weight = pickle.load(open(model_dir + '/random_weight.pkl', 'rb'))

    # feature names and sub-feature names
    features = ["S2.SUMPI", "S3.WATER.V2", "S4.SSPRED", "S5.DISO",
                "S6.CHARGE.V2", "S7.ELECHB.V2", "S8.CationPi.V2", "S9.LARKS.V2"]

    # training 8 feature separately (Jul 2020).
    # for feature in features:
    #     with open(processed_data_dir + '/trained_weights_v3.{}.pkl'.format(feature), 'wb') as f:
    #         weights = train_one_feature_model(pos_grids=pos_grids,
    #                                 neg_grids=neg_grids,
    #                                 init_weight=random_weight,
    #                                 feature=feature,
    #                                 iter_1=500,
    #                                 iter_2=10000,
    #                                 log_filepath='{}/training_log_v3.{}.txt'.format(processed_data_dir, feature))
    #         pickle.dump(weights, f)
    
    # training all 8 features at once (Jul 2020).
    # with open(processed_data_dir + '/trained_weights_v3.{}.pkl'.format('ALL.FEATURES'), 'wb') as f:
    #     weights = train_all_feature_model(pos_grids=pos_grids,
    #                             neg_grids=neg_grids,
    #                             init_weight=random_weight,
    #                             iter_1=500,
    #                             iter_2=10000,
    #                             log_filepath='{}/training_log_v3.{}.txt'.format(processed_data_dir, 'ALL.FEATURES'))
    #     pickle.dump(weights, f)

    # training 16 sub-features at once (Jul 2020).
    # with open(processed_data_dir + '/init_random_weights_v4.{}.pkl'.format('16FEATURES'), 'wb') as f:
    #     random_weight = init_grid_weight()
    #     pickle.dump(random_weight, f)

    # with open(processed_data_dir + '/trained_weights_v4.{}.pkl'.format('16FEATURES'), 'wb') as f:
    #     weights = train_16feature_model(pos_grids=pos_grids,
    #                                     neg_grids=neg_grids,
    #                                     init_weight=random_weight,
    #                                     iter_1=100,
    #                                     iter_2=20000,
    #                                     log_filepath='{}/training_log_v4.{}.txt'.format(processed_data_dir, '16FEATURES'))
    #     print("AUC on TEST set", pool_auc_score_all_feature(pos_grids=pos_grids_eval,
    #                                                         neg_grids=neg_grids_eval,
    #                                                         weight=weights))
    #     pickle.dump(weights, f)

    # training 16 sub-features separately (Jul 2020).
    # for feature in feature_tagABs:
    #     for i in range(2):
    #         sub_feature_name = feature_tagABs[feature][i]
    #         init_weight = pickle.load(open(processed_data_dir + '/training_16features_seperately_v5_Jul2020/init_weight_top_tail_5pct.pkl', 'rb'))
    #         log_filepath = '{}/training_16features_seperately_v5_Jul2020/training_log_v5.{}.{}.txt'.\
    #                                                 format(processed_data_dir, feature, sub_feature_name)

    #         with open(processed_data_dir + '/training_16features_seperately_v5_Jul2020/trained_weights_v5.{}.{}.pkl'.\
    #                 format(feature, sub_feature_name), 'wb') as f:
    #             weight = train_sub_feature_model(pos_grids=pos_grids,
    #                                               neg_grids=neg_grids,
    #                                               init_weight=init_weight,
    #                                               feature=feature,
    #                                               sub_feature=i,
    #                                               iter_1=500,
    #                                               iter_2=10000,
    #                                               log_filepath=log_filepath)

    #             one_feature_pos_grids_val = [g[feature] for g in pos_grids_eval]
    #             one_feature_neg_grids_val = [g[feature] for g in neg_grids_eval]
    #             val_str = ("AUC on TEST set: {}".format(pool_auc_score_sub_feature(one_feature_pos_grids=one_feature_pos_grids_val,
    #                                                                      one_feature_neg_grids=one_feature_neg_grids_val,
    #                                                                      one_feature_weight=weight,
    #                                                                      sub_feature=i)))
    #             with open(log_filepath, 'a+') as f0:
    #                 f0.write(val_str+'\n')
    #             pickle.dump(weight, f)

    # training 16 sub-features competitively with signs (Jul 2020).
    # init_weight = pickle.load(open(processed_data_dir + '/training_16features_seperately_v5_Jul2020/init_weight_top_tail_25pct.pkl', 'rb'))
    # weight_filepath = processed_data_dir + '/training_16features_competitively_v7_Jul2020/trained_weights_v7.{}.pkl'.format('16FEATURES')
    # log_filepath = processed_data_dir + '/training_16features_competitively_v7_Jul2020/training_log_v7.{}.txt'.format('16FEATURES')
    # with open(weight_filepath, 'wb') as f:
    #     weights = train_16feature_model(pos_grids=pos_grids,
    #                                     neg_grids=neg_grids,
    #                                     init_weight=init_weight,
    #                                     iter_1=0,
    #                                     iter_2=10500,
    #                                     log_filepath=log_filepath)
        
    #     val_str = ("AUC on TEST set: {}".format(pool_auc_score_all_feature(pos_grids=pos_grids_eval,
    #                                                                         neg_grids=neg_grids_eval,
    #                                                                         weight=weights)))
    #     print(val_str)
    #     with open(log_filepath, 'a+') as f0:
    #         f0.write(val_str+'\n')
    #     pickle.dump(weights, f)

    # training 16 sub-features competitively with signs, 5-fold cross-validation(Jul 2020).
    # kf_sets = Kfold_split(pos_tags=pos_tags,
    #                       neg_tags=neg_tags,
    #                       pos_tags_groups=pos_tags_groups,
    #                       n_splits=5)
    # for i, (pos_train_tags, pos_test_tags, neg_train_tags, neg_test_tags) in enumerate(kf_sets):
    #     pos_grids = [load_grid(tag) for tag in pos_train_tags]
    #     neg_grids = [load_grid(tag) for tag in neg_train_tags]
    #     pos_grids_eval = [load_grid(tag) for tag in pos_test_tags] 
    #     neg_grids_eval = [load_grid(tag) for tag in neg_test_tags]

    #     with open(processed_data_dir + '/training_16features_competitively_v9_Aug2020/init_random_weights.{}.fold_{}.pkl'.\
    #                 format('16FEATURES', i), 'wb') as f:
    #         init_weight = init_grid_weight()
    #         pickle.dump(init_weight, f)

    #     weight_filepath = processed_data_dir + '/training_16features_competitively_v9_Aug2020/trained_weights.{}.fold_{}.pkl'.\
    #                                         format('16FEATURES', i)
    #     log_filepath = processed_data_dir + '/training_16features_competitively_v9_Aug2020/training_log.{}.fold_{}.txt'.\
    #                                         format('16FEATURES', i)
    #     with open(weight_filepath, 'wb') as f:
    #         weights = train_16feature_model(pos_grids=pos_grids,
    #                                         neg_grids=neg_grids,
    #                                         init_weight=init_weight,
    #                                         iter_1=0,
    #                                         iter_2=5000,
    #                                         log_filepath=log_filepath)
            
    #         val_str = ("AUC on TEST set: {}".format(pool_auc_score_all_feature(pos_grids=pos_grids_eval,
    #                                                                             neg_grids=neg_grids_eval,
    #                                                                             weight=weights)))
    #         print(val_str)
    #         with open(log_filepath, 'a+') as f0:
    #             f0.write(val_str+'\n')
    #         pickle.dump(weights, f)

    # training 8 sub-features competitively with signs (Sep 2020).
    # load grids.
    pos_tags = pickle.load(open(processed_data_dir + '/tp_set_tags.pkl', 'rb'))
    neg_tags = pickle.load(open(processed_data_dir + '/pdb_set_tags.pkl', 'rb'))
    pos_tags_groups = pickle.load(open(processed_data_dir + '/tp_set_groups.pkl', 'rb'))
    # pos_train_tags, pos_test_tags, neg_train_tags, neg_test_tags = train_test_split_(pos_tags=pos_tags,
                                                                                        # neg_tags=neg_tags,
                                                                                        # pos_tags_groups=pos_tags_groups,
                                                                                        # test_size=0.3)
    pos_train_tags, pos_test_tags, neg_train_tags, neg_test_tags = pos_tags, pos_tags, neg_tags, neg_tags
    pos_grids = [load_grid(tag) for tag in pos_train_tags]
    neg_grids = [load_grid(tag) for tag in neg_train_tags]
    pos_grids_eval = [load_grid(tag) for tag in pos_test_tags] 
    neg_grids_eval = [load_grid(tag) for tag in neg_test_tags]

    selected_features = [
        # "pipi (srpipi)",
        "pipi (lrpipi)",
        "water (Water)",
        "water (Carbon)",
        # "sec. structure (ssH)",
        # "sec. structure (ssE)",
        "disorder (disL)",
        "disorder (disS)",
        "charge (srELEC)",
        # "charge (lrELEC)",
        # "hydrogen bond (sr_hb)",
        "hydrogen bond (lr_hb)",
        # "cation pi (srCATPI)",
        # "cation pi (lrCATPI)",
        "K-Beta (larkSIM)",
        # "K-Beta (larkFAR)",
    ]

    # training selected sub-features competitively with signs using entire set.
    with open(model_dir + '/init_random_weights.{}.pkl'.\
                format('{}FEATURES'.format(len(selected_features))), 'wb') as f:
        init_weight = init_grid_weight()
        pickle.dump(init_weight, f)

    weight_filepath = model_dir + '/trained_weights.{}.pkl'.\
                                        format('{}FEATURES'.format(len(selected_features)))
    log_filepath = model_dir + '/training_log.{}.txt'.\
                                        format('{}FEATURES'.format(len(selected_features)))
    with open(weight_filepath, 'wb') as f:
        weights = train_selected_features_model(pos_grids=pos_grids, 
                                                neg_grids=neg_grids, 
                                                init_weight=init_weight,
                                                selected_features=selected_features, 
                                                iter_1=0, 
                                                iter_2=10000, 
                                                log_filepath=log_filepath)
        
        val_str = ("AUC on TEST set: {}".format(pool_auc_score_selected_features(pos_grids=pos_grids,
                                                                            neg_grids=neg_grids,
                                                                            weight=weights,
                                                                            selected_features=selected_features)))
        print(val_str)
        with open(log_filepath, 'a+') as f0:
            f0.write(val_str+'\n')
        pickle.dump(weights, f) 
