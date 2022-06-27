"""
This is the module with functions to construct a sequence "embedding" - by assigning a threshold ("weight") to a
sequence "grid" for all biophysical feature statistics. The idea behind this design is that I found that the
distributions for theses biophysical feature statistics in PDB are normal distributions, and therefore if a residue's
inferred biophysical feature statistics is abnormally high or abnormally low, there is a higher possibility that they
will cause abnormal phase transition behavior. So I will put a statistics threshold ("weight") for each residue in each
feature, and by comparing the "grid" to the "weight" using rewarding/penalizing function, I calculated a overall score
for each feature for a given sequence. The training/optimization of the sequence "embedding" system is just the problem
of finding the best threshold ("weight") such that my positive samples and negative samples in the training set have
very different overall scores, which is performed in the "Trainer" module.
"""
import numpy as np
import pickle
import sys
from pathlib import Path
from random import sample, choice

# home_dir = str(Path.home())
# sys.path.append(home_dir + '/PycharmProjects/Predictor2.0')
from Config import processed_data_dir, grid_cache_dir
from Loader import load_grid, seq2grid
from Names import canonical_amino_acids, feature_tagABs, sub_feature_names, sub_feature_signs


#######################################
# Functions to mutate grid weights 
#######################################

# Load the combined human grid.
combined_human_grid = pickle.load(open(grid_cache_dir + '/combined_human_grid.pkl', 'rb'))

def init_grid_weight(pool_grid=combined_human_grid):
    """Initialize grid weight by random selection from a pool grid (default using human_grid)."""
    # print("INITIALIZE GRID WEIGHT ...")
    # create empty weight. For each amino acid in each feature, create a numpy array of 4 float numbers.
    # 4 float numbers: upper & lower threshold for the tagA & tagB grid score for this amino acid in this feature.
    weight = {"S2.SUMPI": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids},
              "S3.WATER.V2": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids},
              "S4.SSPRED": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids},
              "S5.DISO": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids},
              "S6.CHARGE.V2": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids},
              "S7.ELECHB.V2": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids},
              "S8.CationPi.V2": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids},
              "S9.LARKS.V2": {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids}}

    # randomly select a weight for each feature and each amino acid from the human protein grids.
    for feature in pool_grid:
        one_feature_pool_grid = pool_grid[feature]
        one_feature_weight = init_grid_weight_one_feature(one_feature_pool_grid)
        weight[feature] = one_feature_weight
    return weight


def init_grid_weight_one_feature(one_feature_pool_grid):
    """Initialize grid weight for one feature."""
    one_feature_weight = {aa: np.asarray([0.0, 0.0, 0.0, 0.0]) for aa in canonical_amino_acids}
    for aa in one_feature_pool_grid:
        one_feature_weight[aa][0] = np.random.choice(one_feature_pool_grid[aa][1])  # upper threshold, tagA
        one_feature_weight[aa][1] = np.random.choice(one_feature_pool_grid[aa][1])  # lower threshold, tagA
        one_feature_weight[aa][2] = np.random.choice(one_feature_pool_grid[aa][2])  # upper threshold, tagB
        one_feature_weight[aa][3] = np.random.choice(one_feature_pool_grid[aa][2])  # lower threshold, tagB
    return one_feature_weight


def mutate_grid_weight_one_feature(one_feature_weight, one_feature_pool_grid):
    """
    use genetic algorithm to mutate weight for one feature. This function is for the optimization of the embedding in the
    "Trainer" module.
    """
    # choose a random number of times to do mutation.
    for n in range(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])):
        # each time pick a random amino acid and threshold to mutate.
        random_aa = np.random.choice(list(canonical_amino_acids.keys()))
        random_thresh = np.random.choice([0, 1, 2, 3])

        if random_thresh < 2:   # tagA threshold mutation.
            one_feature_weight[random_aa][random_thresh] = np.random.choice(one_feature_pool_grid[random_aa][1])
        else:   # tagB threshold mutation.
            one_feature_weight[random_aa][random_thresh] = np.random.choice(one_feature_pool_grid[random_aa][2])
    return one_feature_weight


def mutate_grid_weight_all_feature(all_feature_weight, all_feature_pool_grid):
    """
    Mutate weight for all 8 features. This function is for the optimization of the embedding in the
    "Trainer" module.
    """
    # randomly choose a feature to mutate.
    random_feature = choice(list(feature_tagABs.keys()))
    one_feature_weight = all_feature_weight[random_feature]
    one_feature_pool_grid = all_feature_pool_grid[random_feature]

    # mutate the selected feature weight.
    all_feature_weight[random_feature] = mutate_grid_weight_one_feature(one_feature_weight, one_feature_pool_grid)

    # return the mutated feature as well as weight
    return random_feature, all_feature_weight 


def mutate_grid_weight_sub_feature(one_feature_weight, one_feature_pool_grid, sub_feature):
    """Mutate weight for sub-feature."""
    # choose a random number of times to do mutation.
    for n in range(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])):
        # each time pick a random amino acid and threshold to mutate.
        random_aa = np.random.choice(list(canonical_amino_acids.keys()))
        random_thresh = np.random.choice([0, 1]) if sub_feature==0 else np.random.choice([2, 3])

        if random_thresh < 2:   # tagA threshold mutation.
            one_feature_weight[random_aa][random_thresh] = np.random.choice(one_feature_pool_grid[random_aa][1])
        else:   # tagB threshold mutation.
            one_feature_weight[random_aa][random_thresh] = np.random.choice(one_feature_pool_grid[random_aa][2])
    return one_feature_weight


def mutate_grid_weight_16feature(_16feature_weight, _16feature_pool_grid):
    """Mutate weight for 16 features instead of 8 features."""
    # randomly choose a random feature (two sub-features) to mutate.
    random_feature = choice(list(feature_tagABs.keys()))
    one_feature_weight = _16feature_weight[random_feature]
    one_feature_pool_grid = _16feature_pool_grid[random_feature]

    # randomly choose a sub-feature to mutate.
    random_sub_feature = choice([0, 1])
    sub_feature_name = (random_feature, feature_tagABs[random_feature][random_sub_feature])

    # mutate the sub-feature weight and update the 16-feature weight.
    _16feature_weight[random_feature] = mutate_grid_weight_sub_feature(one_feature_weight=one_feature_weight,
                                                                       one_feature_pool_grid=one_feature_pool_grid, 
                                                                       sub_feature=random_sub_feature)

    # return the mutated feature as well as weight
    return sub_feature_name, _16feature_weight 
    

def mutate_grid_weight_selected_features(_16feature_weight, _16feature_pool_grid, selected_features):
    """mutate weight for selected sub-features.

    Args:
        _16feature_weight (dict): weight for 16 features.
        _16feature_pool_grid (dict): grid for 16 features.
        selected_features (list): selected list of sub-features.

    Returns:
        tuple: (mutated feature, 0 or 1), mutated _16feature_weight
    """
    # randomly choose a sub-feature from list to mutate.
    random_sub_feature = choice(selected_features)
    random_feature, sub_idx = sub_feature_names[random_sub_feature]

    # one-feature weight and grid.
    one_feature_weight = _16feature_weight[random_feature]
    one_feature_pool_grid = _16feature_pool_grid[random_feature]

    # mutate the sub-feature weight and update the 16-feature weight.
    _16feature_weight[random_feature] = mutate_grid_weight_sub_feature(one_feature_weight=one_feature_weight,
                                                                       one_feature_pool_grid=one_feature_pool_grid, 
                                                                       sub_feature=sub_idx)

    # return the mutated sub feature as well as weight
    return (random_feature, feature_tagABs[random_feature][sub_idx]), _16feature_weight 



##############################################
# Functions to calculate grid2weight scores
##############################################


def grid2weight_score(grid, weight, show_sub_feature=False):
    """for any given grid and weight, calculate scores for different features."""
    grid2weight_score = {"S2.SUMPI": None,
                         "S3.WATER.V2": None,
                         "S4.SSPRED": None,
                         "S5.DISO": None,
                         "S6.CHARGE.V2": None,
                         "S7.ELECHB.V2": None,
                         "S8.CationPi.V2": None,
                         "S9.LARKS.V2": None}
    for feature in grid2weight_score:
        one_feature_grid = grid[feature]
        one_feature_weight = weight[feature]
        if show_sub_feature:
            grid2weight_score[feature] = [0, 0]
            for i in range(2):
                grid2weight_score[feature][i] = grid2weight_score_sub_feature(one_feature_grid=one_feature_grid,
                                                                            one_feature_weight=one_feature_weight,
                                                                            sub_feature=i)
        else:
            grid2weight_score[feature] = grid2weight_score_one_feature(one_feature_grid, one_feature_weight)
    return grid2weight_score


def grid2weight_score_one_feature(one_feature_grid, one_feature_weight):
    """g2w score for one feature."""
    sum_score = 0
    for aa in one_feature_grid.keys():
        sum_score += np.where(one_feature_grid[aa][1] > one_feature_weight[aa][0], 1, 0).sum()
        sum_score -= np.where(one_feature_grid[aa][1] < one_feature_weight[aa][1], 1, 0).sum()
        sum_score += np.where(one_feature_grid[aa][2] > one_feature_weight[aa][2], 1, 0).sum()
        sum_score -= np.where(one_feature_grid[aa][2] < one_feature_weight[aa][3], 1, 0).sum()
    return sum_score


def grid2weight_score_sub_feature(one_feature_grid, one_feature_weight, sub_feature, sign=1):
    sum_score = 0
    if sub_feature == 0:
        for aa in one_feature_grid.keys():
            sum_score += np.where(one_feature_grid[aa][1] > one_feature_weight[aa][0], 1, 0).sum()
            sum_score -= np.where(one_feature_grid[aa][1] < one_feature_weight[aa][1], 1, 0).sum()
    else:
        for aa in one_feature_grid.keys():
            sum_score += np.where(one_feature_grid[aa][2] > one_feature_weight[aa][2], 1, 0).sum()
            sum_score -= np.where(one_feature_grid[aa][2] < one_feature_weight[aa][3], 1, 0).sum()
    return sum_score*sign


def grid2weight_residue_scores_one_feature(one_feature_grid, one_feature_weight):
    """g2w scores along sequence for one feature."""
    residue_scores = {'res_type':[], 'res_idx':[], 'res_reward':[], 'res_penalize':[], 'res_score': []}
    for aa in one_feature_grid.keys():
        res_len = one_feature_grid[aa].shape[1] # length of this residue's grid.
        res_types = [aa]*res_len
        res_idxs = one_feature_grid[aa][0, :].astype(int).tolist()
        lower_reward = np.where(one_feature_grid[aa][1] > one_feature_weight[aa][0], 1, 0)
        lower_penali = np.where(one_feature_grid[aa][1] < one_feature_weight[aa][1], 1, 0)
        upper_reward = np.where(one_feature_grid[aa][2] > one_feature_weight[aa][2], 1, 0)
        upper_penali = np.where(one_feature_grid[aa][2] < one_feature_weight[aa][3], 1, 0)
        rewards = (lower_reward + upper_reward).tolist()
        penalizes = (lower_penali + upper_penali).tolist()
        res_scores = (lower_reward + upper_reward - lower_penali - upper_penali).tolist()
        
        # update residue_scores.
        residue_scores['res_type'] += res_types
        residue_scores['res_idx'] += res_idxs
        residue_scores['res_reward'] += rewards
        residue_scores['res_penalize'] += penalizes
        residue_scores['res_score'] += res_scores
    
    # sort lists based on res_idx.
    residue_scores['res_idx'], \
    residue_scores['res_type'], \
    residue_scores['res_reward'], \
    residue_scores['res_penalize'], \
    residue_scores['res_score'] = zip(*sorted(zip(residue_scores['res_idx'],
                                                    residue_scores['res_type'],
                                                    residue_scores['res_reward'],
                                                    residue_scores['res_penalize'],
                                                    residue_scores['res_score'])))
    return residue_scores

def grid2weight_residue_scores_sub_feature(one_feature_grid, one_feature_weight, sub_feature, sign=1):
    """g2w scores along sequence for sub-feature."""
    residue_scores = {'res_type':[], 'res_idx':[], 'res_score': []}
    for aa in one_feature_grid.keys():
        res_len = one_feature_grid[aa].shape[1] # length of this residue's grid.
        res_types = [aa]*res_len
        res_idxs = one_feature_grid[aa][0, :].astype(int).tolist()
        tagA_reward = np.where(one_feature_grid[aa][1] > one_feature_weight[aa][0], 1, 0)
        tagA_penali = np.where(one_feature_grid[aa][1] < one_feature_weight[aa][1], 1, 0)
        tagB_reward = np.where(one_feature_grid[aa][2] > one_feature_weight[aa][2], 1, 0)
        tagB_penali = np.where(one_feature_grid[aa][2] < one_feature_weight[aa][3], 1, 0)
        tagA_scores = ((tagA_reward - tagA_penali)*sign).tolist()
        tagB_scores = ((tagB_reward - tagB_penali)*sign).tolist()
        res_scores = tagA_scores if sub_feature==0 else tagB_scores
        
        # update residue_scores.
        residue_scores['res_type'] += res_types
        residue_scores['res_idx'] += res_idxs
        residue_scores['res_score'] += res_scores
    
    # deal with edge case - empty residue_scores.
    if len(residue_scores['res_idx'])==0:
        return residue_scores
    
    # sort lists based on res_idx.
    residue_scores['res_idx'], \
    residue_scores['res_type'], \
    residue_scores['res_score'] = zip(*sorted(zip(residue_scores['res_idx'],
                                                    residue_scores['res_type'],
                                                    residue_scores['res_score'])))
    return residue_scores


def grid2weight_score_all_feature(grid, weight):
    """g2w score for all features."""
    sum_score = 0.0
    for feature in feature_tagABs:
        for i in range(2):
            one_feature_grid, one_feature_weight = grid[feature], weight[feature]
            sub_feature_score = grid2weight_score_sub_feature(one_feature_grid=one_feature_grid,
                                                            one_feature_weight=one_feature_weight,
                                                            sub_feature=i,
                                                            sign=sub_feature_signs[(feature, i)])
            sum_score += sub_feature_score
    return sum_score


def grid2weight_score_selected_features(grid, weight, selected_features):
    """g2w score for selected subfeatures.

    Args:
        grid (dict): sequence "grid".
        weight (dict): grid "weight".
        selected_features (list): list of selected sub-features.

    Returns:
        float: grid2weight score
    """
    sum_score = 0.0
    for sub_feature_name in selected_features:
        feat, sub_feat = sub_feature_names[sub_feature_name]
        # print(sub_feature_name, feat, sub_feat, sub_feature_signs[(feat, sub_feat)])
        one_feature_grid, one_feature_weight = grid[feat], weight[feat]
        sub_feature_score = grid2weight_score_sub_feature(one_feature_grid=one_feature_grid,
                                                        one_feature_weight=one_feature_weight,
                                                        sub_feature=sub_feat,
                                                        sign=sub_feature_signs[(feat, sub_feat)])
        sum_score += sub_feature_score 
    return sum_score


def grid2weight_residue_scores_all_feature(grid, weight):
    """g2w scores along sequence for all 8 features."""
    return {feature: grid2weight_residue_scores_one_feature(grid[feature], weight[feature]) for feature in feature_tagABs}


def grid2weight_residue_scores_all_sub_feature(grid, weight):
    """g2w scores along sequence for all 16 sub-features."""
    return {feature: [
        grid2weight_residue_scores_sub_feature(grid[feature], weight[feature], 0, sub_feature_signs[(feature, 0)]),
        grid2weight_residue_scores_sub_feature(grid[feature], weight[feature], 1, sub_feature_signs[(feature, 1)]),
    ] for feature in feature_tagABs}


def test():
    """Test function."""
    # weight = init_grid_weight()
    # with open(processed_data_dir+'/random_weight.pkl', 'wb') as f:
    #     pickle.dump(weight, f)
    weight = pickle.load(open(processed_data_dir + '/random_weight.pkl', 'rb'))
    
    # get grid for sequence "12asA_PDB"
    # grid = load_grid('12asA_PDB')
    # use the sequence of "12asA" directly instead
    grid = seq2grid(sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL")
    print(grid2weight_score(grid, weight))

if __name__ == "__main__":
    test()