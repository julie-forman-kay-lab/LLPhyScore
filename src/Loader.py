"""
This is the module with functions to convert protein sequence to a 'grid' data structure.
Each 'grid' is a nested python dictionary that has multiple levels.
1. In the first level, the dict keys are 8 different biophysical features; (dict values are the info stored in the next
level)
2. In the second level, the dict keys are 20 different amino acid residues; (dict values are the info stored in the
next level)
3. In the third level, each amino acid residue (keys in the second level) is mapped to a 3xN matrix (N can be different
for different residues in different sequences). The first column of the matrix is the position index of this residue;
The second column and third column is the inferred biophysical feature statistics for this residue (e.g. mean
electrostatic charge for glycine) extracted from PDB.
"""
import numpy as np
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

# home_dir = str(Path.home())
# sys.path.append(home_dir + '/PycharmProjects/Predictor2.0')
from Config import grid_cache_dir, score_db_dir
from Names import canonical_amino_acids, feature_tagABs
from Utils import GridScore, just_canonical


def load_grid(tag):
    """
    In order to save time on training, I have pre-calculated and locally saved the 'grids' for ~10000 sequences in my
    training set. This function load a pre-calculated 'grid' for a given sequence tag (name).
    :param tag: name of the sequence (in the training set)
    :return: grid of this sequence.
    """
    grid = {}
    for feature in feature_tagABs:
        feature_grid = pickle.load(open(grid_cache_dir + '/{}/{}.G'.format(feature, tag), 'rb'))
        grid[feature] = feature_grid
    return grid


def seq2grid(sequence):
    """
    For new sequence input, I need to calculate the 'grid'. This function converts a protein sequence to 'grid'.
    :param sequence: protein sequence.
    :return: grid: nested dictionary that represents a protein sequence by its inferred biophysical feature
    statistics from PDB.
    """
    assert just_canonical(sequence)
    grid = {}
    for feature in feature_tagABs:
        # For each biophysical feature, two statistical metrics were used (stored in feature_tagABs).
        tagA, tagB = feature_tagABs[feature][0], feature_tagABs[feature][1]
        # The inference of biophysical feature statistics for a given residue in sequence is by comparing this residue's
        # sequence context to all the sequences in PDB, and use the average PDB statistics as inferred value. This is
        # done by the GridScore class imported from utils.
        feature_grid_score = GridScore(dbpath=score_db_dir + "/{}".format(feature),
                                       tagA=tagA, tagB=tagB, max_xmer=40)
        _tag, res_scores = feature_grid_score.score_sequence(('_tag', sequence))
        # After calculating the biophysical feature statistics for the entire sequence, encapsulate them in a 'grid'.
        feature_grid = {}
        for aa_ in canonical_amino_acids:
            feature_grid[aa_] = [[], [], []]

        for r in res_scores:
            feature_grid[r.aa][0].append(r.ires)
            feature_grid[r.aa][1].append(r.A)
            feature_grid[r.aa][2].append(r.B)

        for aa_ in feature_grid:
            feature_grid[aa_] = np.asarray(feature_grid[aa_])

        grid[feature] = feature_grid
    return grid


def seqs2grids(sequences):
    """
    batch version of seq2grid function (time-efficient).
    For multiple sequences ({tag: seq} dictionary), convert them to grids ({tag: grid}).
    """
    grids = {tag: {} for tag in sequences}
    for feature in feature_tagABs:
        # load GridScore database for one feature
        # print("LOADING {} DATABASE".format(feature))
        tagA, tagB = feature_tagABs[feature][0], feature_tagABs[feature][1]
        feature_grid_score = GridScore(dbpath=score_db_dir + "/{}".format(feature),
                                       tagA=tagA, tagB=tagB, max_xmer=40)
        print("CONVERTING SEQUENCES TO {} GRIDS".format(feature))
        # generate the one-feature grids for all seqs.
        for tag, seq in tqdm(sequences.items()):
            _tag, res_scores = feature_grid_score.score_sequence((tag, seq))
            feature_grid = {}
            for aa_ in canonical_amino_acids:
                feature_grid[aa_] = [[], [], []]

            for r in res_scores:
                feature_grid[r.aa][0].append(r.ires)
                feature_grid[r.aa][1].append(r.A)
                feature_grid[r.aa][2].append(r.B)

            for aa_ in feature_grid:
                feature_grid[aa_] = np.asarray(feature_grid[aa_])

            grids[tag][feature] = feature_grid
    return grids


def test():
    """Test function."""
    test_grid = seq2grid(
        'MSFCSFFGGEVFQNHFEPGVYVCAKCGYELFSSRSKYAHSSPWPAFTETIHADSVAKRPEHNRSEALKVSCGKCGNGLGHEFLNDGPKPGQSRFIFSSSLKFVPKGKETSASQGH')
    print(test_grid)

if __name__ == "__main__":
    test()