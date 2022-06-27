"""
This is the module where a pretrained ML model is loaded to calculate the phase separation probability of any given
sequence or fasta file."""
import argparse
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from scipy.stats import percentileofscore

# home_dir = str(Path.home())
# sys.path.append(home_dir + '/PycharmProjects/Predictor2.0')
from Config import cleaned_data_dir, model_dir
from Loader import seq2grid, seqs2grids
from Model import grid2weight_score, grid2weight_score_sub_feature, grid2weight_residue_scores_sub_feature
from Names import feature_tagABs, feature_names, sub_feature_signs
from Utils import readfasta, FastaFile, just_canonical, make_linekey, get_closest_gridpoint, ResidueGridScores, \
                    GridScore, calculate_modified_zscore, smooth_scores

# final models using different weights.
final_models = {
    # weight trained against PDB.
    'PDB': pickle.load(open(model_dir+'/trained_weights.8FEATURES.{}.pkl'.format('PDB'), 'rb')),
    # weight trained against human.
    'human': pickle.load(open(model_dir+'/trained_weights.8FEATURES.{}.pkl'.format('human'), 'rb')), 
    # weight trained against human+PDB.
    'human+PDB': pickle.load(open(model_dir+'/trained_weights.8FEATURES.{}.pkl'.format('human+PDB'), 'rb')), 
}

# grid2weight scores of human proteome using different weights.
human_g2w_scores = {
    # weight trained against PDB.
    'PDB': pd.read_csv(cleaned_data_dir+'/human_g2w_scores_using_{}weight.csv'.format('PDB')),
    # weight trained against human.
    'human': pd.read_csv(cleaned_data_dir+'/human_g2w_scores_using_{}weight.csv'.format('human')),
    # weight trained against human+PDB.
    'human+PDB': pd.read_csv(cleaned_data_dir+'/human_g2w_scores_using_{}weight.csv'.format('human+PDB')),
}

# grid2weight scores of PDB proteome using different weights.
PDB_g2w_scores = {
    # weight trained against PDB.
    'PDB': pd.read_csv(cleaned_data_dir+'/PDB_g2w_scores_using_{}weight.csv'.format('PDB')),
    # weight trained against human.
    'human': pd.read_csv(cleaned_data_dir+'/PDB_g2w_scores_using_{}weight.csv'.format('human')),
    # weight trained against human+PDB.
    'human+PDB': pd.read_csv(cleaned_data_dir+'/PDB_g2w_scores_using_{}weight.csv'.format('human+PDB')),
}

# grid2weight scores of human+PDB proteome using different weights.
humanPDB_g2w_scores = {
    # weight trained against PDB.
    'PDB': pd.concat([human_g2w_scores['PDB'], PDB_g2w_scores['PDB']]).reset_index(drop=True),
    # weight trained against human.
    'human': pd.concat([human_g2w_scores['human'], PDB_g2w_scores['human']]).reset_index(drop=True),
    # weight trained against human+PDB.
    'human+PDB': pd.concat([human_g2w_scores['human+PDB'], PDB_g2w_scores['human+PDB']]).reset_index(drop=True),
}

# features in the predictor.
features = [
    'protein-water',
    'protein-carbon',
    'hydrogen bond (long-range)',
    'pi-pi (long-range)',
    'disorder (long)',
    'K-Beta similarity',
    'disorder (short)',
    'electrostatic (short-range)',
    '8-feature sum'
]

g2w_means_stds_PDB = {
    'PDB': pd.DataFrame([PDB_g2w_scores['PDB'].mean(), PDB_g2w_scores['PDB'].std()]).to_dict('list'),
    'human': pd.DataFrame([PDB_g2w_scores['human'].mean(), PDB_g2w_scores['human'].std()]).to_dict('list'),
    'human+PDB': pd.DataFrame([PDB_g2w_scores['human+PDB'].mean(), PDB_g2w_scores['human+PDB'].std()]).to_dict('list'),
}

g2w_means_stds_human = {
    'PDB': pd.DataFrame([human_g2w_scores['PDB'].mean(), human_g2w_scores['PDB'].std()]).to_dict('list'),
    'human': pd.DataFrame([human_g2w_scores['human'].mean(), human_g2w_scores['human'].std()]).to_dict('list'),
    'human+PDB': pd.DataFrame([human_g2w_scores['human+PDB'].mean(), human_g2w_scores['human+PDB'].std()]).to_dict('list'),
}

g2w_means_stds_humanPDB = {
    'PDB': pd.DataFrame([humanPDB_g2w_scores['PDB'].mean(), humanPDB_g2w_scores['PDB'].std()]).to_dict('list'),
    'human': pd.DataFrame([humanPDB_g2w_scores['human'].mean(), humanPDB_g2w_scores['human'].std()]).to_dict('list'),
    'human+PDB': pd.DataFrame([humanPDB_g2w_scores['human+PDB'].mean(), humanPDB_g2w_scores['human+PDB'].std()]).to_dict('list'),
}

# sr/lr tag meanings (16 features) for 8 physical factors.
feature_tagABs_new = {
    "S2.SUMPI": ['pi-pi (short-range)', 'pi-pi (long-range)'],
    "S3.WATER.V2": ['protein-water', 'protein-carbon'],
    "S4.SSPRED": ['sec. structure (helices)', 'sec. structure (strands)'],
    "S5.DISO": ['disorder (long)', 'disorder (short)'],
    "S6.CHARGE.V2": ['electrostatic (short-range)', 'electrostatic (long-range)'],
    "S7.ELECHB.V2": ['hydrogen bond (short-range)', 'hydrogen bond (long-range)'],
    "S8.CationPi.V2": ['cation-pi (short-range)', 'cation-pi (long-range)'],
    "S9.LARKS.V2": ['K-Beta similarity', 'K-Beta non-similarity']
    }

# signs +/- for 16 sub-features.
feature_signs = {
    'pi-pi (short-range)': 1,
    'pi-pi (long-range)': 1,
    'protein-water': 1,
    'protein-carbon': -1,
    'sec. structure (helices)': -1,
    'sec. structure (strands)': 1,
    'disorder (long)': 1,
    'disorder (short)': 1,
    'electrostatic (short-range)': 1,
    'electrostatic (long-range)': -1,
    'hydrogen bond (short-range)': 1,
    'hydrogen bond (long-range)': 1,
    'cation-pi (short-range)': -1,
    'cation-pi (long-range)': -1,
    'K-Beta similarity': 1,
    'K-Beta non-similarity': -1
}

##################################
# Load sequences and get scores
##################################

def get_g2w_scores(grids, weight, features):
    g2w_scores = {
    'tag': [], 
    'feature': [],
    'g2w_score': []
    }
    for tag, grid in grids.items():
        for feature in feature_tagABs_new:
            for i in range(2):
                feat = feature_tagABs_new[feature][i]
                if feat in features:
                    g2w_score = grid2weight_score_sub_feature(one_feature_grid=grid[feature], 
                                                              one_feature_weight=weight[feature],
                                                              sub_feature=i,
                                                              sign=feature_signs[feat])
                    g2w_scores['tag'].append(tag)
                    g2w_scores['feature'].append(feat)
                    g2w_scores['g2w_score'].append(g2w_score)
    df = pd.DataFrame(g2w_scores)
    
    # pivot table and get g2w score sum.
    new_df = pd.pivot_table(df, 
                            values='g2w_score', 
                            index=['tag'],
                            columns=['feature'])
    new_df['{}-feature sum'.format(len(features)-1)] = new_df.sum(axis=1)
    new_df = new_df.reset_index()
    new_df = new_df[['tag'] + features]
    new_df.columns.name = None

    return new_df


def get_zscores(g2w_scores, g2w_mean_std):
    g2w_mean = pd.DataFrame(pd.DataFrame(g2w_mean_std).iloc[0, :]).T.reset_index(drop=True)
    g2w_std = pd.DataFrame(pd.DataFrame(g2w_mean_std).iloc[1, :]).T.reset_index(drop=True)

    zscores = g2w_scores[list(g2w_mean_std.keys())].sub(g2w_mean.values, axis='columns').div(g2w_std.values, axis='columns')
    zscores['tag'] = g2w_scores['tag']
    zscores = zscores[g2w_scores.columns.tolist()]
    # print(zscores)
    return zscores


def get_percentile_ranking(g2w_scores, g2w_scores_control):
    pct_ranks = {feat:[] for feat in g2w_scores.columns.tolist() \
                 if feat in g2w_scores_control.columns.tolist()}
    for feat in pct_ranks.keys():
        feat_scores, feat_scores_control = g2w_scores[feat].values, g2w_scores_control[feat].values
        for fs in feat_scores:
            pct_rank = percentileofscore(a=feat_scores_control,
                                         score=fs)
            pct_ranks[feat].append(pct_rank)
    df = pd.DataFrame(pct_ranks)
    df['tag'] = g2w_scores['tag']
    df = df[g2w_scores.columns.tolist()]
    return df


def get_modified_zscores(g2w_scores, g2w_scores_control):
    modified_zscores = {feat:[] for feat in g2w_scores.columns.tolist() \
                        if feat in g2w_scores_control.columns.tolist()}
    for feat in modified_zscores.keys():
        feat_scores, feat_scores_control = g2w_scores[feat].values, g2w_scores_control[feat].values
        for fs in feat_scores:
            modified_zscore = calculate_modified_zscore(x=fs,
                                                        a=feat_scores_control)
            modified_zscores[feat].append(modified_zscore)
    df = pd.DataFrame(modified_zscores)
    df['tag'] = g2w_scores['tag']
    df = df[g2w_scores.columns.tolist()]
    return df


def get_g2w_scores_residue_level(grids, weight, features):
    g2w_scores_residue_level = {}
    for tag, grid in grids.items():
        g2w_scores_residue_level[tag] = {}
        for feature in feature_tagABs_new:
            for i in range(2):
                feat = feature_tagABs_new[feature][i]
                if feat in features:
                    g2w_score_residue_level = grid2weight_residue_scores_sub_feature(one_feature_grid=grid[feature],
                                                                                    one_feature_weight=weight[feature],
                                                                                    sub_feature=i,
                                                                                    sign=feature_signs[feat])
                    g2w_scores_residue_level[tag][feat] = g2w_score_residue_level

        # calculate residue-level score sum over features.
        sum_feat_name = '{}-feature sum'.format(len(features)-1)
        any_feat = features[0]
        g2w_scores_residue_level[tag][sum_feat_name] = {
            'res_type': g2w_scores_residue_level[tag][any_feat]['res_type'],
            'res_idx': g2w_scores_residue_level[tag][any_feat]['res_idx'],
            'res_score': np.array([0]*len(g2w_scores_residue_level[tag][any_feat]['res_idx'])),
        }
        for feat in features:
            if feat != sum_feat_name:
                g2w_scores_residue_level[tag][sum_feat_name]['res_score'] = tuple(
                    np.add(
                        g2w_scores_residue_level[tag][sum_feat_name]['res_score'],
                        g2w_scores_residue_level[tag][feat]['res_score']
                        )
                    )

    return g2w_scores_residue_level


def get_smoothed_scores_residue_level(g2w_scores_residue_level, smooth_window=50):
    """get the residue-level scores after smoothing out. For each residue, average its score with its n closest neighbors."""
    smoothed_scores_residue_level = {
        'tag': [],
        'feature': [],
        'residue_type':[],
        'residue_idx': [],
        'score': []
    }
    for tag in g2w_scores_residue_level:
        raw_scores_res_level = g2w_scores_residue_level[tag]
        for feat in features:
            raw_scores_feature_res_level = raw_scores_res_level[feat]['res_score']
            # raw_scores_feature_sum_res_level = raw_scores_res_level['{}-feature sum'.format(len(features)-1)]['res_score']
            smoothed_scores_res_level = smooth_scores(scores=raw_scores_feature_res_level,
                                                    smooth_window=smooth_window)
            # smoothed_scores_res_level = smooth_scores(scores=raw_scores_feature_sum_res_level,
            #                                           smooth_window=smooth_window)
            for i in range(len(smoothed_scores_res_level)):
                res_type = raw_scores_res_level['{}-feature sum'.format(len(features)-1)]['res_type'][i]
                res_idx = raw_scores_res_level['{}-feature sum'.format(len(features)-1)]['res_idx'][i]
                res_smoothed_score = smoothed_scores_res_level[i]
                smoothed_scores_residue_level['tag'].append(tag)
                smoothed_scores_residue_level['feature'].append(feat)
                smoothed_scores_residue_level['residue_type'].append(res_type)
                smoothed_scores_residue_level['residue_idx'].append(res_idx)
                smoothed_scores_residue_level['score'].append(res_smoothed_score)
    df = pd.DataFrame(smoothed_scores_residue_level)
    return df
    

def get_top_n_residues_sum_score(g2w_scores_residue_level, n=100):
    """get the sum score of n top-scored residues in a sequence."""
    top_n_res_sum_score = {
        'tag': [],
        # 'feature': [],
        'top_{}_residues_sum'.format(n): []
    }
    for tag in g2w_scores_residue_level:
        raw_scores_res_level = g2w_scores_residue_level[tag]
        raw_scores_feature_sum_res_level = raw_scores_res_level['{}-feature sum'.format(len(features)-1)]['res_score']
        top_n_sum = sum(sorted(raw_scores_feature_sum_res_level)[::-1][:n])
        top_n_res_sum_score['tag'].append(tag)
        top_n_res_sum_score['top_{}_residues_sum'.format(n)].append(top_n_sum)
    df = pd.DataFrame(top_n_res_sum_score)
    return df


##################################
# run Predictor 2.0 on fasta file
#################################

def run_fasta_scorer(args):
    seqs = readfasta(args.fasta)
    grids = seqs2grids(sequences=seqs)
    print('CALCULATING SCORES')

    # load datasets based on model_train_base.
    final_model = final_models[args.model_train_base]

    # load the base for calculating percentile ranking and z-score.
    # 2020.12.05 - set the calculating base as human proteome.
    if args.model_train_base=='human':
        g2w_scores_base = human_g2w_scores['human']
        g2w_mean_std_base = g2w_means_stds_human['human']
    elif args.model_train_base=='PDB':
        # g2w_scores_base = PDB_g2w_scores['PDB']
        # g2w_mean_std_base = g2w_means_stds_PDB['PDB']
        g2w_scores_base = human_g2w_scores['PDB']
        g2w_mean_std_base = g2w_means_stds_human['PDB']
    else:
        # g2w_scores_base = humanPDB_g2w_scores['human+PDB']
        # g2w_mean_std_base = g2w_means_stds_humanPDB['human+PDB']
        g2w_scores_base = human_g2w_scores['human+PDB']
        g2w_mean_std_base = g2w_means_stds_human['human+PDB']

    # sequence-level scoring.
    if args.score_type in ['raw', 'percentile', 'zscore', 'modified_zscore', 'all']:
        raw_scores = get_g2w_scores(grids, weight=final_model, features=features)
        # output raw scores.
        if args.score_type=='raw':
            output_scores = raw_scores
        # output percentile ranking.
        if args.score_type=='percentile':
            output_scores = get_percentile_ranking(g2w_scores=raw_scores, g2w_scores_control=g2w_scores_base)
        # output z-score.
        if args.score_type=='zscore':
            output_scores = get_zscores(g2w_scores=raw_scores, g2w_mean_std=g2w_mean_std_base)
        # output modified z-score.
        if args.score_type=='modified_zscore':
            output_scores = get_modified_zscores(g2w_scores=raw_scores, g2w_scores_control=g2w_scores_base)
        # output all score types.
        if args.score_type=='all':
            tag_col = raw_scores['tag']
            output_scores_raw = raw_scores.drop(['tag'], axis=1)
            output_scores_pct = get_percentile_ranking(g2w_scores=raw_scores, g2w_scores_control=g2w_scores_base)\
                                .drop(['tag'], axis=1)
            output_scores_zscore = get_zscores(g2w_scores=raw_scores, g2w_mean_std=g2w_mean_std_base)\
                                .drop(['tag'], axis=1)
            output_scores_mzscore = get_modified_zscores(g2w_scores=raw_scores, g2w_scores_control=g2w_scores_base)\
                                .drop(['tag'], axis=1)
            output_scores_all = {
                'raw': output_scores_raw,
                'pct': output_scores_pct,
                'zscore': output_scores_zscore,
                'm_zscore': output_scores_mzscore,
                }
            result = pd.concat(output_scores_all)
            output_scores = result.unstack(level=0)
            output_scores[('tag', '')] = tag_col
            output_scores = output_scores[[('tag', '')] + [col for col in output_scores if col != ('tag', '')]]

    # residue-level scoring.
    if args.score_type in ['top100_residue_sum', 'residue_level']:
        raw_scores_residue_level = get_g2w_scores_residue_level(grids, weight=final_model, features=features)
        if args.score_type=='top100_residue_sum':
            output_scores = get_top_n_residues_sum_score(g2w_scores_residue_level=raw_scores_residue_level,
                                                         n=100)
        if args.score_type=='residue_level':
            output_scores = get_smoothed_scores_residue_level(g2w_scores_residue_level=raw_scores_residue_level,
                                                              smooth_window=50)

    if args.output_filename:
        output_scores.to_csv(args.output_filename, index=False)
    else:
        output_scores = output_scores.set_index("tag")
        pd.set_option('display.max_rows', None)
        if args.score_type=="all":
            print(output_scores.stack(-2))
        elif args.score_type in ["raw", "percentile", "zscore", "modified_zscore"]:
            print(output_scores.stack(-1))
        elif args.score_type in ["top100_residue_sum", "residue_level"]:
            print(output_scores)
        pd.set_option('display.max_rows', 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run Predictor 2.0 on fasta files")
    parser.add_argument('--input file name', '-i', dest='fasta',
                        help=('input fasta file name'),
                        type=str)
    parser.add_argument('--output file name', '-o', dest='output_filename',
                        help=('output file name'),
                        type=str, default=None)
    parser.add_argument('--model training base', '-m', dest='model_train_base',
                        help=('training base of the model'),
                        type=str, default='human+PDB', choices=['human', 'PDB', 'human+PDB'])
    parser.add_argument('--score type', '-s', dest='score_type',
                        help=('type of score for 8 features'),
                        type=str, default='percentile', choices=['raw', 'percentile', 'zscore', 'modified_zscore', 'all',\
                                                                 'top100_residue_sum', 'residue_level'])
    args = parser.parse_args()
    run_fasta_scorer(args)