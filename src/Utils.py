"""Utility functions and classes."""
import pickle
import sys
from pathlib import Path
from os.path import exists
from math import sqrt
import numpy as np

# home_dir = str(Path.home())
# sys.path.append(home_dir + '/PycharmProjects/Predictor2.0')
from Names import canonical_amino_acids


def just_canonical(seq):
    """Function to load sequence and make sure all amino acids are canonical."""
    for n in range(len(seq)):
        if not (seq[n] in canonical_amino_acids):# and seq[n]!='U':
            # print(seq[n])
            return False
    return True


def make_tag(tagline, ftype):
    """Function to make a name tag from the FASTA header ('>...') for different datasets (training, human and pdb)."""
    if ftype=="TrainingSet" or ftype=="TRAIN" or ftype=="TRAINING":
        return tagline.split()[0][1:]
    if ftype == "Human" or ftype=="HUMAN":
        return tagline.split('|')[1] + "_" + tagline.split('|')[2].split('_HUMAN')[0] + "_HUMAN"
    if ftype == "PDB" or ftype=='pdb':
        return tagline.split()[0][1:] + "_PDB"


def readfasta(filepath):
    """Function to read a fasta file into a {tag:seq} dictionary."""
    f = open(filepath, 'r')
    seqs = {}
    onfasta, seq = "", ""
    for l in f.readlines():
        if not l.strip():
            continue
        if l[0] == '>':
            if onfasta != "" and just_canonical(seq):
                seqs[onfasta] = seq
            seq = ""
            onfasta = l.split()[0][1:]
        else:
            seq += l.split()[0]
    seqs[onfasta] = seq
    return seqs

class FastaFile(object):
    """Class to read, check and interpret fasta files."""
    def __init__(self, filepath: str, filetype: str):
        self.filepath = filepath
        self.filetype = filetype

        self.fastadict = self.read_fasta()
        self.missing = self.check_missing()

    def read_fasta(self):
        """reads fasta file and returns a dict {tagline: sequence}."""
        filepath = self.filepath
        filetype = self.filetype

        seqdict = {}
        fp = open(filepath, 'r')
        lines = fp.readlines()
        tag = ""
        seq = ""
        for line in lines:
            if not line.strip():
                continue
            if line[0] == '>':
                if tag != "" and just_canonical(seq):
                    seqdict[tag] = seq
                seq = ""
                tag = make_tag(line, self.filetype)
            else:
                seq += line.split()[0]
            if tag != "" and just_canonical(seq):
                seqdict[tag] = seq
        fp.close()
        return seqdict

    def keys(self):
        """returns all the tagline names in the fasta file."""
        return list(self.fastadict.keys())

    def check_missing(self):
        """check if all '>' headers in fasta files were included in the fastadict."""
        missing = []
        inputseqs = open(self.filepath).readlines()
        for i in inputseqs:
            if i != "\n":
                if i[0] == ">":
                    onfasta = make_tag(i, self.filetype)
                    if onfasta not in self.fastadict:
                        missing.append(onfasta)
        return missing

def make_linekey(zscore):
    """Function to round up/down zscores. This is a function needed by the GridScore class to calculate statistics."""
    round_z = round(zscore * 2) / 2
    if round_z <= 12.0 or round_z >= -8.0:
        return round_z

    if round_z > 12.0:
        round_z = 12.0
    if round_z < -8.0:
        round_z = -8.0

    return round_z


def get_closest_gridpoint(rxGRID, fv1, fv2):
    """
    Function to find the closest point in a grid. This is a function needed by the GridScore class to match a residue's
    sequence context to a similar context in PDB's sequences.
    """
    dlist = []
    for g1 in rxGRID.keys():
        fg1 = float(g1)
        for g2 in rxGRID[g1].keys():
            fg2 = float(g2)
            dist = sqrt((fv1 - fg1) ** 2 + (fv2 - fg2) ** 2)
            dlist.append([dist, g1, g2])
    dlist.sort()
    return dlist[0][1], dlist[0][2]


class ResidueGridScores:
    """Class to store grid scores for a sequence."""
    def __init__(self, ires: int, aa: str, n_outside_grid: int, A: float, B: float):
        self.ires = int(ires)                       # index of a residue.
        self.aa = str(aa)                           # name of a residue.
        self.n_outside_grid = int(n_outside_grid)   # number of residues that could not be matched to PDB.
        self.A = float(A)                           # biophysical feature statistics A.
        self.B = float(B)                           # biophysical feature statistics B.

    def __str__(self):
        return "%4i %1s %2i %10.8f %10.8f" % (self.ires, self.aa, self.n_outside_grid, self.A, self.B)


class GridScore:
    """
    This is the class to infer biophysical feature statistics from PDB for a given sequence. I analyzed the structures
    of the entire PDB (resolution>2.0A), and saved average frequency and standard deviation for a total of 8 biophysical
    feature for any X_N_Y residue pair. Here X is residue A, N is the number of residues in between (0-40), and Y is
    residue B. Then I break the input sequence to many X_N_Y residue pairs, and match these residue pairs to PDB.
    """
    def __init__(self, dbpath: str, tagA: str, tagB: str, max_xmer: int):
        self.dbpath = dbpath    # path of the file storing PDB statistics.
        self.tagA = tagA  # typically the name of the "short range" observation frequency
        self.tagB = tagB  # typically the name of the "long range" observation frequency
        self.PairFreqDB = {}

        # store the statistics (average and std) for all residue pairs.
        assert (exists(self.dbpath + "/PCON2.FREQS.wBOOTDEV"))
        ffile = open(self.dbpath + "/PCON2.FREQS.wBOOTDEV").readlines()
        for f in ffile:
            l = f.split()

            pair_key = l[0]  # This is the amino acid pair key (ie: H_3_P means HxxxP)

            self.PairFreqDB[pair_key] = {}

            for triplet in range((len(l) - 1) // 3):  # data is stores in blocks of 3 ie: "X_srpipi 0.035311 0.001289"
                ptype = l[1 + triplet * 3]  # name of frequency statistic associated with a given pair ie: X_srpipi
                freq = float(l[1 + triplet * 3 + 1])  # frequency
                sdev = float(l[1 + triplet * 3 + 2])  # standard deviation (from bootstrap analysis)

                self.PairFreqDB[pair_key][ptype] = [freq, sdev]

        # Each residue's biophysical impact are considered to be limited to its context (40 residues). Therefore I also
        # need to take their average statistics in different context lengths.
        assert (exists(self.dbpath + "/STEP6_PICKLES/SC_GRIDS.pickle4"))
        self.ZGridDB = pickle.load(open(self.dbpath + "/STEP6_PICKLES/SC_GRIDS.pickle4", 'rb'))

        self.xmers = []
        for n in range(max_xmer):
            self.xmers.append(n + 1)
        self.min_xmer = 1
        self.max_xmer = max_xmer

        self.AvgSdevDB = {}
        for xmer in self.xmers:
            self.AvgSdevDB[xmer] = {"LR": {}, "SR": {}}
            assert (exists(self.dbpath + "/STEP4_AVGnSDEVS/PCON2.xmer" + str(xmer)))
            PairAvgSdevFile = open(self.dbpath + "/STEP4_AVGnSDEVS/PCON2.xmer" + str(xmer)).readlines()
            for f in PairAvgSdevFile:
                l = f.split()
                self.AvgSdevDB[xmer]["SR"][l[0]] = [float(l[2]), float(l[3])]  # AVG, SDEV
                self.AvgSdevDB[xmer]["LR"][l[0]] = [float(l[5]), float(l[6])]  # AVG, SDEV

    def score_mseq_onexmer(self, mseq, use_xmer):
        """This function calculates the average statistics for one xmer (x is the window size to do summing on)."""
        # print mseq

        LR_SUMF = 0.0
        SR_SUMF = 0.0
        TOTF = 0.0

        TOTLR = 0.0
        TOTSR = 0.0

        M1 = mseq[use_xmer]

        for P in range(use_xmer):  # +1):
            Xp = P
            POS1 = use_xmer - 1 - P

            key1 = mseq[Xp] + "_" + str(POS1) + "_" + M1

            LR_SUMF += self.PairFreqDB[key1]["Y_" + self.tagB][0] * (1.0 / self.PairFreqDB[key1]["Y_" + self.tagB][1])
            SR_SUMF += self.PairFreqDB[key1]["Y_" + self.tagA][0] * (1.0 / self.PairFreqDB[key1]["Y_" + self.tagA][1])
            TOTLR += (1.0 / self.PairFreqDB[key1]["Y_" + self.tagB][1])
            TOTSR += (1.0 / self.PairFreqDB[key1]["Y_" + self.tagA][1])

        for P in range(use_xmer):
            Xp = use_xmer + 1 + P
            POS1 = P

            key1 = M1 + "_" + str(POS1) + "_" + mseq[Xp]

            LR_SUMF += self.PairFreqDB[key1]["X_" + self.tagB][0] * (1.0 / self.PairFreqDB[key1]["X_" + self.tagB][1])
            SR_SUMF += self.PairFreqDB[key1]["X_" + self.tagA][0] * (1.0 / self.PairFreqDB[key1]["X_" + self.tagA][1])
            TOTLR += (1.0 / self.PairFreqDB[key1]["X_" + self.tagB][1])
            TOTSR += (1.0 / self.PairFreqDB[key1]["X_" + self.tagA][1])

        LR_FREQ = LR_SUMF / TOTLR
        SR_FREQ = SR_SUMF / TOTSR

        return SR_FREQ, LR_FREQ

    def score_mseq_smart(self, mseq):

        assert (len(mseq) % 2 == 1)

        midpoint = int((len(mseq) - 1) / 2)

        assert (midpoint <= self.max_xmer)

        scores = {}

        frequency_sum_A = 0.0  # Typically the short range frequency (tagA)
        frequency_sum_B = 0.0  # Typically the long range frequency (tagB)

        denom_total_A = 0.0
        denom_total_B = 0.0

        mid_res_aa = mseq[midpoint]
        for P in range(midpoint):
            # Values for when mid_res is the X in X_n_Y
            cterm_position = midpoint + 1 + P
            gap_length = P
            pairdb_key = mid_res_aa + "_" + str(gap_length) + "_" + mseq[cterm_position]

            frequency_sum_A += self.PairFreqDB[pairdb_key]["X_" + self.tagA][0] \
                               * (1.0 / self.PairFreqDB[pairdb_key]["X_" + self.tagA][1])

            frequency_sum_B += self.PairFreqDB[pairdb_key]["X_" + self.tagB][0] \
                               * (1.0 / self.PairFreqDB[pairdb_key]["X_" + self.tagB][1])

            denom_total_A += (1.0 / self.PairFreqDB[pairdb_key]["X_" + self.tagA][1])
            denom_total_B += (1.0 / self.PairFreqDB[pairdb_key]["X_" + self.tagB][1])

            # Values for when mid_res is the Y in X_n_Y
            nterm_position = midpoint - P - 1
            gap_length = P
            key1Y = mseq[nterm_position] + "_" + str(gap_length) + "_" + mid_res_aa

            frequency_sum_A += self.PairFreqDB[key1Y]["Y_" + self.tagA][0] \
                               * (1.0 / self.PairFreqDB[key1Y]["Y_" + self.tagA][1])

            frequency_sum_B += self.PairFreqDB[key1Y]["Y_" + self.tagB][0] \
                               * (1.0 / self.PairFreqDB[key1Y]["Y_" + self.tagB][1])

            denom_total_A += (1.0 / self.PairFreqDB[key1Y]["Y_" + self.tagA][1])
            denom_total_B += (1.0 / self.PairFreqDB[key1Y]["Y_" + self.tagB][1])

            ######SAVE WINDOW SCORE

            if self.min_xmer <= P + 1 <= self.max_xmer:
                final_frequency_A = frequency_sum_A / denom_total_A
                final_frequency_B = frequency_sum_B / denom_total_B

                scores[P + 1] = [final_frequency_A, final_frequency_B]

        return scores

    def score_sequence(self, tagseq):
        """This is the function to generate biophysical statistical scores for the entire sequence."""
        tag = tagseq[0]
        seqref = tagseq[1]

        scoredata = []

        for nt in range(len(seqref)):

            if nt > 0 and nt + 1 < len(seqref):
                mid_res_aa = seqref[nt]

                grid_counts = [0.0, 0.0, 0.0]
                n_outside_grid = 0

                use_xmer = self.min_xmer
                mseq = seqref[nt - use_xmer:nt + use_xmer + 1]
                while len(mseq) == 2 * use_xmer + 1 and mseq.find('X') == -1 and use_xmer <= self.max_xmer:
                    use_xmer += 1
                    mseq = seqref[nt - use_xmer:nt + use_xmer + 1]
                use_xmer -= 1
                mseq = seqref[nt - use_xmer:nt + use_xmer + 1]

                avg_zscore_sr = 0.0
                avg_zscore_lr = 0.0
                zscore_n = 0.0

                if len(mseq) >= 2 * self.min_xmer + 1 and mseq.find('X') == -1:
                    all_xmer_scores = self.score_mseq_smart(mseq)
                    # try:
                    #     all_xmer_scores = self.score_mseq_smart(mseq)
                    # except:
                    #     print(tag)
                    #     print(seqref)
                    #     exit()
                    #     break

                    for XN in range(self.max_xmer - self.min_xmer + 1):
                        xmer = XN + self.min_xmer

                        if xmer in all_xmer_scores:

                            freq_sr = all_xmer_scores[xmer][0]
                            freq_lr = all_xmer_scores[xmer][1]

                            zscore_sr = (freq_sr - self.AvgSdevDB[xmer]["SR"][mid_res_aa][0]) \
                                        / self.AvgSdevDB[xmer]["SR"][mid_res_aa][1]
                            zscore_lr = (freq_lr - self.AvgSdevDB[xmer]["LR"][mid_res_aa][0]) \
                                        / self.AvgSdevDB[xmer]["LR"][mid_res_aa][1]

                            SK = make_linekey(zscore_sr)
                            LK = make_linekey(zscore_lr)

                            avg_zscore_sr += zscore_sr
                            avg_zscore_lr += zscore_lr
                            zscore_n += 1.0

                            matched_to_grid = False
                            if (SK in self.ZGridDB[mid_res_aa][xmer]):
                                if (LK in self.ZGridDB[mid_res_aa][xmer][SK]):
                                    GD = self.ZGridDB[mid_res_aa][xmer][SK][LK]
                                    grid_counts[0] += GD[0]  # Total
                                    grid_counts[1] += GD[1]  # SR
                                    grid_counts[2] += GD[2]  # LR
                                    matched_to_grid = True

                            if not matched_to_grid:
                                n_outside_grid += 1
                                SK, LK = get_closest_gridpoint(self.ZGridDB[mid_res_aa][xmer], zscore_sr, zscore_lr)
                                GD = self.ZGridDB[mid_res_aa][xmer][SK][LK]
                                grid_counts[0] += GD[0]  # Total
                                grid_counts[1] += GD[1]  # SR
                                grid_counts[2] += GD[2]  # LR

                freq_by_grid_sr = 0.0
                freq_by_grid_lr = 0.0
                if grid_counts[0] > 0:
                    freq_by_grid_sr = grid_counts[1] / grid_counts[0]
                    freq_by_grid_lr = grid_counts[2] / grid_counts[0]
                    avg_zscore_sr /= zscore_n
                    avg_zscore_lr /= zscore_n

                ostring = "%20s %5i %1s %2i %12s %8.5f %8.5f %12s %8.5f %8.5f " % \
                          ("temp", nt, mid_res_aa, n_outside_grid, self.tagA,
                           avg_zscore_sr, freq_by_grid_sr, self.tagB,
                           avg_zscore_lr, freq_by_grid_lr)

                rscore = ResidueGridScores(nt + 1, mid_res_aa, n_outside_grid, freq_by_grid_sr, freq_by_grid_lr)
                scoredata.append(rscore)
        return tag, scoredata



def calculate_modified_zscore(x, a):
    """calculate the modified z-score of x against a (to account for skewed normal distribution).
    
    Args:
        x (float): raw value.
        a (np.array): background values to calculate modified z-score.

    Returns:
        [float]: modified z-score for x against a.

    """
    a_median = np.median(a)
    # print("Median: ", a_median)
    a_mean = np.mean(a)
    # print("Mean: ", a_mean)
    median_abs_dev = np.median(np.absolute(a-a_median))
    # print("median absolute deviation (MAD)", median_abs_dev)
    mean_abs_dev = np.mean(np.absolute(a-a_mean))
    # print("mean absolute deviation (MeanAD)", mean_abs_dev)

    if median_abs_dev==0:
        return (x-a_median)/(1.253314*mean_abs_dev)
    else:
        return (x-a_median)/(1.486*median_abs_dev)


def smooth_scores(scores, smooth_window):
    """get smoothed list of scores given a scores list/array and a smooth window.

    Args:
        scores ([list, np.array]): input scores.
        smooth_window ([int]): smooth window.
    Returns:
        np.array
    """
    if len(scores) < 2 * smooth_window:
        return scores
    else:
        half_window = smooth_window//2
        new_scores = [0]*len(scores)
        for i in range(len(scores)):
            if i < half_window:
                new_scores[i] = np.mean(scores[:smooth_window])
            elif i+1 > len(scores) - half_window:
                new_scores[i] = np.mean(scores[:-smooth_window])
            else:
                new_scores[i] = np.mean(scores[i-half_window:i+half_window])
        return new_scores