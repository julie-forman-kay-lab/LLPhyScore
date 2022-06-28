"""
This is the module to calculate the pair-wise Jaccard similarity for all sequence-sequence pairs within any given
sequence or fasta file.
"""
import numpy as np
import sys

# Data directory configuration
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_dir)
sys.path.append(parent_dir+'/src')
# print(sys.path)

from src.Config import original_data_dir
from src.Names import amino_order
from src.Utils import FastaFile

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

ORDER = {}
for a in range(20):
    ORDER[amino_order[a]] = a


def make_grid(aaseq: str):
    """
    Turns a protein sequence into a numpy array (20x20) of dipeptide counts
    :param gseq: an amino acid sequence
    :return: numpy array (20x20)
    """

    grid = np.zeros((1, 20, 20))

    for n in range(1):
        for s in range(len(aaseq)-1-n):

            a1 = ORDER[aaseq[s]]
            a2 = ORDER[aaseq[s+1+n]]

            grid[n][a1][a2] += 1

    return grid


def jsim(g1, g2):
    """
    Calculate similarity between two dipeptide count grids

    :param g1: 20x20 np array dipeptide count grid
    :param g2: 20x20 np array dipeptide count grid
    :return: jaccard similarity (0.0 to 1.0)
    """
    overlap = np.sum(np.where(g1 < g2, g1, g2))
    union = np.sum(np.where(g1 < g2, g2, g1))
    return overlap/union


def score_by_all(seq1: str, seq2: str):

    grid_1 = make_grid(seq1)
    grid_2 = make_grid(seq2)
    return jsim(grid_1, grid_2)


def score_by_smallest(seq1: str, seq2: str):

    if len(seq1) < len(seq2):
        smallest = seq1
        longest = seq2
    else:
        smallest = seq2
        longest = seq1

    diff = len(longest) - len(smallest)

    grid_s = make_grid(smallest)

    max_score = 0.0
    for n in range(diff+1):
        window = longest[n:n+len(smallest)]
        assert(len(window) == len(smallest))

        grid_l = make_grid(window)

        score = jsim(grid_s, grid_l)
        if score > max_score:
            max_score = score

    return max_score


def score_by_window(seq1: str, seq2: str, window: int=100):

    if len(seq1) < len(seq2):
        smallest = seq1
        longest = seq2
    else:
        smallest = seq2
        longest = seq1

    if len(smallest) < window:
        return score_by_smallest(seq1, seq2)

    max_score = 0.0
    for x in range(len(smallest)-window):
        xseq = smallest[x:x+window]
        grid_x = make_grid(xseq)
        for y in range(len(longest)-window):
            yseq = longest[y:y+window]
            grid_y = make_grid(yseq)

            score = jsim(grid_x, grid_y)
            if score > max_score:
                max_score = score

    return max_score

########## Use case example WIP

from src.Config import original_data_dir
data_path = original_data_dir + '/clustering/clustering_data'

tpset = FastaFile(data_path+"/COMPLETE_SEQUENCES_UNTAGGED_20191126.fasta", "TrainingSet")
iseqs = tpset.fastadict

# Make distance pairs
ofile = open(data_path+"/SeqDistances.mult.CompleteSequences.20191213.txt", 'w')

def getstrings( tup ):
    x = tup[0]
    y = tup[1]

    ostr = "%10.8f %10.8f %s %s" % (score_by_all(iseqs[x], iseqs[y]), score_by_smallest(iseqs[x], iseqs[y]), x, y)

    return ostr

worklist = []
for x in iseqs.keys():
    for y in iseqs.keys():
        if x != y:
            worklist.append((x,y))
            print(getstrings((x,y)))


from multiprocessing import Pool
from tqdm import *

pool = Pool(processes=14)
with tqdm(total=len(worklist)) as pbar:
    for i, t in tqdm(enumerate(pool.imap_unordered(getstrings, worklist))):
        ofile.write(t+'\n')
        pbar.update()

ofile.close()
