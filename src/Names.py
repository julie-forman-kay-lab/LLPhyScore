"""This is the module to store residue names, maps and biophysical features."""

# amino acid names, 3-letter to 1-letter conversion.
amino3_to_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
               'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
               'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
               'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
               'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

# amino acid names, 1-letter to 3-letter conversion.
amino1_to_3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',
               'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY',
               'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
               'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
               'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

# default amino acid order.
amino_order = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# canonical amino acids.
canonical_amino_acids = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
                         'E': 0, 'Q': 0, 'G': 0, 'H': 0, 'I': 0,
                         'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0,
                         'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

# 1-letter amino acid to order number.
A_to_order = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
              'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
              'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
              'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

# order number to 1-letter amino acid.
order_to_A = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C',
              5: 'E', 6: 'Q', 7: 'G', 8: 'H', 9: 'I',
              10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P',
              15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V'}

# 3-letter amino acid to order number.
AAA_to_order = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}

# order number to 3-letter amino acid.
order_to_AAA = {0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS',
                5: 'GLU', 6: 'GLN', 7: 'GLY', 8: 'HIS', 9: 'ILE',
                10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE', 14: 'PRO',
                15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL'}

# the sr/lr tag meanings in different feature grids.
feature_tagABs = {"S2.SUMPI": ["srpipi", "lrpipi"],
                  "S3.WATER.V2": ["Water", "Carbon"],
                  "S4.SSPRED": ["ssH", "ssE", "ssL"],
                  "S5.DISO": ["disL", "disS"],
                  "S6.CHARGE.V2": ["srELEC", "lrELEC"],
                  "S7.ELECHB.V2": ["sr_hb", "lr_hb"],
                  "S8.CationPi.V2": ["srCATPI", "lrCATPI"],
                  "S9.LARKS.V2": ["larkSIM", "larkFAR"]}

# the output feature names.
feature_names = {"S2.SUMPI": 'pipi',
                  "S3.WATER.V2": 'water',
                  "S4.SSPRED": 'sec. structure',
                  "S5.DISO": 'disorder',
                  "S6.CHARGE.V2": 'charge',
                  "S7.ELECHB.V2": 'hydrogen bond',
                  "S8.CationPi.V2": 'cation pi',
                  "S9.LARKS.V2": 'K-Beta'}

# 16 sub-feature names.
sub_feature_names = {
    "pipi (srpipi)": ("S2.SUMPI", 0),
    "pipi (lrpipi)": ("S2.SUMPI", 1),
    "water (Water)": ("S3.WATER.V2", 0),
    "water (Carbon)": ("S3.WATER.V2", 1),
    "sec. structure (ssH)": ("S4.SSPRED", 0),
    "sec. structure (ssE)": ("S4.SSPRED", 1),
    "disorder (disL)": ("S5.DISO", 0),
    "disorder (disS)": ("S5.DISO", 1),
    "charge (srELEC)": ("S6.CHARGE.V2", 0),
    "charge (lrELEC)": ("S6.CHARGE.V2", 1),
    "hydrogen bond (sr_hb)": ("S7.ELECHB.V2", 0),
    "hydrogen bond (lr_hb)": ("S7.ELECHB.V2", 1),
    "cation pi (srCATPI)": ("S8.CationPi.V2", 0),
    "cation pi (lrCATPI)": ("S8.CationPi.V2", 1),
    "K-Beta (larkSIM)": ("S9.LARKS.V2", 0),
    "K-Beta (larkFAR)": ("S9.LARKS.V2", 1)
}

# signs +/- for 16 sub-features.
sub_feature_signs = {
    ("S2.SUMPI", 0): 1,
    ("S2.SUMPI", 1): 1,
    ("S3.WATER.V2", 0): 1,
    ("S3.WATER.V2", 1): -1,
    ("S4.SSPRED", 0): -1,
    ("S4.SSPRED", 1): 1,
    ("S5.DISO", 0): 1,
    ("S5.DISO", 1): 1,
    ("S6.CHARGE.V2", 0): 1,
    ("S6.CHARGE.V2", 1): -1,
    ("S7.ELECHB.V2", 0): 1,
    ("S7.ELECHB.V2", 1): 1,
    ("S8.CationPi.V2", 0): -1,
    ("S8.CationPi.V2", 1): -1,
    ("S9.LARKS.V2", 0): 1,
    ("S9.LARKS.V2", 1): -1
}

# selected 8, 12 and 16 features in the final predictor training.
selected_8features = [
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

selected_12features = [
    # "pipi (srpipi)",
    "pipi (lrpipi)",
    "water (Water)",
    "water (Carbon)",
    "sec. structure (ssH)",
    # "sec. structure (ssE)",
    "disorder (disL)",
    "disorder (disS)",
    "charge (srELEC)",
    "charge (lrELEC)",
    # "hydrogen bond (sr_hb)",
    "hydrogen bond (lr_hb)",
    # "cation pi (srCATPI)",
    "cation pi (lrCATPI)",
    "K-Beta (larkSIM)",
    "K-Beta (larkFAR)",
]

selected_16features = [
    "pipi (srpipi)",
    "pipi (lrpipi)",
    "water (Water)",
    "water (Carbon)",
    "sec. structure (ssH)",
    "sec. structure (ssE)",
    "disorder (disL)",
    "disorder (disS)",
    "charge (srELEC)",
    "charge (lrELEC)",
    "hydrogen bond (sr_hb)",
    "hydrogen bond (lr_hb)",
    "cation pi (srCATPI)",
    "cation pi (lrCATPI)",
    "K-Beta (larkSIM)",
    "K-Beta (larkFAR)",
]