"""This is the config file for local data directories and paths (too large to upload to github.)"""

from pathlib import Path

home_dir = str(Path.home())
original_data_dir = home_dir + '/VSCode/LLPhyScore'

# data directories
raw_data_dir = original_data_dir + '/data/raw'
processed_data_dir = original_data_dir + '/data/processed'
cleaned_data_dir = original_data_dir + '/data/cleaned'

# model directory
model_dir = original_data_dir + "/model"

# cache directory to store sequence grids
grid_cache_dir = original_data_dir + '/GridCache'

# score database directory to store observed sequence statistics
score_db_dir = original_data_dir + '/ScoreDBs'
