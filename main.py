import os
from training import train_and_eval

os.environ["HYDRA_FULL_ERROR"] = '1'

train_and_eval()
