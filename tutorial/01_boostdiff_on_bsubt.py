

from boostdiff.main_boostdiff import BoostDiff
import pandas as pd
import numpy as np
import os

file_etha = "../data/b_subtilis_salt/exprs_salt.txt"
file_ctrl = "../data/b_subtilis_salt/exprs_smm.txt"


n_processes = 4
n_estimators = 50
n_features = 2500
n_subsamples = 20
keyword = "saltvssmm"
normalize = True
loss = "exponential"

output_folder = "../outputs_{}".format(keyword)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = BoostDiff()
model.run(file_etha, file_ctrl, output_folder, n_estimators, 
                  n_features, n_subsamples, n_processes=n_processes, 
                  keyword=keyword,normalize=normalize,
                  regulators="all", loss=loss)
                  