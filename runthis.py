# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 01:00:36 2021

@author: gihan
"""
import pandas as pd

from boostdiff.main_boostdiff import BoostDiff


if __name__ == '__main__':
    
    file_disease = r"C:\Users\gihan\boostdiff_inference\data\expr_disease.txt"
    file_control = r"C:\Users\gihan\boostdiff_inference\data\expr_control.txt"
    output_folder =  r"C:\Users\gihan\boostdiff_inference\data\sample_output"
    
    n_processes = 1
    n_estimators = 10
    n_features = 20
    n_subsamples = 50
    keyword = "test"
    normalize = "unit_variance"
    index = "Gene"
    learning_rate= 0.5

    model = BoostDiff()
    model.run(file_disease, file_control, output_folder, n_estimators, 
              n_features, n_subsamples, n_processes=n_processes, 
              keyword=keyword,  normalize=normalize, index=index, learning_rate=learning_rate)