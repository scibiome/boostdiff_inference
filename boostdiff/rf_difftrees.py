

import numpy as np
import pandas as pd
import multiprocessing
from operator import itemgetter
import os

from .differential_trees.diff_tree import DiffTree


class RFDiff():
    
    
    def __init__(self, max_features=100, min_samples_leaf=2, min_samples_split=6,
                 max_depth=10, variable_importance = "disease_improvement"):
        
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.variable_importance = variable_importance
    
    
    def get_forest_single(self, X_disease, X_control, output_idx, input_idx, gene_names,
                          n_trees=100,
                          normalize=True):
    
        assert(X_disease.shape[1] == X_control.shape[1])
        
        # print("X_disease.head()", X_disease.head())
        n_genes = X_disease.shape[1]
        n_disease = X_disease.shape[0]
        n_control = X_control.shape[0]
        
        # Initialize 2D arrays to store variable importances
        vim = np.zeros((n_trees, n_genes))
        
        # Expression of target gene as predicted values
        y_disease = X_disease.iloc[:,output_idx].values
        y_control = X_control.iloc[:,output_idx].values
        
        if normalize:
            
            # Scale data together to unit variance
            y_orig_both = np.concatenate([y_disease, y_control])
            y_orig_both = y_orig_both / np.std(y_orig_both)
            
            y_disease = y_orig_both[:len(y_disease)]
            y_control = y_orig_both[len(y_disease):]
                        
        # Remove target gene from candidate regulators
        input_idx = list(input_idx)
        if output_idx in input_idx:
            input_idx.remove(output_idx)
        
        # Subset the regulators as input features
        X_disease_input = X_disease.iloc[:,input_idx].values
        X_control_input = X_control.iloc[:,input_idx].values
        
        # Build the forest
        for i in range(n_trees):
                
            # Step 1: Build the tree
            tree = DiffTree(self.min_samples_leaf, self.min_samples_split, 
                          n_disease, n_control,
                          self.max_depth, self.max_features)
            
            tree.build(X_disease_input, X_control_input, y_disease, y_control)  
    
            # Step 2: Extract the variable importances
            if self.variable_importance == "disease_improvement":
                vim[i,input_idx] = tree.get_variable_importance_disease_gain()
                
            elif self.variable_importance == "differential_improvement":
                vim[i,input_idx] = tree.get_variable_importance_differential_impvrovement()
        
        # print("vi_rf", vi_rf)
        # print(vim.shape)
        return vim
    
    
    def process_importance(self, arr, method = 'mean'):
        
        if method == 'sum':            
            return np.sum(arr, axis=0)
            
        elif method == 'mean':
            return np.mean(arr, axis=0)
                
            
    def get_link_list_rf(self, VIM_tuple, gene_names, regulators='all', output_folder=None, keyword=''):
        
        # Check input arguments      
        # if not isinstance(VIM_tuple[0], np.ndarray):
        #     raise ValueError('VIM must be a square array')
            
        # elif VIM_tuple[0].shape[0] != VIM_tuple[0].shape[1]:
        #     raise ValueError('VIM must be a square array')
        
        n_genes = VIM_tuple.shape[0]
        
        # Subset the candidate regulators
        if regulators == 'all':
            input_idx = list(range(n_genes))
        else:
            input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
        
        # Get the non-ranked list of regulatory links [(target, regulator, score)]
        vInter_rf = [(i,j,score) for (i,j),score in np.ndenumerate(VIM_tuple) if j in input_idx and i!=j]
        
        
        # Rank the list according to the weights of the edges        
        vInter_sort_rf = sorted(vInter_rf,key=itemgetter(2),reverse=True)
        
        nInter = len(vInter_sort_rf)
        
        # Write to output files
        filename_rf = os.path.join(output_folder,"rf_difftrees_{}.txt".format(keyword))
        
        with open(filename_rf, 'w') as f:
            
            for i in range(nInter):
                (target_idx, TF_idx, score) = vInter_sort_rf[i]
                TF_idx = int(TF_idx)
                target_idx = int(target_idx)
                f.write('{}\t{}\t{:.10f}\n'.format(gene_names[target_idx],gene_names[TF_idx],score))
                
            f.close()


    def run_rf(self, df_disease, df_control, gene_names, regulators, output_folder,  
                        n_processes, n_trees=100,
                        min_samples_dis=15, min_samples_con=10, 
                        feature_processing='mean',
                        keyword="output"):
                        
        
        """
        Input: Filename of gene expression matrix with rows: genes, columns: drug treatment
        Output: numpy.ndarray with rows: drug treatment, columns: genes (features)
        TODO: assert that they have the same number and arrangement of features
        """
        
        print("\n====parameters========")
        print("n_trees", n_trees)
        print("max_depth", self.max_depth)
        print("min_samples_leaf", self.min_samples_leaf)
        print("min_samples_split", self.min_samples_split)
        print("variable_importance", self.variable_importance)
        print("max_features", self.max_features)        
        print("min_samples_dis", min_samples_dis)
        print("min_samples_con", min_samples_con)
    
        # Put genes in columns
        df_disease = df_disease.T
        df_control = df_control.T

        n_disease = df_disease.shape[0]
        n_control = df_control.shape[0]
        
        # Subsample the rows of the dataframes with replacement
        X_disease_sub = df_disease.sample(n=n_disease, replace=True)
        X_control_sub = df_control.sample(n=n_control, replace=True) 
    
        gene_cols = X_disease_sub.columns
        
        X_disease_sub = X_disease_sub.astype(np.float64)
        X_control_sub = X_control_sub.astype(np.float64)
        
        n_genes = X_disease_sub.shape[1]
        
        # Subset the candidate regulators
        if isinstance(regulators, str):
            
            if regulators=="all":
                input_idx = list(range(n_genes))
            else:
                print("Please indicate valid regulators (list of gene names or 'all)")
        else:
            input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
        
        # The no. of samples used for for inference should be large
        if n_disease > int(min_samples_dis) and n_control > int(min_samples_con):
            
            PROCESSES = n_processes
            with multiprocessing.Pool(PROCESSES) as pool:
            
                input_data = list()
                for i in range(n_genes):
                    
                    # print("gene", i)
                    input_data.append([X_disease_sub, X_control_sub, i, input_idx, gene_cols, n_trees])
    
                all_output = [pool.apply_async(self.get_forest_single, input_data_i) for input_data_i in input_data]
                # Each output contains the variable importances for each gene
                outputs = [p.get() for p in all_output]
            
                pool.close()
                
            # Collate the variable importances
            VIM = np.zeros((n_genes,n_genes))
            
            for i,output in enumerate(outputs):
                VIM[i,:] = self.process_importance(output, method=feature_processing)

            self.get_link_list_rf(VIM, gene_names, regulators, output_folder, keyword)
        
        else:
            print("Min_samples not satisfied")
            
        