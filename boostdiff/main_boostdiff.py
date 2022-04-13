

from .adaboost import AdaBoostDiffRegressor
from .differential_trees.diff_tree import DiffTree
import numpy as np
import time
import os
import pandas as pd
import multiprocessing
from operator import itemgetter
import matplotlib.pyplot as plt


class BoostDiff():
    
    
    def __init__(self):
        pass


    def adaboost_single_gene(self, X_disease, X_control, output_idx, input_idx, gene_names,
                          max_features, min_samples_leaf, min_samples_split, max_depth,
                          n_subsample, learning_rate, loss, n_estimators,
                          output_folder, normalize, variable_importance):


        '''
        Builds a model for  a single target gene
        using the modified AdaBoost algorithm with the
        differential tree as base learner 
        
        Parameters
        ----------
        df_disease: pandas data frame containing the gene expression data of the
                    disease samples.
        df_control: pandas data frame containing the gene expression data of the
                    control samples.
        output_idx: int index of the target gene to be predicted
        input_idx: list of int indices to be used as predictors
        gene_names: list of strings containing the gene names for all genes
                    in the dataset
        max_features: no. of features to be considered for each split
        min_samples_leaf: minimum no. of samples in a leaf node
        min_samples_split: minimum no. of samples in a node for splitting
        max_depth: maximum depth of a differential tree
        n_subsamples: no. of samples in disease and control datasets used to
                      fit a differential tree
        learning_rate: learning rate for AdaBoost training
        loss: loss function used for AdaBoost ('square','exponential')
        n_estimators: no. of trees in the AdaBoost ensemble
        n_threads: no. of threads
        keyword: string keyword used for naming the output files
        output_folder: name of output folder
        normalize: normalization method ('unit_variance','minmax')
        variable_importance: variable importance measure to be used
                ('disease_importance','differential_improvement')
                
        Returns
        -------
        (vi, difference): tuple containing the vi and difference
        vi: numpy array containing the variable importances
        difference: mean difference in training error in disease samples
                    minus mean difference in training error in control samples
        '''
  
        y_orig_dis = X_disease.iloc[output_idx,:].values
        y_orig_con = X_control.iloc[output_idx,:].values
        
        expr_data_disease = X_disease.iloc[input_idx,:]
        expr_data_control = X_control.iloc[input_idx,:]
        
        
        if normalize == "unit_variance":
            
            # Scale together to unit variance
            y_orig_both = np.concatenate([y_orig_dis, y_orig_con])
            y_orig_both = y_orig_both / np.std(y_orig_both)
            
            y_orig_dis = y_orig_both[:len(y_orig_dis)]
            y_orig_con = y_orig_both[len(y_orig_dis):]
                
            
        # Initialize the AdaBoosted differential trees
        model = AdaBoostDiffRegressor(DiffTree, min_samples_leaf, min_samples_split, max_depth, max_features,
                                  n_estimators, learning_rate, 
                                  loss, variable_importance) 

        # Build the ensemble model (one AdaBoost model per gene)
        model.fit(expr_data_disease, expr_data_control, y_orig_dis, y_orig_con, n_subsample=n_subsample)

        try:    
            predictions_dis = model.predict(expr_data_disease.T.values)
            predictions_dis = predictions_dis.flatten()
            
            predictions_con = model.predict(expr_data_control.T.values)
            predictions_con = predictions_con.flatten()
            
            # Prediction error
            error_disease = np.absolute(predictions_dis - y_orig_dis)
            error_control = np.absolute(predictions_con - y_orig_con)
    
            # Mean error across all samples
            error_dis_dis = np.mean(error_disease)
            error_dis_con = np.mean(error_control)
            
            # Difference in mean prediction error
            difference = error_dis_dis - error_dis_con
            
            output_file = os.path.join(output_folder, "{}_training_progress.png".format(gene_names[output_idx]))
            fig = plt.figure(figsize=(10,7))
            plt.plot(range(1, model.estimator_count), model.errors_disease[1:], label = 'disease error')
            plt.plot(range(1, model.estimator_count), model.errors_control[1:], label='control error')
            plt.title("AdaBoost training progress \n Gene {}".format(gene_names[output_idx]), fontsize=18)
            plt.xlabel("No. of boosting iterations", fontsize=20)
            plt.ylabel("Training error", fontsize=20)
            plt.legend()
            plt.savefig(output_file)
            plt.close(fig)
            
            # Compute feature importance
            VIM = model.calculate_adaboost_importance()
            
            vi = np.zeros((len(gene_names),))
            vi[input_idx] = VIM
            
        except Exception as e:
            
            # print("NAN GENE", gene_names[output_idx])
            vi = np.zeros((len(gene_names),))
            difference = np.nan
        
        return (vi, difference) 
        
                              
    def modified_adaboost(self, df_disease, df_control, gene_names, regulators, 
                          output_folder, 
                          n_estimators, num_features,
                          n_subsamples,
                          min_samples_leaf, min_samples_split, max_depth,
                          min_samples, learning_rate, loss, n_threads,
                          keyword='', normalize="unit_variance", variable_importance="disease_importance"):
                        
        '''
        Runs BoostDiff to reconstruct the differential network using
        gene expression data from the disease (target condition) and
        control (baseline conditon) samples
        
        Parameters
        ----------
        df_disease: pandas data frame containing the gene expression data of the
                    disease samples.
        df_control: pandas data frame containing the gene expression data of the
                    control samples.
        gene_names: list of strings containing the gene names for all genes
                    in the dataset
        regulators: 'all' if all genes can be considered potential predictors
                    or a list of gene names to be used as predictors
        output_folder: name of output folder
        n_estimators: no. of trees in the AdaBoost ensemble
        num_features: no. of features to be considered for each split
        n_subsamples: no. of samples in disease and control datasets used to
                      fit a differential tree
        min_samples_leaf: minimum no. of samples in a leaf node
        min_samples_split: minimum no. of samples in a node for splitting
        max_depth: maximum depth of a differential tree
        min_samples: minimum no. of samples in the disease and control datasets
                     to consider before performing AdaBoost
        learning_rate: learning rate for AdaBoost training
        loss: loss function used for AdaBoost ('square','exponential')
        n_threads: no. of threads
        keyword: string keyword used for naming the output files
        normalize: normalization method ('unit_variance','minmax')
        variable_importance: variable importance measure to be used
                ('disease_importance','differential_improvement')
        '''

        n_genes = len(gene_names)
        
        print("Total no. of genes: ", n_genes)
        
        if regulators == "all":
            print("Using all other genes as regulators.")
        else:
            print("No. of regulators: ", len(regulators))
        
        # Collate the variable importances here
        VIM_dis = np.zeros((n_genes, n_genes))
        dict_differences = {}
          
        # The no. of samples used for for inference should be large
        if df_disease.shape[1] > min_samples and df_control.shape[1] > min_samples:
            
            if n_threads > 1:
                        
                PROCESSES = n_threads
                                
                with multiprocessing.Pool(PROCESSES) as pool:
                              
                    input_data = list()
                    
                    for target_gene_idx in range(n_genes):
                        
                        # print("Target gene:", target_gene_idx, gene_names[target_gene_idx])
                            
                        # Regulators
                        if regulators == "all":
                            input_idx = np.arange(len(gene_names))
                        else:
                            input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

                        # Remove target gene from candidate regulators
                        input_idx = list(input_idx)
                        if target_gene_idx in input_idx:
                            input_idx.remove(target_gene_idx)
                            
                        input_data.append([df_disease, df_control, target_gene_idx, input_idx, gene_names, 
                                           num_features, min_samples_leaf, min_samples_split, 
                                           max_depth, n_subsamples, learning_rate, loss, n_estimators, output_folder, normalize,
                                           variable_importance])
        
                    all_output = [pool.apply_async(self.adaboost_single_gene, input_data_i) for input_data_i in input_data]
    
                    # Each of output arrays contains the variable importances for each gene/model
                    outputs = [p.get() for p in all_output]
                    pool.close()
            
                    for m, output in enumerate(outputs):
        
                        try:
                            
                            VIM_dis[m,:] = output[0]
                            difference = output[1]
                            dict_differences[gene_names[m]] = difference

                        except:
                            
                            VIM_dis[m,:] = np.zeros((n_genes,))            
                            
                            
            else:
                
                for target_gene_idx in range(n_genes):
                    
                    # Regulators
                    if regulators == "all":
                        input_idx = np.arange(len(gene_names))
                    else:
                        input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

                    # Remove target gene from candidate regulators
                    input_idx = list(input_idx)
                    if target_gene_idx in input_idx:
                        input_idx.remove(target_gene_idx)
                        
                    output = self.adaboost_single_gene(df_disease, df_control, target_gene_idx, input_idx, gene_names,
                          num_features, min_samples_leaf, min_samples_split, max_depth,
                          n_subsamples, learning_rate, loss, n_estimators, output_folder, normalize, variable_importance)

                    try:
                        VIM_dis[target_gene_idx, :] = output[0]
                        difference = output[1]
                        dict_differences[gene_names[target_gene_idx]] = difference
                                                    
                    except:
                        VIM_dis[target_gene_idx,:] = np.zeros((n_genes,))
                        


        return VIM_dis, dict_differences
            
    
    def write_differences(self, dict_differences, output_folder, keyword):
        
        '''
        Write the calculated differences in prediction error for each subset
        '''
        
        df = pd.DataFrame(dict_differences.items())
        file_out = os.path.join(output_folder, "differences_{}.txt".format(keyword))
        df.to_csv(file_out, sep="\t", index=False, mode='w', header=False)
     
        
    def write_link_list(self, VIM_tuple, gene_names, regulators='all', output_folder=None, keyword=''):
        
        '''
        Write the raw output (ordered link list) to file
        '''
        
        if not isinstance(VIM_tuple[0], np.ndarray):
            raise ValueError('VIM must be a square array')
            
        elif VIM_tuple.shape[0] != VIM_tuple.shape[1]:
            raise ValueError('VIM must be a square array')
            
        n_genes = VIM_tuple[0].shape[0]
        
        # Subset the candidate regulators
        if regulators == 'all':
            input_idx = list(range(n_genes))
        else:
            input_idx = [m for m, gene in enumerate(gene_names) if gene in regulators]

        # Get the non-ranked list of regulatory links [(target, regulator, score)]
        vInter = [(i,j,score) for (i,j),score in np.ndenumerate(VIM_tuple) if j in input_idx]

        # Rank the list according to the weights of the edges        
        vInter_sort = sorted(vInter,key=itemgetter(2),reverse=True)
        nInter = len(vInter_sort)
        
        # Write to output file
        filename_out = os.path.join(output_folder, "boostdiff_network_{}.txt".format(keyword))
        
        with open(filename_out, 'w') as f:
    
            for r in range(nInter):
                
                (target_idx, reg_idx, score) = vInter_sort[r]
                if score > 0.0:
                    f.write('{}\t{}\t{:.10f}\n'.format(gene_names[int(target_idx)], gene_names[int(reg_idx)],score))
                
            f.close()
    
    
    def run(self, file_disease, file_control, output_folder,
            n_estimators, n_features, n_subsamples, 
            min_samples_leaf=2, min_samples_split=6, max_depth=2,
            min_samples=15,  learning_rate = 1.0, loss = "square", n_processes=1,
            keyword = "", regulators='all', normalize="unit_variance",
            index = "Gene", variable_importance="disease_importance"):
        
        '''
        Input: tab-separated disease and control files
        Output files: ordered link list
                      differences file
        '''
    
        print('Running the inference using %d processes' % n_processes)

        output_folder_disease = os.path.join(output_folder, "disease")
        output_folder_control = os.path.join(output_folder, "control")
        folder_progress_disease = os.path.join(output_folder_disease, "{}_training_progress".format(keyword))
        folder_progress_control = os.path.join(output_folder_control, "{}_training_progress".format(keyword))
        
        if not os.path.exists(folder_progress_disease):
            os.makedirs(folder_progress_disease)  
        if not os.path.exists(folder_progress_control):
            os.makedirs(folder_progress_control)  
        
        print("==================")
        print("file_disease", file_disease)
        print("file_control", file_control)
        df_disease = pd.read_csv(file_disease, sep="\t")
        df_control = pd.read_csv(file_control, sep="\t")
        
        if index != None:
            df_disease = df_disease.set_index(index) 
            df_control = df_control.set_index(index)  
        
        # FOR TROUBLESHOOTING
        # df_disease = df_disease.iloc[:50, :20]
        # df_control = df_control.iloc[:50, :20]
        # print(df_disease.shape, df_control.shape)
        
        # Get the gene names (should be the same for disease and control)
        assert((df_disease.T.columns == df_control.T.columns).all())
        gene_names = df_disease.T.columns
                
        start=time.time() 
        
        # DISEASE as target condition
        print("Running disease as target condion")
        vim_dis, dict_differences_dis = self.modified_adaboost(df_disease, df_control, gene_names, regulators, 
                      folder_progress_disease, 
                      n_estimators, n_features,
                      n_subsamples,
                      min_samples_leaf, min_samples_split, max_depth, min_samples, 
                      learning_rate, loss, 
                      n_processes, keyword, normalize, variable_importance)
    
        # Rank edges by variable importance
        self.write_link_list(vim_dis, gene_names, regulators, output_folder_disease, keyword) 
        self.write_differences(dict_differences_dis, output_folder_disease, keyword)
        
        # CONTROL as target condition
        print("Running control as target condion")
        vim_con, dict_differences_con = self.modified_adaboost(df_control, df_disease, gene_names, regulators, 
                      folder_progress_control, 
                      n_estimators, n_features,
                      n_subsamples,
                      min_samples_leaf, min_samples_split, max_depth, min_samples,
                      learning_rate, loss, 
                      n_processes, keyword, normalize, variable_importance)
    
        # Rank edges by variable importance
        self.write_link_list(vim_con, gene_names, regulators, output_folder_control, keyword) 
        self.write_differences(dict_differences_con, output_folder_control, keyword)
        
        end = time.time()
        print("Time elapsed: {} min".format(((end-start)/60))) 