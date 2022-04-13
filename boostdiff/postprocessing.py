
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import networkx as nx


def plot_histogram(file_diff, file_output=""):
    
    '''
    Post-processing visualization (optional): Examine the distribution of 
                               mean difference in prediction error
    Input: filename containing differences in prediction error
    Output: plots the histogram
    '''

    df_diff = pd.read_csv(file_diff, sep="\t")
    df_diff.columns = ["gene","error_diff"]

    plt.figure(figsize=(10,7))
    plt.hist(df_diff.error_diff, bins=100)
    plt.title("Distribution of differences in prediction error \n after building the Adaboost model", fontsize=20)
    plt.xlabel("Difference in prediction error", fontsize=16)
    plt.ylabel("No. of genes", fontsize=16)
    plt.show()
    
    if file_output:
        plt.savefig(file_output, format="png")
    
        
def filter_network(file_raw_net, file_diff, n_top_edges=100, **kwargs):

    """
    Performs the two-step required post-processing
        
    Step 1: Filters the network based on target genes where boosting
            found a more predictive model in the target condition
            with the most extreme mean difference in prediction error
    Step 2: Re-rank the edges after filtering and consider the first n_top_edges
            as the final differential network
            
    Keyword Args:
        p (float): the pth percentile of target genes 
        n_top_targets (int): the n_top_targets target genes 
    """
    
    # Load the differences file
    df_diff = pd.read_csv(file_diff, sep="\t", header=None)
    df_diff.columns = ["gene","error_diff"]    
    
    df_net = pd.read_csv(file_raw_net, sep="\t", header=None)
    df_net.columns = ["target","regulator","weight"]
    
    # Identify filtering method
    if "p" in kwargs:
        
        p = kwargs.get("p", 3)
        print("Identifying thresholds based on {}th percentile".format(p))
    
        # Determine the threshold based on percentile
        thresh = np.percentile(np.asarray(df_diff["error_diff"]), p)
        top_target_genes = list(df_diff[df_diff.error_diff < thresh].gene)
        
        print("No of target genes in disease condition satisfying threshold: ", len(top_target_genes))
    
    elif "n_top_targets" in kwargs:
        
        n_top_targets = kwargs.get("n_top_targets", 10)
        
        # Get top edges based on difference in prediction error
        print("Getting the top {} target genes...".format(n_top_targets))
        df_diff = df_diff.sort_values(by=['error_diff'], ascending=True)
        top_target_genes = list(df_diff.head(n_top_targets)["gene"])
    
    # Sort and get the top edges
    print("Extracting the top {} edges...".format(n_top_edges))
    df_filt = df_net[df_net.target.isin(top_target_genes)]
    df_filt = df_filt.sort_values(by=["weight"], ascending = False)
    df_filt = df_filt.head(n_top_edges)
    
    return df_filt


def visualize_network(df_filtered, n_largest_components = 5, output_folder="", keyword=""):
    
    '''
    Visualize the largest subnetworks of the final differential network
    Saves file to default png file to the output folder
    '''
    
    G = nx.from_pandas_edgelist(df_filtered, 'regulator', 'target', ['weight'], create_using=nx.DiGraph())
    pos = nx.spring_layout(G)    
    
    all_components = [c for c in sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)]
    
    count = 1
    for comp in all_components[:n_largest_components]:
    
        G_sub = G.subgraph(comp)
        d_sub = dict(G_sub.degree)
        
        pos = nx.shell_layout(G_sub)
        fig, ax = plt.subplots(figsize=(9,9))
    
        ax.set_axis_off()
        nx.draw_networkx_edges(G_sub, pos, ax=ax, arrowstyle="-|>", arrowsize=20, min_source_margin=35, min_target_margin=38)
        nx.draw_networkx_nodes(G_sub, pos, node_color="lightblue", 
                               node_size=[2100 for v in d_sub.values()], ax=ax)
        nx.draw_networkx_labels(G_sub, pos, font_size=21,font_color='black', ax=ax)
        
        file_output = os.path.join(output_folder, "subnetwork_{}.svg".format(count))
        print("file_output", file_output)
        plt.savefig(file_output, format="svg")
        # plt.show()
        
        count+=1