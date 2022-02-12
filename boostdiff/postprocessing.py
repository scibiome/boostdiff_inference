
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import networkx as nx
import matplotlib as mpl



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
    
    
        
def filter_network(output_folder, p=3, n_top_edges=100):
    
    '''
    Performs the two-step required post-processing
        
    Step 1: Filters the network based on target genes where boosting
            found a more predictive model in the target condition
    Step 2: Re-rank the edges after filtering and consider the first n_top_edges
            as the final differential network
            
    Inputs:
        output_folder

    Returns the filtered pandas data frame
    '''
    
    # =========DISEASE condition=========
    # Load the differences file
    file_diff = glob.glob(os.path.join(output_folder, "disease", "differences*.txt"))
    df_diff_disease = pd.read_csv(file_diff, sep="\t")
    df_diff_disease.columns = ["gene","error_diff"]    
    
    # Load the raw unfiltered network
    file_raw_output = glob.glob(os.path.join(output_folder, "disease", "boostdiff_network*.txt"))
    df_net_disease = pd.read_csv(file_raw_output, sep="\t", header=None)
    df_net_disease.columns = ["target","regulator","weight"]
    
    # =========CONTROL condition=========
    # Load the differences file
    file_diff_control = glob.glob(os.path.join(output_folder, "control", "differences*.txt"))
    df_diff_control = pd.read_csv(file_diff_control, sep="\t")
    df_diff_control.columns = ["gene","error_diff"]    
    
    # Load the raw unfiltered network
    file_raw_output_control = glob.glob(os.path.join(output_folder, "control", "boostdiff_network*.txt"))
    df_net_control = pd.read_csv(file_raw_output_control, sep="\t", header=None)
    df_net_control.columns = ["target","regulator","weight"]
    
    # Determine the threshold based on percentile
    threshold1 = np.percentile(np.asarray(df_diff_disease["error_diff"]), p)
    threshold2 = np.percentile(np.asarray(df_diff_control["error_diff"]), p)
    print("Identifying thresholds based on {}th percentile".format(p))
    
    df1 = df_diff_disease[df_diff_disease.error_diff < threshold1]
    list_genes_disease = list(df1.gene)
    df2 = df_diff_control[df_diff_control.error_diff < threshold2]
    list_genes_control = list(df2.gene)
    print("No of target genes in disease condition satisfying threshold: ", len(list_genes_disease))
    print("No of target genes in control condition satisfying threshold: ", len(list_genes_control))

    df_filtered_dis = df_net_disease[df_net_disease.target.isin(list_genes_disease)]
    df_filtered_dis = df_filtered_dis.sort_values(by=["weight"], ascending = False)
    df_filtered_con = df_net_control[df_net_control.target.isin(list_genes_control)]
    df_filtered_con = df_filtered_con.sort_values(by=["weight"], ascending = False)
    print("Re-ranking edges after filtering...")
    print("No of edges in the filtered network (disease as target condition): ", df_filtered_dis.shape[0])
    print("No of edges in the filtered network (control as target condition): ", df_filtered_con.shape[0])

    df_filtered_dis = df_filtered_dis.head(n_top_edges)
    df_filtered_con = df_filtered_con.head(n_top_edges)
    
    df_filtered_dis["target_condition"] = "disease"
    df_filtered_con["target_condition"] = "control"
        
    return pd.concat([df_filtered_dis, df_filtered_con])
        

def cmap_map(function, cmap):
    
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)


def visualize_network(df_filtered, n_largest_components = 5, output_folder=""):
    
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
        
        # pos=nx.random_layout(G_sub)
        pos = nx.shell_layout(G_sub)
        # pos = hierarchy_pos(G_sub, "CCL2")
        fig, ax = plt.subplots(figsize=(9,9))
    
        ax.set_axis_off()
        nx.draw_networkx_edges(G_sub, pos, ax=ax, arrowstyle="-|>", arrowsize=20, min_source_margin=35, min_target_margin=38)
        nx.draw_networkx_nodes(G_sub, pos, node_color="lightblue", 
                               node_size=[2100 for v in d_sub.values()], ax=ax)
        nx.draw_networkx_labels(G_sub, pos, font_size=21,font_color='black', ax=ax)
        
        # file_output = r"/home/giga/COVID/COVID_99thpercentile_top200/COVID_2ndpercentile_top200_subnetwork_{}.svg".format(count)
        file_output = os.path.join(output_folder, "COVID_subnetwork_{}.svg".format(count))
        print("file_output", file_output)
        # plt.savefig(file_output, format="svg")
        # plt.show()
        
        count+=1