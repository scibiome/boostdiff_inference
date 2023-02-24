
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


def colorize_by_condition(df_dis, df_con):
    """
    Parameters
    ----------
    df_dis:
    df_con
    Input dfs should already be filtered
    Should have columns ["target","regulator"]

    Returns
    -------
    The colorized df
    Edges that are stronger in disease: darkred
    Edges that are stronger in control: darkgreen
    Edges that are conflicting (if any): black
    """

    df_dis = df_dis[["target", "regulator"]]
    df_dis.loc[:, "condition"] = "disease"

    df_con = df_con[["target", "regulator"]]
    df_con.loc[:, "condition"] = "control"

    df_both = pd.concat([df_dis, df_con])
    df_both = df_both.drop_duplicates(['target', 'regulator'])

    # Find and mark conflicting edges (should be colored gray)
    df_conflict = pd.merge(df_dis, df_con, how='inner', on=['target', 'regulator'])
    df_conflict = df_conflict[['target','regulator']]

    if df_conflict.empty:
        df_final = df_both
    else:
        df_conflict.loc[:, "condition"] = "both"
        print("No. of conflicting edges:", df_conflict.shape[0])

        df_final = pd.concat([df_both, df_conflict])

    # Add color using a mapping
    color_map = {"disease": "darkred", "control": "darkgreen", "both": "black"}
    df_final["color"] = df_final["condition"].map(color_map)

    return df_final


def plot_grn(df_preprocessed, layout="graphviz_layout", show_conflicting=True,
             filename=None, highlight_genes=None):

    """
    Parameters
    ----------
    df_preprocessed: the output of colorized_by_condition
    layout: 'random_layout', 'graphviz_layout'
    show_conflicting: whether to show conflicting edges between
                    the two runs
    filename: filename of the differential network image
    highlight_genes: list of genes to be highlighted, will be colored blue

    Returns
    -------
    Plots the directed graph with networkx
    Image will be saved as a png file
    """

    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches

    G = nx.from_pandas_edgelist(df_preprocessed, 'regulator', 'target', ['color'], create_using=nx.DiGraph())

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]

    if highlight_genes:
        node_colors = ['lightblue' if node in highlight_genes else 'lightgray' for node in G.nodes()]
    fig, ax = plt.subplots(figsize=(20, 20))

    # Draw the differential GRN
    if layout == "random_layout":
        pos = nx.random_layout(G)

    elif layout == 'graphviz_layout':
        # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pydot.graphviz_layout.html
        pos = graphviz_layout(G, prog='sfdp')

    if highlight_genes:
        nx.draw_networkx_nodes(G, pos, node_size=1800, ax=ax, node_color=node_colors)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=1800, ax=ax, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)
    nx.draw_networkx_edges(G, pos, edge_color=colors, arrows=True, arrowsize=20, width=2, min_target_margin=25, ax=ax)

    # Add custom legend
    # https://matplotlib.org/1.3.1/examples/pylab_examples/legend_demo_custom_handler.html
    # Citation: https://stackoverflow.com/questions/22348229/matplotlib-legend-for-an-arrow
    def make_legend_arrow(legend, orig_handle,
                          xdescent, ydescent,
                          width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height)
        return p

    arrow1 = plt.arrow(0, 0, 0, 0, color="darkred", head_width=0)
    arrow2 = plt.arrow(0, 0, 0, 0, color="darkgreen", head_width=0)

    if show_conflicting:
        arrow3 = plt.arrow(0, 0, 0, 0, color="black", head_width=0)
        ax.legend([arrow1, arrow2, arrow3], ['Stronger in disease', 'Stronger in control', 'Conflicting'],
                  handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), }, loc="upper right",
                  fontsize=14)
    else:
        ax.legend([arrow1, arrow2], ['Stronger in disease', 'Stronger in control'],
                  handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), }, loc="upper right",
                  fontsize=14)

    if filename:
        plt.savefig(filename, bbox_inches="tight", format="png")
    plt.show()