
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import networkx as nx
import seaborn as sns
import random
from networkx.drawing.nx_agraph import write_dot, graphviz_layout


def plot_histogram(output_folder, keyword):
    
    '''
    Post-processing visualization (optional): Examine the distribution of 
                               mean difference in prediction error
    Input: output_folder
    Output: plots the histograms
    '''

    file_diff_dis = os.path.join(output_folder, "disease", "differences_{}.txt".format(keyword))
    file_diff_con = os.path.join(output_folder, "control", "differences_{}.txt".format(keyword))
        
    df_diff_dis = pd.read_csv(file_diff_dis, sep="\t", header=None)
    df_diff_dis.columns = ["gene","diff"]
    df_diff_con = pd.read_csv(file_diff_con, sep="\t", header=None)
    df_diff_con.columns = ["gene","diff"]
    
    sns.set_style("dark")
    fig, ax = plt.subplots(1,2, figsize=(15,6), sharey=False)
    ax[0].hist(df_diff_dis["diff"], bins=200)
    ax[1].hist(df_diff_con["diff"], bins=200)
    ax[0].set_xlabel("Mean difference in prediction error", fontsize=16)
    ax[1].set_xlabel("Mean difference in prediction error", fontsize=16)
    ax[0].set_ylabel("No. of genes", fontsize=16)
    ax[1].set_ylabel("No. of genes", fontsize=16)
    ax[0].set_title("Disease as target condition", fontsize=18)
    ax[1].set_title("Control as target condition", fontsize=18)
    ax[0].grid(axis='y', color='0.95')
    ax[1].grid(axis='y', color='0.95')
    
    plt.suptitle("Distribution of differences in mean prediction error", fontsize=22, y=1.05)
   
    plt.show()
        
        
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


def colorize_by_condition(df_dis, df_con, color1="darkred",color2="darkblue", color3="black"):
    """
    Parameters
    ----------
    df_dis:
    df_con
    Input dfs should already be filtered
    Should have columns ["target","regulator"]
    Returns
    -------
    The colorized df defaults:
    Edges that are stronger in disease: darkred
    Edges that are stronger in control: darkblue
    Edges that are conflicting (if any): black
    """

    df_dis = df_dis[["target", "regulator"]]
    if not df_dis.empty:
        df_dis.loc[:, "condition"] = "condition1"

    df_con = df_con[["target", "regulator"]]
    if not df_con.empty:
        df_con.loc[:, "condition"] = "condition2"

    if df_dis.empty:
        df_both = df_con
    elif df_con.empty:
        df_both = df_dis
    else:
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
    color_map = {"condition1": color1, "condition2": color2, "both": color3}
    df_final["color"] = df_final["condition"].map(color_map)

    return df_final


def plot_grn(df_dis, df_con, layout="graphviz_layout", show_conflicting=True,
             filename=None, highlight_genes=None, cond1_label = "Stronger in disease", cond2_label = "Stronger in control",
            color1 = "darkred", color2="darkblue", color3="black", node_size=1400, fontsize=18,
            gv_layout="sfdp", figsize=(25, 25)):

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

    df_preprocessed = colorize_by_condition(df_dis, df_con, color1=color1,color2=color2, color3=color3)
    G = nx.from_pandas_edgelist(df_preprocessed, 'regulator', 'target', ['color'], create_using=nx.DiGraph())

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    if highlight_genes:
        node_colors = ['lightblue' if node in highlight_genes else 'lightgray' for node in G.nodes()]
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the differential GRN
    if layout == "random_layout":
        pos = nx.random_layout(G)

    elif layout == 'graphviz_layout':
        # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pydot.graphviz_layout.html
        pos = graphviz_layout(G, prog=gv_layout)

    if highlight_genes:
        nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax, node_color=node_colors)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=fontsize)
    nx.draw_networkx_edges(G, pos, edge_color=colors, arrows=True, arrowsize=20, width=2, min_target_margin=25, ax=ax)

    # Add custom legend
    # https://matplotlib.org/1.3.1/examples/pylab_examples/legend_demo_custom_handler.html
    # Citation: https://stackoverflow.com/questions/22348229/matplotlib-legend-for-an-arrow
    def make_legend_arrow(legend, orig_handle,
                          xdescent, ydescent,
                          width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height)
        return p

    # df_color = df_preprocessed[["condition","color"]].drop_duplicates()
    # color_mapping = dict(zip(df_color.condition, df_color.color))
    
    # if len(color_mapping) == 2:
    arrow1 = plt.arrow(0, 0, 0, 0, color=color1, head_width=0)
    arrow2 = plt.arrow(0, 0, 0, 0, color=color2, head_width=0)
        
    if show_conflicting:
        arrow3 = plt.arrow(0, 0, 0, 0, color=color3, head_width=0)
        ax.legend([arrow1, arrow2, arrow3], [cond1_label, cond2_label, 'Conflicting'],
                  handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), }, loc="upper right",
                  fontsize=14)
    else:
        ax.legend([arrow1, arrow2], [cond1_label, cond2_label],
                  handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), }, loc="upper right",
                  fontsize=14)

    ax.set_facecolor('white')
    if filename:
        plt.savefig(filename, bbox_inches="tight", format="svg", transparent=True)

    # Hide grid lines
    plt.grid(False)
    
    plt.show()



def get_targets_list(df_diff, percentile=3):

    pth = np.nanpercentile(np.asarray(df_diff["diff"]), percentile) 
    df = df_diff[df_diff["diff"] < pth]
    
    return list(df.gene)

   
def get_unique_nodes(df):
    
    return list(set(list(df.regulator) + list(df.target)))


def analyze_correlation_distributions(file_disease, file_control, df_sub, n_top_edges=50, index="Gene"):
    
    # Get raw gene expression matrices
    df_disease = pd.read_csv(file_disease, sep="\t")
    df_control = pd.read_csv(file_control, sep="\t")
    
    if index:
        df_disease = df_disease.set_index(index)
        df_control = df_control.set_index(index)
    
    list_genes = get_unique_nodes(df_sub.head(n_top_edges))
    
    # Get the correlation matrices
    df_dis_sub = df_disease[df_disease.index.isin(list_genes)]
    corr_dis = df_dis_sub.T.corr()
    df_con_sub = df_control[df_control.index.isin(list_genes)]
    corr_con = df_con_sub.T.corr()
    
    target_edges = []
    for index, row in df_sub.head(n_top_edges).iterrows():
        target_edges.append((row["regulator"], row["target"]))
        
    nden_dis = [score for (i,j),score in np.ndenumerate(corr_dis) if (corr_dis.columns[i],corr_dis.columns[j]) in target_edges]
    nden_con = [score for (i,j),score in np.ndenumerate(corr_con) if (corr_con.columns[i],corr_con.columns[j]) in target_edges]    
    df_corrs_dis = pd.DataFrame({"corrs": nden_dis, "label":"Disease"})
    df_corrs_con = pd.DataFrame({"corrs": nden_con, "label":"Control"})
    df_corrs_both = pd.concat([df_corrs_dis, df_corrs_con])
    df_corrs_both = df_corrs_both.reset_index()
    
    return df_corrs_both


def get_correlations(file_dis_expr, file_con_expr, output_folder,
                     percentile=3, n_edges=300, keyword="", index=None):
    
    file_diff_dis = os.path.join(output_folder, "disease", "differences_{}.txt".format(keyword))
    file_diff_con = os.path.join(output_folder, "control", "differences_{}.txt".format(keyword))
    
    file_net_dis = os.path.join(output_folder, "disease", "boostdiff_network_{}.txt".format(keyword))
    file_net_con = os.path.join(output_folder, "control", "boostdiff_network_{}.txt".format(keyword))
    
    # Get top disease edges
    df_expr = pd.read_csv(file_net_dis, sep="\t", header=None)
    df_expr.columns = ["target","regulator","weight"]
    df_diff = pd.read_csv(file_diff_dis, sep="\t", header=None)
    df_diff.columns = ["gene","diff"]
    list_genes_disease = get_targets_list(df_diff, percentile)
    df_sub = df_expr[df_expr.target.isin(list_genes_disease)]

    # GEt top control edges
    df_expr2 = pd.read_csv(file_net_con, sep="\t", header=None)
    df_expr2.columns = ["target","regulator","weight"]
    df_diff2 = pd.read_csv(file_diff_con, sep="\t", header=None)
    df_diff2.columns = ["gene","diff"]
    list_genes_control = get_targets_list(df_diff2, percentile)
    df_sub2 = df_expr2[df_expr2.target.isin(list_genes_control)]

    # Select random edges
    df_both_all = pd.concat([df_sub, df_sub2])
    random_sampled = random.sample(range((df_sub.shape[0] + df_sub2.shape[0])), n_edges)
    df_random = df_both_all.iloc[random_sampled]

    # Get dfs 
    df_dis = analyze_correlation_distributions(file_dis_expr, file_con_expr, df_sub, n_top_edges=n_edges, index=index)
    df_con = analyze_correlation_distributions(file_dis_expr, file_con_expr, df_sub2, n_top_edges=n_edges, index=index)   
    df_rand = analyze_correlation_distributions(file_dis_expr, file_con_expr, df_random, n_top_edges=n_edges, index=index)

    return df_dis, df_con, df_rand
    
    
def plot_corr_dists(df_dis, df_con, df_rand, title="Analysis 1", filename=None, bw_adjust=0.07):
    
    """
    
    """
    
    fig, ax = plt.subplots(1,3, figsize=(20,7), sharey=True)
    
    # sns.set_theme(style="whitegrid")
    sns.set_theme()
    plt.style.use("seaborn-v0_8-dark")
    # sns.set(font_scale = 2.5)
    a = sns.violinplot(x="label", y="corrs", hue="label", data=df_dis, bw_adjust=bw_adjust, ax=ax[0])
    b = sns.violinplot(x="label", y="corrs", hue="label", data=df_con, bw_adjust=bw_adjust, ax=ax[1])
    c = sns.violinplot(x="label", y="corrs", hue="label", data=df_rand, bw_adjust=bw_adjust, ax=ax[2])
    
    xlabels = [i.get_text() for i in a.get_xticklabels()]
    ax[0].set_ylabel("Pearson correlation", fontsize=25)
    ax[1].set_ylabel("", fontsize=18)
    ax[2].set_ylabel("", fontsize=18)
    ax[0].set_xlabel("", fontsize=18)
    ax[1].set_xlabel("")
    ax[2].set_xlabel("")
    ax[0].set_title("Disease as target condition \n Top differential edges \n BoostDiff", fontsize=22)
    ax[1].set_title("Control as target condition \n Top differential edges \n BoostDiff", fontsize=22)
    ax[2].set_title("Randomly selected edges \n", fontsize=26)
    a.set_xticks(a.get_xticks(), labels=xlabels, size=25)
    b.set_xticks(a.get_xticks(), labels=xlabels, size=25)
    c.set_xticks(a.get_xticks(), labels=xlabels, size=25)
    
    sns.set_style(style='white') 
    fig.suptitle(title, fontsize=32, y=1.15) # or plt.suptitle('Main title')

    if filename:
        plt.savefig(filename, format="svg", bbox_inches='tight')
    plt.show()