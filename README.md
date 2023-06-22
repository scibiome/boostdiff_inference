# BoostDiff 
BoostDiff (Boosted Differential Trees) - Tree-based Inference of Differential Networks from Gene Expression Data


## General info
BoostDiff is a tree-based method for inferring differential networks from large-scale transcriptomics data 
by simultaneously comparing gene expression  from two biological conditions (e.g. disease vs. control conditions). 
The network is inferred by building modified AdaBoost ensembles of differential trees as base learners. BoostDiff modifies regression trees to use differential variance improvement (DVI) as the novel splitting criterion. 

## Installation

To install the package from git:

```
conda create --name bdenv python=3.7
conda activate bdenv
conda install numpy pandas networkx pandas matplotlib cython

git clone https://github.com/gihannagalindez/boostdiff_inference.git  && cd boostdiff_inference
pip install .
```


## Data input

BoostDiff accepts two filenames corresponding to gene expression matrices from the disease and control conditions.
Each input is a tab-separated file. The rows correspond to genes, while the columns correspond to samples. Note that the two inputs should have the same set of features, where the first column should be named "Gene".

Disease expression data:
| Gene  |   Disease1   |   Disease2  | Disease3  | 
|-------------------|-----------|-----------|------|
| ACE2   | 0.345  | 0.147  |0.267 | 
| APP   | 0.567  | 0.794  | 0.590 | 

Control expression data:
| Gene  |   Control1   |   Control2  | Control3  | 
|-------------------|-----------|-----------|------|
| ACE2   | 0.084  | 0.147  |0.91 | 
| APP   | 0.567  | 0.794  | 0.590 | 

## Parameters:
1. **run**(file_disease, file_control, output_folder, n_estimators, \
          n_features, n_subsamples, keyword=keyword, n_processes=n_processes)

- `file_disease`: *string*, path to the disease expression data 
- `file_control`: *string*, path to the control expression data 
- `output_folder`: *string*, the folder where the output will be generated
- `n_estimators`: *int*, the no. of trees to be built in the modified AdaBoost ensemble
- `n_features`: *int*, the no. of regulators (predictor genes/features) to be used
- `n_subsamples`: *int*, the no. of bootstrapped samples to be used per condition (disease samples and control samples) when building the tree ensemble
- `min_samples_leaf`: *int (default = 2)*, the minimum no. of samples in a leaf
- `min_samples_split`: *int (default = 6)*, the minimum no. of samples required to split an internal node
- `max_depth`: *int (default = 2)*, the maximum depth of a differential tree; max_depth=0 means tree stumps will be built
- `min_samples`: *int (default = 15)*, the minimum no. of samples required per condition (disease samples and control samples) 
- `learning_rate`: *float (default = 1.0)*, the minimum no. of samples required per condition (disease samples and control samples) 
- `loss`: *string (default = "square")*, the loss used for AdaBoost {"square","exponential","linear"}
- `n_processes` *int (default = 1")*, for parallel computing 
- `keyword` *string*, keyword used for naming the output files
- `regulators`: the string 'all' if all genes can be considered potential predictors or a list of gene names to be used as predictors
- `normalize` *string or bool*:, if False, then no normalization will be performed; default is "unit_variance" normalization

## Example

Import the BoostDiff package:

```python
from boostdiff.main_boostdiff import BoostDiff
```

### Step 1: Run BoostDiff 

```python

file_disease = "/path/to/boostdiff_inference/data/expr_disease.txt"
file_control = "/path/to/boostdiff_inference/data/expr_control.txt"
output_folder = "/path/to/output/"

n_estimators = 50
n_features = 50
n_subsamples = 50
keyword = "test"
n_processes = 2

model = BoostDiff()
model.run(file_disease, file_control, output_folder, n_estimators, \
          n_features, n_subsamples, keyword=keyword, n_processes=n_processes)

```

BoostDiff will output two subfolders, each containing two txt files.
<br />
<br /> Note that BoostDiff runs the algorithm twice:
<br /> Run 1: The disease condition will be used as the target condition (with control condition as baseline). Results will be generated in the subfolder "disease".
<br /> Run 2: The control/healthy condition will be used as the target condition (with disease condition as baseline).  Results will be generated in the subfolder "control".
<br /> <br /> In each subfolder, the first txt file shows the data for the mean difference in prediction error between the disease and control samples after training the boosted differential trees. The second txt file contains the raw output network.

###  Step 2: Filtering

To obtain the final differential network, the raw network should be filtered for target genes in which BoostDiff found a more predictive model for the target condition. This additional step is crucial and part of the pipeline, as a trained model will not always be more predictive of a target condition. 

Sample processing for the run where disease condition was used as the target condition:

```python
import boostdiff.postprocessing as pp

# Specify the output files containing the mean difference in prediction error after running the BoostDiff algorithm
file_diff_dis = "/path/to/output/disease/differences_test.txt"
file_diff_con = "/path/to/output/control/differences_test.txt"

# Specify the output file containing the raw network output after running the BoostDiff algorithm
file_net_dis = "/path/to/output/disease/boostdiff_network_test.txt"
file_net_con = "/path/to/output/control/boostdiff_network_test.txt"

# Filter the raw output based on the no. of top targets in the differences files
# Then the top 50 edges for the run where the disease condition is the target condition
# Also the top 50 edges for the run where the control condition is the target condition
df_dis = pp.filter_network(file_net_dis, file_diff_dis, n_top_targets=10, n_top_edges=50)
df_con = pp.filter_network(file_net_con, file_diff_con, n_top_targets=10, n_top_edges=50)

# Example for real, large-scale datasets: filtering based on 3rd percentile with the p parameter 
# df_filtered = pp.filter_network(file_net, file_diff, p=3, n_top_edges=100)

# For plotting the differential network
# Colorize by condition
df_both = pp.colorize_by_condition(df_dis, df_con)
# Generate and save the plot
file_grn = "/path/to/output/diff_grn.png"
pp.plot_grn(df_both, layout="graphviz_layout", show_conflicting=True, filename=file_grn)


# Save the final differential network to file
file_output = "/path/to/output/filtered_network_disease.txt"
df_filtered.to_csv(file_output, sep="\t")
```
Here is a sample differential network:
![diff_grn](data/sample_output/diff_grn.png)

## General recommendations and filtering on real datasets

The ideal number of samples per condition as input to BoostDiff would be at least 30 samples each. For real transcriptomics datasets, we recommend to set a relatively low number of base learners: n_estimators=50, so that 50 adaptively boosted differential trees will be built. For filtering, we recommend to start with filtering target genes using p=3 parameter and n_top_edges=250 to 500 per condition (500 to 1000 edges in the final network).

df_filtered = pp.filter_network(file_net, file_diff, p=3, n_top_edges=250)

## Citation 

Galindez, G. G., List, M., Baumbach, J., Blumenthal, D. B., & Kacprowski, T. (2022). Inference of differential gene regulatory networks from gene expression data using boosted differential trees. bioRxiv. doi: https://doi.org/10.1101/2022.09.26.509450.

##  References

[1] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830.
<br /> [2] Huynh-Thu, V. A., Irrthum, A., Wehenkel, L., & Geurts, P. (2010). Inferring regulatory networks from expression data using tree-based methods. PloS one, 5(9), e12776. <br />
[3] Bhuva, D. D., Cursons, J., Smyth, G. K., & Davis, M. J. (2019). Differential co-expression-based detection of conditional relationships in transcriptional data: comparative analysis and application to breast cancer. Genome biology, 20(1), 1-21.

## Contact 
Gihanna Galindez: gihanna.galindez@plri.de
