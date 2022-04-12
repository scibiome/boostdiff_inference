# BoostDiff 
BoostDiff (Boosted Differential Trees) - Tree-based Inference of Differential Networks from Gene Expression Data


## General info
The python package BoostDiff (Boosted Differential Trees) is a tree-based method for inferring differential networks from large-scale transcriptomics data 
by simultaneously comparing data from two biological contexts (e.g. disease vs. control conditions). 
The network is inferred by building AdaBoost ensembles of differential trees.

## Installation

To install the package from git:

`git clone https://github.com/gihannagalindez/boostdiff/.git  && cd boostdiff`

`pip install .`


## Data input

BoostDiff accepts two filenames corresponding to gene expression matrices from the disease and control conditions.
Each input is a tab-separated file. The rows correspond to genes, while the columns correspond to samples.
Note that the first column should be named "Gene".


| Gene  |   Disease1   |   Disease2  | Disease3  | 
|-------------------|-----------|-----------|------|
| ACE2   | 0.345  | 0.147  |0.267 | 
| APP   | 0.567  | 0.794  | 0.590 | 

| Gene  |   Control1   |   Control2  | Control3  | 
|-------------------|-----------|-----------|------|
| ACE2   | 0.084  | 0.147  |0.91 | 
| APP   | 0.567  | 0.794  | 0.590 | 


## Example


Import the BoostDiff package:

```python
from boostdiff.main import BoostDiff
```

### Run BoostDiff 

```python
file_control = "/tests/data/expr_control.txt"
file_disease = "/tests/data/expr_disease.txt"
output_folder = "/tests/output/"

n_estimators = 100
n_features = 50
n_subsamples = 50
keyword = "test"
n_processes = 2

model = BoostDiff()
model.run(file_disease, file_control, output_folder, n_processes,\
          n_estimators, n_features, n_subsamples, keyword=keyword)

```

BoostDiff will output two txt files and a folder containing the plots for the training progression for AdaBoost models.
The first txt file shows the data for the mean difference in prediction error between the disease and control samples after training the boosted differential trees.
The second txt file contains the raw output network.

###  Postprocessing

To obtain the final differential network, the raw network will be filtered for genes in which BoostDiff found a more predictive model for the disease condition.

```python
import boostdiff.postprocessing as pp

# Specify the output file containing the mean difference in prediction error after running the BoostDiff algorithm
file_diff = "/tests/output/differences.txt"
# Specify the output file containing the raw network output after running the BoostDiff algorithm
file_diff = "/tests/output/boostdiff_network.txt"

# Specify the output file for the histogram plot
file_histogram = "/tests/output/histogram.png"

# Plot the histogram
pp.plot_histogram(file_diff, file_histogram)

# Filter the raw output based on a user-specificed percentile (or threshold)
# Returns a pandas data frame after filtering for the threshold identified based on the 99th percentile.
df_filtered = pp.filter_network(file_raw_output, file_diff, p=99)

# Save the final differential network to file
file_output = /tests/output/filtered_network.txt
df_filtered.to_csv(file_output, sep="\t")
```


## Contact 
Gihanna Galindez: gihanna.galindez@plri.de
