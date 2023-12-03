library(GEOquery)
library(biomaRt)
library(dplyr)

# ====================COVID-19===================
# Data source: GEO database

# Get dataset
gset <- getGEO("GSE156063", GSEMatrix =TRUE)
res = pData(gset[[1]])

# Get sample names per condition
names_covid = res[res$`disease state:ch1` == "SC2",]$description
names_ctrl = res[res$`disease state:ch1` == "no virus",]$description

# Load the raw counts (Download the file manually from the GEO database)
file_raw = "GSE156063_swab_gene_counts.csv"
df_raw = read.table(file_raw, sep=",", header=TRUE)
df_raw = data.frame(df_raw, row.names = "X")
head(df_raw)

# vst
dds <- DESeqDataSetFromMatrix(countData = df_raw, colData = df_info, design = ~ disease_state)
vsd <- vst(dds, blind = FALSE)
df_expr = data.frame(assay(vsd))

# Map and aggregate
# Retrieve the mappings from biomart
mart <- useMart(biomart = "ENSEMBL_MART_ENSEMBL", host = "https://www.ensembl.org", path = "/biomart/martservice", dataset = "hsapiens_gene_ensembl")
hgnc_mapping <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"), filters = "ensembl_gene_id", values=rownames(df_expr), mart = mart)

indicesLookup <- match(rownames(df_expr), hgnc_mapping[["ensembl_gene_id"]])
df_expr$hgnc <- hgnc_mapping[indicesLookup, "hgnc_symbol"]
df_new <- df_expr %>% mutate_if(is.character, na_if, c('')) %>% na.omit
df_new <- df_new %>% group_by(hgnc) %>% summarise_all("mean")

df_covid = df_new[, c("hgnc", names_covid)]
df_ctrl = df_new[, c("hgnc", names_ctrl)]

# Write expression data
file_covid = "df_covid.txt"
file_novirus = "df_ctrl_covid.txt"
write.table(df_covid, file = file_covid, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)
write.table(df_novirus, file = file_novirus, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)
