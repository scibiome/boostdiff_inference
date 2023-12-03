
library(GEOquery)
library(biomaRt)
library(dplyr)

# ====================Crohn's disease===================
# Data source: GEO database

# Get dataset
gset <- getGEO("GSE126124", GSEMatrix =TRUE)
res = pData(gset[[1]])
df_expr = data.frame(exprs(gset[[1]]))

# Get data from colon tissue
df_tiss = res[res$`tissue:ch1` == "colon biopsy",]

names_ctrl = rownames(df_tiss[df_tiss$`disease type:ch1` == "Control",])
names_crohns = rownames(df_tiss[df_tiss$`disease type:ch1` == "Crohn's Disease",])

# Map and aggregate
# Retrieve the mappings from biomart
mart <- useMart(biomart = "ENSEMBL_MART_ENSEMBL", host = "https://www.ensembl.org", path = "/biomart/martservice", dataset = "hsapiens_gene_ensembl")
hgnc_mapping <- getBM(attributes = c("affy_hugene_1_0_st_v1", "hgnc_symbol"), filters = "affy_hugene_1_0_st_v1", values=rownames(df_expr), mart = mart)
indicesLookup <- match(rownames(df_expr), hgnc_mapping[["affy_hugene_1_0_st_v1"]])
df_expr$hgnc <- hgnc_mapping[indicesLookup, "hgnc_symbol"]
df_new <- df_expr %>% mutate_if(is.character, na_if, c('')) %>% na.omit
df_new <- df_new %>% group_by(hgnc) %>% summarise_all("mean")

# Create expression data for Crohn's and control
df_crohns = df_new[, c("hgnc", names_crohns)]
df_ctrl = df_new[, c("hgnc", names_ctrl)]

# Write expression data
file_crohns = "df_crohns.txt"
file_ctrl_crohns = "df_ctrl_crohns.txt"
write.table(df_crohns, file = file_crohns, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)
write.table(df_ctrl_crohns, file = file_ctrl_crohns, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)
