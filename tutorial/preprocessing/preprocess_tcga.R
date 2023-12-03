
# Data sources: https://xenabrowser.net/ 

# ====================TCGA - BRCA====================
# Data sets > TCGA Breast Cancer (BRCA) 
# Download files:
# - Phenotypes (TCGA.BRCA.sampleMap%252FBRCA_clinicalMatrix, TCGA.BRCA.sampleMap%252FHiSeqV2)
# - gene expression RNAseq > IlluminaHiSeq (TCGA.BRCA.sampleMap%252FHiSeqV2)

# Load clinical data
file_meta = "TCGA.BRCA.sampleMap%252FBRCA_clinicalMatrix"
df = read.table(file_meta, sep="\t", header=TRUE)

names_ctrl = df[df$sample_type == "Solid Tissue Normal",]$sampleID
names_brca = df[df$sample_type == "Primary Tumor",]$sampleID

# Load expression data
file_expr = "TCGA.BRCA.sampleMap%252FHiSeqV2"
df_expr = read.table(file_expr, sep="\t", header=TRUE)
vector_names = colnames(df_expr)
vector_names <-  unlist(lapply(vector_names, function(x) gsub("\\.", "-", x)))
colnames(df_expr) = vector_names
head(df_expr)

common = intersect(names_brca, colnames(df_expr))
df_brca = subset(df_expr, select=c("sample",common))
colnames(df_brca)[colnames(df_brca) == 'sample'] <- 'Gene'

common = intersect(names_ctrl, colnames(df_expr))
df_ctrl = subset(df_expr, select=c("sample",common))
colnames(df_ctrl)[colnames(df_ctrl) == 'sample'] <- 'Gene'

# Write expression data
file_disease = "df_brca.txt"
file_ctrl = "df_ctrl_brca.txt"
write.table(df_brca, file = file_disease, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)
write.table(df_ctrl, file = file_ctrl, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)


# ====================TCGA - PRAD====================
# Data sets > TCGA Prostate Cancer (PRAD) 
# Download files: 
# - Phenotypes (TCGA.BRCA.sampleMap%252FBRCA_clinicalMatrix)
# - gene expression RNAseq > IlluminaHiSeq (TCGA.BRCA.sampleMap%252FHiSeqV2)

# Load clinical data
file_meta = "TCGA.PRAD.sampleMap%252FPRAD_clinicalMatrix"
df = read.table(file_meta, sep="\t", header=TRUE)

names_ctrl = df[df$sample_type == "Solid Tissue Normal",]$sampleID
names_prad = df[df$sample_type == "Primary Tumor",]$sampleID

# Load expression data
file_expr = "TCGA.PRAD.sampleMap%252FHiSeqV2"
df_expr = read.table(file_expr, sep="\t", header=TRUE)
vector_names = colnames(df_expr)
vector_names <-  unlist(lapply(vector_names, function(x) gsub("\\.", "-", x)))
colnames(df_expr) = vector_names
head(df_expr)

common = intersect(names_prad, colnames(df_expr))
df_disease = subset(df_expr, select=c("sample",common))
colnames(df_disease)[colnames(df_disease) == 'sample'] <- 'Gene'

common = intersect(names_ctrl, colnames(df_expr))
df_ctrl = subset(df_expr, select=c("sample",common))
colnames(df_ctrl)[colnames(df_ctrl) == 'sample'] <- 'Gene'

# Write expression data
file_disease = "df_prad.txt"
file_ctrl = "df_ctrl_prad.txt"
write.table(df_disease, file = file_disease, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)
write.table(df_ctrl, file = file_ctrl, append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)