import pandas as pd
import numpy as np
import h5py
import time
from tqdm import tqdm
import json
import os

# This script convert the RNA expression matrix from raw counts into TPM format. 
# You only need to replace the "target_directory" and "mRNA_file" with your own data dir
# Ensure hg19 or hg38 you would like to use in "gene_info_file"


with open('../../data/gene_name_mapping/pc_genes.json', 'r') as f:
    used_genes = json.load(f).keys()

# Replace with your own data dir
target_directory = "../../data/GDAC"

# disease_name_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD', 'DLBC', 'ESCA', 'GBM', 'GBMLGG', 'HNSC', 'KICH', 'KIPAN',
#                          'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'READ', 'SARC', 'SKCM'
#                          'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

disease_name_list = ['LGG']
for disease_name in disease_name_list:
    print('processing ', disease_name)
    # Replace with your own data dir
    mRNA_file = os.path.join(target_directory, disease_name, f'gdac.broadinstitute.org_{disease_name}.mRNAseq_Preprocess.Level_3.2016012800.0.0', f'{disease_name}.uncv2.mRNAseq_raw_counts.txt')
    output_norm_file = os.path.join(target_directory, disease_name, 'mRNA_TPM.csv')


    mRNA_df = pd.read_table(mRNA_file)
    mRNA_df['gene_name'] = mRNA_df.iloc[:, 0].str.split('|').str[0]
    mRNA_df = mRNA_df[mRNA_df['gene_name'].isin(used_genes)]
    mRNA_df = mRNA_df.set_index('gene_name')
    mRNA_df = mRNA_df.drop(columns=['HYBRIDIZATION R'])
    data_gene_list = mRNA_df.index.tolist()

    # if your matrix belongs to hg38, just replace gene_info_file with gene_info_file = 'data/gene_name_mapping/hg38_merge_genelength_pc.csv'
    gene_info_file = 'data/gene_name_mapping/hg19_merge_genelength_pc.csv'

    # output_file = 'tpm_protein_coding_GDAC_.h5'

    print("--- Memory-Efficient TPM Normalization Script Started ---")

    # --- 2. loading gene length ---
    print(f"Step 1: Loading gene metadata from '{gene_info_file}'...")
    try:
        gene_info_df = pd.read_csv(gene_info_file)
        gene_info_df['gene_name_upper'] = gene_info_df['gene_name'].str.upper()
        gene_info_df = gene_info_df[gene_info_df['length'] > 0]
        duplicated_rows = gene_info_df.duplicated(subset=['gene_name'])
        print("Repeated Rows：")
        print(gene_info_df[duplicated_rows])
        print(len(gene_info_df))
        gene_info_df = gene_info_df.drop_duplicates(['gene_name'])
        print(len(gene_info_df))

        gene_info_df = gene_info_df.loc[gene_info_df['gene_name'].isin(used_genes)]
        gene_name_list = gene_info_df['gene_name'].values.tolist()
        gene_info_df = gene_info_df.set_index('gene_name')
        print(f"Loaded metadata for {len(gene_info_df)} protein-coding genes.")
    except FileNotFoundError:
        print(f"Error: The file '{gene_info_file}' was not found.")
        exit()

    print("Step 3: Aligning gene lists...")
    common_genes = np.intersect1d(data_gene_list, gene_name_list)
    n_common_genes = len(common_genes)
    print(f"Found {len(common_genes)} common protein-coding genes.")

    aligned_gene_info = gene_info_df.reindex(common_genes)
    mRNA_df = mRNA_df.loc[mRNA_df.index.isin(common_genes)]
    epsilon = 1e-9


    def calculate_tpm(
        counts_df: pd.DataFrame,
        gene_info_df: pd.DataFrame,
        length_col: str = 'length',
        min_expressed_samples: float=0.1, # If more than 90% of the samples express this gene, retain it; otherwise, remove it.
        min_count_per_sample: int = 1,
        epsilon: float = 1e-9
    ) -> pd.DataFrame:
        """
        计算TPM，并可选地在计算前进行低表达基因的过滤。

        Args:
            counts_df (pd.DataFrame): 原始Read Count矩阵。
                                      行索引是基因ID，列索引是样本ID。
            gene_info_df (pd.DataFrame): 基因信息表。
                                         行索引必须是与counts_df匹配的基因ID。
                                         必须包含一个基因长度列。
            length_col (str, optional): gene_info_df中包含基因长度的列名。默认为'length'。
            min_expressed_samples (int, optional):
                过滤阈值。一个基因必须在至少这么多数目的样本中被检测到才会被保留。
                如果为 None，则不执行此过滤。默认为 None。
            min_count_per_sample (int, optional):
                与 min_expressed_samples 配合使用。被“检测到”所需的最小read count。
                默认为 1 (即 count > 0)。
            epsilon (float, optional): 一个很小的数，用于防止除以零。默认为 1e-9。

        Returns:
            pd.DataFrame: 经过过滤和TPM标准化后的矩阵。
        """
        print("--- Starting TPM Calculation ---")
        print(f"Initial counts matrix shape: {counts_df.shape}")

        # --- 新增步骤：基于原始Counts进行基因过滤 ---
        if min_expressed_samples is not None:
            print(
                f"Applying pre-filtering: Keeping genes with counts >= {min_count_per_sample} in at least {int(min_expressed_samples*len(counts_df.columns))} samples.")
            # 计算每个基因在多少个样本中表达量 >= min_count_per_sample
            expressed_in_samples = (counts_df >= min_count_per_sample).sum(axis=1)
            # 找出通过过滤的基因
            genes_to_keep = expressed_in_samples[expressed_in_samples >= int(min_expressed_samples*len(counts_df.columns))].index

            # 应用过滤
            counts_filtered = counts_df.loc[genes_to_keep]
            print(f"Shape after filtering: {counts_filtered.shape} ({len(genes_to_keep)} genes kept)")
        else:
            print("No pre-filtering applied.")
            counts_filtered = counts_df.copy()  # 保持变量名一致

        # --- 1. 数据对齐和过滤 ---
        # 找出共有的基因，并过滤掉长度小于等于0的基因
        valid_genes_by_length = gene_info_df[gene_info_df[length_col] > 0].index
        common_genes = counts_filtered.index.intersection(valid_genes_by_length)

        # 筛选出共有的、有效的基因，并确保顺序一致
        counts_aligned = counts_filtered.loc[common_genes]
        gene_lengths = gene_info_df.loc[common_genes, length_col]
        print(f"Shape after aligning with gene lengths: {counts_aligned.shape}")

        # --- 2. 计算RPK (Reads Per Kilobase) ---
        gene_lengths_kb = gene_lengths / 1000.0
        rpk_df = counts_aligned.div(gene_lengths_kb + epsilon, axis=0)

        # --- 3. 计算TPM (Transcripts Per Million) ---
        scaling_factor = rpk_df.sum(axis=0) / 1_000_000
        tpm_df = rpk_df.div(scaling_factor + epsilon, axis=1)

        print("--- TPM Calculation Finished ---")
        return tpm_df.T

    tpm_result = calculate_tpm(mRNA_df, aligned_gene_info)
    tpm_result.to_csv(output_norm_file)
    # dd = pd.read_csv(output_norm_file, index_col=0)
    print('ok')
