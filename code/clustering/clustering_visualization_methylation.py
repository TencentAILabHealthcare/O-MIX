# %%
import json
import os
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from tqdm import trange

# 确保可以导入您的自定义模块
sys.path.append("../")
from omix.model.model_uni import PerformerModel
from omix.preprocess_bulk import Preprocessor  # 假设您的预处理类在这里
from omix.tokenizer import GeneVocab, tokenize_and_pad_batch

LOAD_MODEL_PATH = "../pretrain/save/methylation_pretrain"

DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
                         'KIRC', 'KIRP',  'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'READ', 'SARC', 'SKCM',
                         'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']

# 数据集所在的根目录
DATA_ROOT_DIR = "../../data/GDAC"

# 其他配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # 推理时可以设置大一点的batch size
N_BINS = 51  # 必须与模型训练时使用的n_bins一致
PAD_TOKEN = "<pad>"
PAD_VALUE = N_BINS
MASK_VALUE = N_BINS+1
MAX_SEQ_LEN = 12920 + 1  # 必须与模型训练时一致 (n_hvg + 1)
load_model = True
cell_emb_style = "cls"  # ["cls", "avg-pool", "w-pool"]
dropout = 0.0


def load_and_combine_data(disease_list: list, data_dir: str) -> anndata.AnnData:
    """
    加载多个数据集，添加pancancer标签，并合并它们。
    """
    used_gene_file = '../../data/GDAC/filtered_genes_list.csv'
    gene_list_df = pd.read_csv(used_gene_file)
    genes_to_keep = gene_list_df['gene_name'].tolist()

    used_gene_file_methyl = '../../data/GDAC/filtered_genes_list_methyl.csv'
    gene_list_df_methyl = pd.read_csv(used_gene_file_methyl)
    genes_to_keep_methyl = gene_list_df_methyl['gene_name'].tolist()

    adata_list = []
    print("Loading and combining data...")
    for disease in disease_list:
        print(f"-> Loading {disease}...")
        disease_path = os.path.join(data_dir, disease)
        if not os.path.exists(disease_path):
            print(f"Warning: Directory for {disease} not found. Skipping.")
            continue

        labels = pd.read_csv(os.path.join(disease_path, 'labels.csv'), sep='\t', index_col=0)
        mrna = pd.read_csv(os.path.join(disease_path, 'mRNA_TPM.csv'), index_col=0)
        methyl = pd.read_csv(os.path.join(disease_path, 'methylation.csv'), index_col=0)
        protein = pd.read_csv(os.path.join(disease_path, 'RPPA.csv'), index_col=0)

        genes_present_in_file = mrna.columns
        common_genes = [gene for gene in genes_to_keep if gene in genes_present_in_file]
        mrna = mrna[common_genes]

        genes_present_in_file_methyl = methyl.columns
        # common_genes_methyl = [gene for gene in genes_to_keep_methyl if gene in genes_present_in_file_methyl]
        # methyl = methyl[common_genes_methyl]


        # 数据预处理
        mrna.index = mrna.index.str.slice(0, 12).str.lower()
        mrna = mrna[~mrna.index.duplicated(keep='first')]

        # 数据预处理
        protein.index = protein.index.str.slice(0, 12).str.lower()
        protein = protein[~protein.index.duplicated(keep='first')]

        methyl.index = methyl.index.str.slice(0, 12).str.lower()
        methyl = methyl[~methyl.index.duplicated(keep='first')]

        # common_samples = labels.index.intersection(mrna.index).intersection(protein.index).intersection(methyl.index)
        common_samples = mrna.index.intersection(protein.index).intersection(methyl.index)
        if disease == 'COAD' or disease == 'READ':
            disease = 'COADREAD'

        if len(common_samples) == 0:
            print(f"Warning: No common samples found for {disease}. Skipping.")
            continue

        methyl = methyl.loc[common_samples]

        adata = anndata.AnnData(X=methyl)
        adata.var_names_make_unique()
        adata.var["gene_name"] = adata.var.index.tolist()

        adata.obs['pancancer_type'] = disease

        adata_list.append(adata)

    if not adata_list:
        raise ValueError("No data could be loaded. Please check your paths and disease list.")

    # 合并所有AnnData对象
    # join='outer' 保留所有数据集的基因，然后我们根据模型的词汇表进行过滤
    combined_adata = anndata.concat(adata_list, join='outer', label='batch', index_unique=None)
    combined_adata.obs_names_make_unique()

    print("Data loading and combination complete.")
    print("Combined data shape:", combined_adata.shape)
    print("Pancancer types loaded:", combined_adata.obs['pancancer_type'].unique().tolist())
    combined_adata.X = np.nan_to_num(combined_adata.X, nan=0.0)
    return combined_adata


def get_pancancer_embeddings(model: PerformerModel, adata: anndata.AnnData, vocab: GeneVocab) -> np.ndarray:
    """
    使用预训练模型为所有样本生成embeddings。
    """
    # model.eval()

    # 1. 预处理数据 (Binning)
    # 注意：这里的预处理器不应该过滤基因或细胞，只做数值转换
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=False,
        binning=N_BINS,
        result_binned_key="X_binned",
        repair=False,
        # filter_gene_by_variance=1.0
    )
    preprocessor(adata)

    input_data = adata.layers["X_binned"]

    # # raw data pca
    # adata_raw = anndata.AnnData(X=input_data, obs=pd.DataFrame(adata.obs['pancancer_type']))
    # run_raw_data_analysis(adata_raw)

    # 2. Tokenize
    gene_ids = np.array(vocab(adata.var.index.tolist()), dtype=int)
    tokenized_data = tokenize_and_pad_batch(
        input_data,
        gene_ids,
        max_len=MAX_SEQ_LEN,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=False,
    )

    all_gene_ids = tokenized_data['genes']
    all_values = tokenized_data['values']

    # 3. 批量推理以获取embeddings
    # num_samples = all_gene_ids.shape[0]
    # all_embeddings = []
    print("Generating embeddings for all samples...")

    src_key_padding_mask = all_gene_ids.eq(vocab[PAD_TOKEN])

    device = next(model.parameters()).device
    print(device)

    aa = np.array(all_gene_ids)
    bb = np.array(all_values)
    cc = np.array(src_key_padding_mask)

    # raw data pca
    # adata_raw = anndata.AnnData(X=bb, obs=pd.DataFrame(adata.obs['pancancer_type']))
    # run_raw_data_analysis(adata_raw)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):

        cell_embeddings = model.encode_batch(
            all_gene_ids,
            all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=BATCH_SIZE,
            time_step=0,
            return_np=True,
        )

    cell_embeddings = np.array(cell_embeddings)
    norm = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_embeddings = cell_embeddings / (norm + 1e-6)

    return cell_embeddings

    # return np.concatenate(all_embeddings, axis=0)

def main():
    """
    主函数: 加载模型和数据，计算指标，并进行可视化。
    """
    # --- 加载模型和词汇表 ---
    print("Loading pre-trained model and vocab...")
    # omix_dir = Path(omix_MODEL_PATH)
    model_dir = Path(LOAD_MODEL_PATH)
    file_name = 'model_e40.pt'
    model_config_file = model_dir / "args.json"
    model_file = model_dir / file_name
    vocab_file = model_dir / "vocab.json"

    print('loading from:', model_file)

    # model_config_file = model_dir / "args.json"
    # model_file = model_dir / "model_e1.pt"
    # vocab_file = model_dir / "vocab.json"


    if not all([model_config_file.exists(), model_file.exists(), vocab_file.exists()]):
        raise FileNotFoundError(f"Model files not found in {model_dir}. Please check the path.")

    special_tokens = [PAD_TOKEN, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    model = PerformerModel(
        ntoken=len(vocab),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        n_input_bins=N_BINS+2,
        cell_emb_style=cell_emb_style,  # 确保这与训练时一致 [cls, avg-pool, w-pool]
        # pre_norm=True,
        input_emb_style="category",
        # dropout=dropout
    )
    print(f'dropout: {dropout}')
    print(f'cell_emb_style: {cell_emb_style}')

    if load_model:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(DEVICE)
    model.eval()

    adata = load_and_combine_data(DISEASE_NAME_LIST, DATA_ROOT_DIR)

    # --- 获取Embeddings ---
    embeddings = get_pancancer_embeddings(model, adata, vocab)
    adata.obsm['X_omix'] = embeddings
    print("Embeddings stored in adata.obsm['X_omix']")

    # --- 运行聚类和计算指标 ---
    print("\n--- Clustering and Metrics ---")
    sc.pp.neighbors(adata, use_rep='X_omix', n_neighbors=30)
    sc.tl.leiden(adata, resolution=1.0, key_added='leiden_clusters')

    true_labels = adata.obs['pancancer_type']
    predicted_labels = adata.obs['leiden_clusters']

    # 计算指标
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    # 轮廓系数使用 (embeddings, 预测的簇标签) 来评估聚类结果的紧密程度
    silhouette = silhouette_score(embeddings, predicted_labels)

    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")

    # --- t-SNE 降维和可视化 ---
    print("\n--- Visualizing with t-SNE ---")
    sc.tl.tsne(adata, use_rep='X_omix')

    # 可视化1: 按真实的pancancer类型着色
    fig1 = sc.pl.tsne(
        adata,
        color='pancancer_type',
        title='t-SNE of Pancancer Embeddings by Cancer Type',
        show=False,
        return_fig=True
    )
    embedding_file = f'embeddings_methylation_pretrain{load_model}.png'
    fig1.savefig(embedding_file, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to {embedding_file}")
    #
    # # 可视化2: 按Leiden聚类结果着色
    # fig2 = sc.pl.tsne(
    #     adata,
    #     color='leiden_clusters',
    #     title='t-SNE of Pancancer Embeddings by Leiden Clusters',
    #     legend_loc='on data',
    #     show=False,
    #     return_fig=True
    # )
    # leiden_file = f'tsne_by_leiden_cluster_rna_pretrain{load_model}.png'
    # fig2.savefig(leiden_file, dpi=300, bbox_inches='tight')
    # print(f"t-SNE plot saved to {leiden_file}")


if __name__ == '__main__':
    main()
