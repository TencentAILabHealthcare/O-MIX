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
from anndata import AnnData
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from tqdm import trange

# 确保可以导入您的自定义模块
sys.path.append("../")
from omix.model.omni_fusion import \
    OmniFusionBlock
from omix.model.model_multi import PerformerModel
from omix.preprocess_bulk import Preprocessor as Preprocessor_RNA
from omix.preprocess_rppa import Preprocessor as Preprocessor_Protein
from omix.tokenizer import GeneVocab, tokenize_and_pad_batch
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

# 指向您训练好的模型文件夹 (包含 best_model.pt, args.json, vocab.json)
LOAD_MODEL_PATH = "../pretrain/save/omix_o_pretrain"
used_dir = LOAD_MODEL_PATH.split('/')[1]

DISEASE_NAME_LIST = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH',
                         'KIRC', 'KIRP',  'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'READ', 'SARC', 'SKCM',
                         'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']

# 数据集所在的根目录
DATA_ROOT_DIR = "../../data/GDAC"

gpu_index = 0
# --- 模型和数据处理参数 ---
DEVICE = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(gpu_index)
BATCH_SIZE = 4  # 对于Cox Loss，较小的batch size可能导致梯度不稳定，可以根据情况调整
N_BINS = 51
PAD_TOKEN = "<pad>"
PAD_VALUE = N_BINS
MASK_VALUE = N_BINS + 1
max_seq_len_protein = 17007+1
max_seq_len_rna = 19205+1
max_seq_len_methyl = 12920 + 1
RENAME_MAP_PATH = '../../data/gene_name_mapping/protein_name_mapping.json'
load_model = True
cell_emb_style = "cls"  # ["cls", "avg-pool", "w-pool"]
dropout = 0.0


def run_raw_data_analysis(adata_raw):
    """
    对原始数据进行聚类和可视化分析。
    这个函数会创建一个数据的副本进行操作，以免影响后续 O-MIX 的分析。
    """

    print("运行PCA进行降维...")
    sc.tl.pca(adata_raw, svd_solver='arpack')

    # --- 2. 聚类和计算指标 ---
    print("\n--- 在PCA降维结果上进行聚类和评估 ---")
    sc.pp.neighbors(adata_raw, use_rep='X_pca', n_neighbors=30)
    sc.tl.leiden(adata_raw, resolution=1.0, key_added='leiden_raw')

    true_labels = adata_raw.obs['pancancer_type']
    predicted_labels = adata_raw.obs['leiden_raw']

    # 计算指标
    ari_raw = adjusted_rand_score(true_labels, predicted_labels)
    nmi_raw = normalized_mutual_info_score(true_labels, predicted_labels)
    # 轮廓系数应在用于聚类的PCA空间中计算
    silhouette_raw = silhouette_score(adata_raw.obsm['X_pca'], predicted_labels)

    print(f"【原始数据】Adjusted Rand Index (ARI): {ari_raw:.4f}")
    print(f"【原始数据】Normalized Mutual Information (NMI): {nmi_raw:.4f}")
    print(f"【原始数据】Silhouette Score (基于PCA): {silhouette_raw:.4f}")

    # --- 3. t-SNE 降维和可视化 ---
    print("\n--- 对原始数据聚类结果进行 t-SNE 可视化 ---")
    sc.tl.tsne(adata_raw, use_rep='X_pca')

    # 可视化1: 按真实的pancancer类型着色
    fig_raw_truth = sc.pl.tsne(
        adata_raw,
        color='pancancer_type',
        title='t-SNE of Raw Data by Cancer Type (PCA-based)',
        show=False,
        return_fig=True
    )
    raw_truth_file = 'tsne_raw_by_cancer_type_rna.png'
    fig_raw_truth.savefig(raw_truth_file, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to {raw_truth_file}")


def load_and_combine_data(disease_list: list, data_dir: str, rename_map: dict):
    """
    加载指定疾病的生存数据和基因表达数据。
    """
    used_gene_file = '../../data/GDAC/filtered_genes_list.csv'
    gene_list_df = pd.read_csv(used_gene_file)
    genes_to_keep = gene_list_df['gene_name'].tolist()

    used_gene_file_methyl = '../../data/GDAC/filtered_genes_list_methyl.csv'
    gene_list_df_methyl = pd.read_csv(used_gene_file_methyl)
    genes_to_keep_methyl = gene_list_df_methyl['gene_name'].tolist()

    adata_list = []
    adata_protein_list = []
    adata_methyl_list = []
    for disease in disease_list:
        print(f"Loading survival data for {disease}...")
        disease_path = os.path.join(data_dir, disease)
        if not os.path.exists(disease_path):
            raise FileNotFoundError(f"Data directory for {disease} not found at {disease_path}")

        # 加载标签（生存信息）和Protein数据
        # *** 关键: 假设labels.csv包含'OS.time'和'OS'列 ***
        labels = pd.read_csv(os.path.join(disease_path, 'labels.csv'), sep='\t', index_col=0)
        mrna = pd.read_csv(os.path.join(disease_path, 'mRNA_TPM.csv'), index_col=0)
        methyl = pd.read_csv(os.path.join(disease_path, 'methylation.csv'), index_col=0)
        protein = pd.read_csv(os.path.join(disease_path, 'RPPA.csv'), index_col=0)

        genes_present_in_file = mrna.columns
        common_genes = [gene for gene in genes_to_keep if gene in genes_present_in_file]
        # mrna = mrna[common_genes]

        genes_present_in_file_methyl = methyl.columns
        common_genes_methyl = [gene for gene in genes_to_keep_methyl if gene in genes_present_in_file_methyl]
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

        mrna = mrna.loc[common_samples]
        methyl = methyl.loc[common_samples]
        
        cols_to_keep = [c for c in protein.columns if c in rename_map]
        protein = protein[cols_to_keep]
        protein = protein.loc[common_samples].rename(columns=rename_map)

        
        if disease == 'COAD' or disease == 'READ':
            disease_name = 'COADREAD'
        else:
            disease_name = disease

        adata = AnnData(X=mrna)
        adata.var_names_make_unique()
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['pancancer_type'] = disease_name

        adata.X = np.nan_to_num(adata.X, nan=0.0)
        print(f"Data loading complete. Found {adata.n_obs} samples.")

        adata_protein = AnnData(X=protein)
        adata_protein.var_names_make_unique()
        adata_protein.var["gene_name"] = adata_protein.var.index.tolist()
        adata_protein.obs['pancancer_type'] = disease_name

        adata_methyl = AnnData(X=methyl)
        adata_methyl.var_names_make_unique()
        adata_methyl.var["gene_name"] = adata_methyl.var.index.tolist()
        adata_methyl.obs['pancancer_type'] = disease_name

        adata_list.append(adata)
        adata_methyl_list.append(adata_methyl)
        adata_protein_list.append(adata_protein)

    combined_adata = anndata.concat(adata_list, join='outer', label='batch', index_unique=None)
    combined_adata.obs_names_make_unique()
    combined_adata.X = np.nan_to_num(combined_adata.X, nan=0.0)
    combined_adata_protein = anndata.concat(adata_protein_list, join='outer', label='batch', index_unique=None)
    combined_adata_protein.obs_names_make_unique()
    combined_adata_methyl = anndata.concat(adata_methyl_list, join='outer', label='batch', index_unique=None)
    combined_adata_methyl.obs_names_make_unique()
    return combined_adata, combined_adata_protein, combined_adata_methyl


def get_pancancer_embeddings_flexible(fusion_model, encoder_dict, data_dict, vocab_mod, config) -> np.ndarray:
    """
    使用预训练模型为所有样本生成 embeddings (适配任意输入)。

    Args:
        fusion_model: 融合模型。
        encoder_dict: 包含各个模态编码器的字典，键为模态名称。
        data_dict: 一个字典，包含所有模态的数据。
                   格式: {'modality_name': (all_gene_ids, all_values, src_key_padding_mask)}
                   例如: {'RNA': (all_gene_ids_rna, all_values_rna, src_key_padding_mask_rna),
                          'Protein': (all_gene_ids_protein, all_values_protein, src_key_padding_mask_protein)}

    Returns:
        np.ndarray: 所有样本的细胞嵌入向量。
    """
    fusion_model.eval()
    for encoder in encoder_dict.values():
        encoder.eval()

    device = next(fusion_model.parameters()).device
    print(f"Model is on device: {device}")

    # 从 data_dict 中提取模态名称和数据
    modalities = list(data_dict.keys())
    # 动态构建 TensorDataset 的输入列表
    # -> [ids_rna, values_rna, mask_rna, ids_protein, values_protein, mask_protein, ...]
    tensors = [tensor for modality in modalities for tensor in data_dict[modality]]

    full_dataset = TensorDataset(*tensors)
    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cell_embeddings = []
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Generating Embeddings"):
            # batch_data 是一个元组，包含了所有模态的张量
            # 需要根据模态数量将其拆分
            
            fusion_input_cls = []  # 存放经过 adapter 的 CLS token
            mod_types_list = []

            batch_size = batch_data[0].shape[0]
            
            # 每个模态有3个张量（ids, values, mask）
            num_tensors_per_modality = 3
            
            for i, modality in enumerate(modalities):
                # 从 batch_data 中切片出当前模态的数据
                start_idx = i * num_tensors_per_modality
                ids, values, padding_mask = batch_data[start_idx : start_idx + num_tensors_per_modality]

                # 将数据移动到正确的设备
                ids, values, padding_mask = ids.to(device), values.to(device), padding_mask.to(device)

                # 使用相应的编码器进行编码
                encoded_output, _ = encoder_dict[modality](
                    src=ids,
                    values=values,
                    src_key_padding_mask=padding_mask,
                    return_seq_embedding=True
                )

                if hasattr(fusion_model, "module"):
                    encoded_samples = fusion_model.module.run_adapter(modality, encoded_output)
                else:
                    encoded_samples = fusion_model.run_adapter(modality, encoded_output)

                cls_token = encoded_samples[:, 0, :].unsqueeze(1).clone()
                fusion_input_cls.append(cls_token)
                
                # 5. 构建 mod_types
                # train 函数中: num_patches = modality_patch_counts[modality]
                # 但因为我们这里传入的是 CLS token (长度为1)，所以对应的 type id 长度也应为 1
                mod_id = vocab_mod[modality]
                current_mod_types = torch.full((batch_size, 1), mod_id, dtype=torch.long, device=device)
                mod_types_list.append(current_mod_types)
            
            mod_types = torch.cat(mod_types_list, dim=1)
            # 假设所有样本的 expert_indices 都为 0
            batch_mcs = torch.tensor([0] * batch_data[0].shape[0]).to(device)

            # 使用解包操作符 (*) 将列表传递给模型
            cell_embeddings_batch = fusion_model(*fusion_input_cls, expert_indices=batch_mcs, mod_types=mod_types, return_cell_embedding=True)
            # print(cell_embeddings_batch.shape)
            cell_embeddings.append(cell_embeddings_batch.cpu().numpy())

    # 将所有批次的结果拼接起来
    cell_embeddings = np.concatenate(cell_embeddings, axis=0)

    # L2 归一化
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
    model_dir = Path(LOAD_MODEL_PATH)
    file_name = 'model_e8.pt'
    model_config_file = model_dir / "args.json"
    model_file = model_dir / file_name
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    if "modality_dict" in model_configs:
        vocab_mod = {k: v for k, v in model_configs["modality_dict"].items()}
        # 确保有 PAD
        if PAD_TOKEN not in vocab_mod:
            vocab_mod[PAD_TOKEN] = len(vocab_mod)
    print('loading from:', model_file)

    special_tokens = [PAD_TOKEN, "<cls>", "<eoc>"]
    vocab_dict = {}

    vocab_file_rna = model_dir / "vocab_rna.json"

    vocab_RNA = GeneVocab.from_file(vocab_file_rna)
    for s in special_tokens:
        if s not in vocab_RNA:
            vocab_RNA.append_token(s)

    vocab_dict['RNA'] = vocab_RNA

    vocab_file = model_dir / "vocab_protein.json"

    vocab_Protein = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab_Protein:
            vocab_Protein.append_token(s)

    vocab_dict['Protein'] = vocab_Protein

    vocab_file = model_dir / "vocab_methyl.json"

    vocab_methyl = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab_methyl:
            vocab_methyl.append_token(s)

    vocab_dict['METHYL'] = vocab_methyl

    with open(RENAME_MAP_PATH, 'r') as f:
        rename_map = json.load(f)

    adata, adata_protein, adata_methyl = load_and_combine_data(DISEASE_NAME_LIST, DATA_ROOT_DIR, rename_map)
    preprocessor = Preprocessor_RNA(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=True,
        binning=51,
        result_binned_key="X_binned",
        repair=False,
        # filter_gene_by_variance=1.0
    )
    preprocessor(adata)

    preprocessor_protein = Preprocessor_Protein(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=False,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=False,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat",  # bulk 用seurat
        binning=51,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor_protein(adata_protein)

    preprocessor = Preprocessor_RNA(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=False,
        binning=51,
        result_binned_key="X_binned",
        repair=False,
        # filter_gene_by_variance=1.0
    )
    preprocessor(adata_methyl)


    gene_ids_RNA = np.array(vocab_RNA(adata.var_names.tolist()), dtype=int)
    proteins_ids_Protein = np.array(vocab_Protein(adata_protein.var_names.tolist()), dtype=int)
    gene_ids_methyl = np.array(vocab_methyl(adata_methyl.var_names.tolist()), dtype=int)

    input_layer_key = "X_binned"
    RNA_data = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    Protein_data = (
        adata_protein.layers[input_layer_key].toarray()
        if issparse(adata_protein.layers[input_layer_key])
        else adata_protein.layers[input_layer_key]
    )
    methyl_data = (
        adata_methyl.layers[input_layer_key].toarray()
        if issparse(adata_methyl.layers[input_layer_key])
        else adata_methyl.layers[input_layer_key]
    )

    tokenized_RNA = tokenize_and_pad_batch(
        RNA_data,
        gene_ids_RNA,
        max_len=max_seq_len_rna,
        vocab=vocab_RNA,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=False,
        # mod_type=mod_type if config.use_mod else None,
        # vocab_mod=vocab_mod if config.use_mod else None,
    )
    tokenized_Protein = tokenize_and_pad_batch(
        Protein_data,
        proteins_ids_Protein,
        max_len=max_seq_len_protein,
        vocab=vocab_Protein,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=False,
        # mod_type=mod_type if config.use_mod else None,
        # vocab_mod=vocab_mod if config.use_mod else None,
    )
    tokenized_methyl = tokenize_and_pad_batch(
        methyl_data,
        gene_ids_methyl,
        max_len=max_seq_len_methyl,
        vocab=vocab_methyl,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=False,
        # mod_type=mod_type if config.use_mod else None,
        # vocab_mod=vocab_mod if config.use_mod else None,
    )

    total_length = 3

    all_gene_ids_rna = tokenized_RNA['genes']
    all_values_rna = tokenized_RNA['values']
    src_key_padding_mask_rna = all_gene_ids_rna.eq(vocab_RNA[PAD_TOKEN])

    all_gene_ids_protein = tokenized_Protein['genes']
    all_values_protein = tokenized_Protein['values']
    src_key_padding_mask_protein = all_gene_ids_protein.eq(vocab_Protein[PAD_TOKEN])

    all_gene_ids_methyl = tokenized_methyl['genes']
    all_values_methyl = tokenized_methyl['values']
    src_key_padding_mask_methyl = all_gene_ids_methyl.eq(vocab_methyl[PAD_TOKEN])

    encoder_dict = {}

    encoder_dict['RNA'] = PerformerModel(
        ntoken=len(vocab_RNA),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab_RNA,
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        n_input_bins=model_configs["n_bins"] + 2,
        cell_emb_style="cls",
        input_emb_style="category"
    )
    encoder_dict['Protein'] = PerformerModel(
        ntoken=len(vocab_Protein),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab_Protein,
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        n_input_bins=model_configs["n_bins"] + 2,
        cell_emb_style="cls",
        input_emb_style="category"
    )
    encoder_dict['METHYL'] = PerformerModel(
        ntoken=len(vocab_methyl),
        d_model=model_configs["layer_size"],
        nhead=model_configs["nhead"],
        d_hid=model_configs["layer_size"],
        nlayers=model_configs["nlayers"],
        vocab=vocab_methyl,
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        n_input_bins=model_configs["n_bins"] + 2,
        cell_emb_style="cls",
        input_emb_style="category"
    )
    print(f'dropout: {dropout}')
    print(f'cell_emb_style: {cell_emb_style}')
    full_modality_index = 0
    fusion_model = OmniFusionBlock(model_configs["num_modalities"], full_modality_index, total_length,
                           model_configs["layer_size"], model_configs["num_layers_fus"], model_configs["num_experts"],
                           model_configs["num_routers"], model_configs["top_k"], model_configs["nhead"],
                           model_configs["dropout"], vocab_mod)

    if load_model:
        print(f'Loading pre-trained weight from {model_file}')
        pretrained_state = torch.load(model_file, map_location='cpu')

        model_dict_RNA = encoder_dict['RNA'].state_dict()
        pretrained_dict_RNA = {
            k: v
            for k, v in pretrained_state['encoder_dict']['RNA'].items()
            if k in model_dict_RNA and v.shape == model_dict_RNA[k].shape
        }
        model_dict_RNA.update(pretrained_dict_RNA)
        encoder_dict['RNA'].load_state_dict(model_dict_RNA)

        model_dict_Protein = encoder_dict['Protein'].state_dict()
        pretrained_dict_Protein = {
            k: v
            for k, v in pretrained_state['encoder_dict']['Protein'].items()
            if k in model_dict_Protein and v.shape == model_dict_Protein[k].shape
        }
        model_dict_Protein.update(pretrained_dict_Protein)
        encoder_dict['Protein'].load_state_dict(model_dict_Protein)

        model_dict_methyl = encoder_dict['METHYL'].state_dict()
        pretrained_dict_methyl = {
            k: v
            for k, v in pretrained_state['encoder_dict']['METHYL'].items()
            if k in model_dict_methyl and v.shape == model_dict_methyl[k].shape
        }
        model_dict_methyl.update(pretrained_dict_methyl)
        encoder_dict['METHYL'].load_state_dict(model_dict_methyl)
        for k, v in model_dict_methyl.items():
            print(f"Loading params from METHYL: {k} with shape {v.shape}")

        model_dict_fusion = fusion_model.state_dict()
        pretrained_dict_fusion = {
            k: v
            for k, v in pretrained_state['fusion_model'].items()
            if k in model_dict_fusion and v.shape == model_dict_fusion[k].shape
        }
        print('loading fusion_model state dict')
        for k, v in pretrained_dict_fusion.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict_fusion.update(pretrained_dict_fusion)
        fusion_model.load_state_dict(model_dict_fusion)

    for key in encoder_dict:
        encoder_dict[key] = encoder_dict[key].to(DEVICE)
    fusion_model = fusion_model.to(DEVICE)

    data_dict = {
        'RNA': (all_gene_ids_rna, all_values_rna, src_key_padding_mask_rna),
        'Protein': (all_gene_ids_protein, all_values_protein, src_key_padding_mask_protein),
        'METHYL': (all_gene_ids_methyl, all_values_methyl, src_key_padding_mask_methyl)
    }

    embeddings = get_pancancer_embeddings_flexible(fusion_model, encoder_dict, data_dict, vocab_mod, model_configs)

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
    embedding_file = f'embeddings_multimodal_pretrain{load_model}_{used_dir}_{file_name}.png'
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
