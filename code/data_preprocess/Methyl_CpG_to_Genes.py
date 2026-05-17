import pandas as pd
import os

def map_cpg_to_gene(
    cpg_matrix_file: str,
    probe_to_gene_file: str,
    output_file: str,
    gene_col: str = "final_gene_name_hg38",
    probe_col: str = "IlmnID",
    sample_id_col: str = "sample_id",
    agg: str = "mean",
):
    """
    将 CpG-level 甲基化矩阵 CSV 映射为 gene-level 矩阵 CSV。
    """

    # 1) 读入数据
    cpg_df = pd.read_csv(cpg_matrix_file)
    map_df = pd.read_csv(probe_to_gene_file, sep="\t")

    # 2) 检查必要列
    if sample_id_col not in cpg_df.columns:
        raise ValueError(f"在 cpg_matrix_file 中找不到样本列: {sample_id_col}")

    if probe_col not in map_df.columns:
        raise ValueError(f"在 probe_to_gene_file 中找不到 probe 列: {probe_col}")

    if gene_col not in map_df.columns:
        raise ValueError(f"在 probe_to_gene_file 中找不到 gene 列: {gene_col}")

    # 3) 去掉 gene 为空的映射
    map_df = map_df.dropna(subset=[gene_col])

    # 4) 找出 cpg 矩阵中实际存在的 probe
    all_probe_cols = [c for c in cpg_df.columns if c != sample_id_col]
    map_df = map_df[map_df[probe_col].isin(all_probe_cols)].copy()

    if map_df.empty:
        raise ValueError("映射表中的 probe 在 cpg_matrix_file 里一个都没找到。")

    # 5) 建立 gene -> probes 映射
    gene_to_probes = map_df.groupby(gene_col)[probe_col].apply(list).to_dict()

    # 6) 构建 gene-level 矩阵
    result_df = pd.DataFrame()
    result_df[sample_id_col] = cpg_df[sample_id_col]

    for gene, probes in gene_to_probes.items():
        valid_probes = [p for p in probes if p in cpg_df.columns]

        if len(valid_probes) == 0:
            result_df[gene] = pd.NA
            continue

        if agg == "mean":
            result_df[gene] = cpg_df[valid_probes].to_numpy().mean(axis=1)
        else:
            raise ValueError(f"暂不支持的聚合方式: {agg}")

    gene_order = list(gene_to_probes.keys())
    result_df = result_df[[sample_id_col] + gene_order]

    result_df.to_csv(output_file, index=False)

    print(f"输入 CpG matrix shape: {cpg_df.shape}")
    print(f"有效 probe 数: {len(set(map_df[probe_col]))}")
    print(f"输出 gene matrix shape: {result_df.shape}")
    print(f"已保存到: {output_file}")


data_dir = "../../data"

if __name__ == "__main__":
    map_cpg_to_gene(
        cpg_matrix_file=os.path.join(data_dir, "example_input/cpg_matrix.csv"),
        probe_to_gene_file=os.path.join(data_dir, "gene_name_mapping/probe_to_gene.csv"),
        output_file=os.path.join(data_dir, "example_input/gene_matrix.csv"),
        gene_col="final_gene_name_hg38",
        sample_id_col="sample_id",
        agg="mean",
    )