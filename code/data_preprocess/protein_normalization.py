import json
import numpy as np
import pandas as pd
import os


def normalize_rppa_matrix_full(
    df: pd.DataFrame,
    log_transform: bool = True,
    c: float = 1.0
) -> pd.DataFrame:
    """
    Normalize the protein matrix with the median values of row and col
    """
    processed_df = df.copy()

    if log_transform:
        if (processed_df < 0).any().any():
            print("ERROR")
        processed_df = np.log2(processed_df + c)

    col_medians = processed_df.median(axis=0)
    centered_by_protein = processed_df.subtract(col_medians, axis=1)

    row_medians = centered_by_protein.median(axis=1)
    normalized_df = centered_by_protein.subtract(row_medians, axis=0)

    return normalized_df


def process_protein_csv(
    input_csv: str,
    output_csv: str,
    rename_map_path: str,
    normalize: bool = False,
    log_transform: bool = True,
    c: float = 1.0
) -> None:
    """
    处理单个 protein csv 文件：
    1. 读取 csv（行是 sample，列是 feature name）
    2. 用 protein_name_mapping.json 统一列名
    3. 若映射后出现重复 gene name，则按均值合并
    4. 可选做 normalize
    5. 保存为 csv
    """
    df = pd.read_csv(input_csv, index_col=0)

    with open(rename_map_path, "r") as f:
        rename_map = json.load(f)

    df.columns = df.columns.str.lower()
    cols_to_keep = [col for col in df.columns if col in rename_map]
    df = df[cols_to_keep]

    # protein name -> gene name
    df = df.rename(columns=rename_map)

    # 可选标准化
    if normalize:
        df = normalize_rppa_matrix_full(df, log_transform=log_transform, c=c)

    df.to_csv(output_csv)
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    data_dir = "../../data"
    input_csv = "example_input/protein_input.csv"
    output_csv = "example_input/protein_output.csv"
    rename_map_path = "gene_name_mapping/protein_name_mapping.json"

    process_protein_csv(
        input_csv=os.path.join(data_dir, input_csv),
        output_csv=os.path.join(data_dir, output_csv),
        rename_map_path=os.path.join(data_dir, rename_map_path),
        normalize=True,   # 改成 True 就会做 normalize
        log_transform=True,
        c=1.0
    )