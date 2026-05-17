import pandas as pd
import json
import os


def is_valid_value(v):
    if pd.isna(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() not in {"nan", "none", "null", "na"}


def csv_to_sentences(input_csv, output_path, output_format="csv"):
    # 读取 csv
    df = pd.read_csv(input_csv, index_col=0)

    # 以 dict 形式累积，方便下游 generate_textual_annotations.py 直接 raw_data.items() 消费
    results = {}

    for sample_id, row in df.iterrows():
        sample_id_str = str(sample_id).replace(".", "-")

        features_list = []
        for feature_name, value in row.items():
            if not is_valid_value(value):
                continue
            clean_key = str(feature_name).strip()
            clean_val = str(value).strip()
            features_list.append(f"{clean_key}: {clean_val}")

        final_text = "clinical text: " + "; \\n ".join(features_list)
        results[sample_id_str] = final_text

    if output_format == "csv":
        # CSV 仍然保留两列形式（sample_id, text），不破坏原有 CSV 兼容性
        rows = [{"sample_id": sid, "text": txt} for sid, txt in results.items()]
        pd.DataFrame(rows).to_csv(output_path, index=False)
    elif output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError("output_format 只能是 'csv' 或 'json'")

data_dir = '../../data/example_input'

# For mutation genes, you just need to concat gene symbols at the end of each sentence
# Example:
# "clinical text: AGE_AT_DX: 71.0 ; \n gene mutations: KEAP1, NF1"
csv_to_sentences(
    input_csv=os.path.join(data_dir, "raw_metadata.csv"),
    output_path=os.path.join(data_dir, "preprocessed_metadata.json"),
    output_format="json",   # 或 "json"
)
