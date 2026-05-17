import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple, Union

from openai import OpenAI

# ================= 配置区 =================
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "bft"

INPUT_JSON = "../../data/example_input/preprocessed_metadata.json"
OUTPUT_JSON_BASE  = "../../data/example_input/textual_annotations.json"

SAVE_INTERVAL = 1  # 建议设为 10 或 20，设为 1 硬盘IO太频繁

# ================= PROMPTS =================

SYSTEM_PROMPT = """You are an expert biological data extraction assistant.
Your task is to synthesize clinical metadata and gene mutation data into a concise, professional natural language summary.

# Constraints
- Do NOT fabricate information. If a specific data point (e.g., age, survival, treatment) is not present in the input, DO NOT mention it.
- The final output should be a single, well-structured paragraph.
- Your entire response must be strictly under 1000 characters.
"""

USER_TEMPLATE = """
Synthesize the biological information from the provided metadata (listed in <<<>>>) and gene mutations (listed in ||||||).

# Instructions

## Final Output Requirements
Construct a **single, cohesive natural language paragraph**. Integrate the following elements **only when valid data is present**:

1.  **Important clinical features (including but not limited to)**: 
    1.1 Age, Sex, Tissue, race, Disease or Normal/Control, disease Subtype, disease Stage, tumor Size, treatments (Surgery/Drugs), and Prognosis (Survival).
    **Constraint**: 
    a. Skip missing fields entirely. ** Do NOT write "Stage was not reported" or "Treatment details are unknown."
    b. Do not invent any clinical details not present in the provided text (e.g., the survival not included, please do not mention like "Prognosis is poor, with survival expected to be less than 6 months").
2.  **Gene Mutations (may be not available)**: 
    2.1 IF valid mutations exist: Append the sentence "The sample harbors mutations in [Genes]." followed by the functional descriptions (e.g., the genes function and/or the relationship between them and diseases).
    **Constraint**: If the mutation genes provided, the details of these genes (including the functions and/or relationships between them and disease) should be included in your response! Otherwise you will be punished.
    2.2 IF Gene mutations are empty/null or not provided:
     **Constraint**: say "no gene mutations were reported." 
3. Other critical information relevant to the sample (phenotype, genes, etc.).
4. Ignore Non-Biological Information. Ignore information that is not relevant to the sample. Specifically, exclude all IDs and details about technical and methodological processes such as sequencing type, 
read length, platform, or any other laboratory technique-related information, or any other non-biological details (e.g., obtained sources).

# Input Data
Clinical Metadata:
<<<
{clinical_text}
>>>

Gene Mutations:
|||
{gene_mutation}
|||
"""


# ## 1. Internal Reasoning (Inside <think>)
# - Identify the patient's demographics (Age, Sex) and Diagnosis (Histology, Grade, Stage).
# - Identify treatments (Chemo, Radiation, Surgery).
# - Identify survival outcomes (OS status, OS months).
# - If Gene mutations provided, identify gene mutations and recall their biological function.
# - **Filter out**: Technical keys (e.g., `_STROMAL_SCORE_`), IDs, and empty values. Ignore information that is not relevant to the sample. Specifically, exclude all IDs and details about technical and methodological processes such as sequencing type, read length, platform, or any other laboratory technique-related information, or any other non-biological details (e.g., obtained sources).
# - **Important!!!!**: Do not invent any clinical details not present in the provided text. All your response should follow the metadata and gene mutations.
# - **Check for Mutations**: Determine if valid gene mutations exist. If the input is "None", "null", "[]", or empty, flag to skip the genetics section.


# ## 2. Final Output Requirements
# Produce a **single natural language paragraph** covering available important information, including age, sex, disease or normal, disease subtypes, tissue, stage, tumor size, treatment, prognosis, mutation genes, etc:
# 1.  **Origin**: "This sample originates from a [Age]-year-old [Sex] diagnosed with [Disease/Subtype]..."
# 2.  **Clinical Course (ONLY IF AVAILABLE)**: Treatments received (or lack thereof) and surgery details.
# 3.  **Prognosis (ONLY IF AVAILABLE)**: Survival time and status.
# 4.  **Genetics (If provided)**: 
#     - **IF mutations are provided**: "The sample harbors mutations in [Genes]. [Gene Name] encodes [Function]..."
#     - **IF NO mutations gene are provided**: **JUST STOP.** DO NOT write "No mutations were identified". DO NOT mention genetics at all.

def parse_entry(text: str) -> Tuple[str, str]:
    low = text.lower()
    ct_key = "clinical text:"
    gm_key = "gene mutations:"
    clinical_text = text
    gene_mutation = ""
    try:
        if ct_key in low and gm_key in low:
            ct_start = low.index(ct_key) + len(ct_key)
            gm_start = low.index(gm_key)
            clinical_text = text[ct_start:gm_start].strip()
            gene_mutation = text[gm_start + len(gm_key):].strip()

        elif ct_key in low:
            ct_start = low.index(ct_key) + len(ct_key)
            clinical_text = text[ct_start:].strip()
    except Exception:
        pass
    return clinical_text, gene_mutation

def load_existing_results(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_file_atomic(path: str, tmp_path: str, data: Dict):
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[ERROR] 写入 {path} 失败: {e}")

def extract_final_response(ori_text: str) -> str:
    if not ori_text: 
        return "ERROR_EMPTY_INPUT"
    
    try:
        # 1. 先去除 Markdown 代码块标记，防止干扰
        text = ori_text.replace("```json", "").replace("```", "").strip()

        # 2. 处理包含 </think> 的情况
        if "</think>" in text:
            parts = text.split("</think>")
            
            # 策略 A：标准情况，取标签之后的内容
            candidate = parts[-1].strip()
            
            if len(candidate) > 0:
                clean_text = candidate
            else:
                # 策略 B (你的情况)：标签后为空，说明内容被包在里面了，或者标签写反了
                # 取标签之前的内容，并把开头的 <think> 去掉
                clean_text = parts[0].replace("<think>", "").strip()
        
        # 3. 处理没有 </think> 的情况
        else:
            # 这里的旧代码 re.sub(r"<think>.*", ...) 是致命的，如果 <think> 在开头，它会删光所有内容
            # 修正：直接把字符串 "<think>" 删掉，保留剩余文本
            clean_text = text.replace("<think>", "").strip()

        # 4. 最终检查
        if not clean_text: 
            # 如果还是空，打印原始文本的前100个字符用于调试
            print(f"[DEBUG] Empty result. Original start: {ori_text[:100]}...")
            return f"ERROR_EMPTY_AFTER_CLEAN \n {ori_text}"
        if " No gene mutations were identified." in clean_text:
            clean_text = clean_text.replace(" No gene mutations were identified", "")
        if " No genetic mutation data is available for this specimen." in clean_text:
            clean_text = clean_text.replace(" No genetic mutation data is available for this specimen.", "")
        if " No gene mutations are reported." in clean_text:
            clean_text = clean_text.replace("No gene mutations are reported.", "")
        if "No mutation data are available for this sample." in clean_text:
            clean_text = clean_text.replace("No mutation data are available for this sample.", "")
        if " No gene mutation data were provided." in clean_text:
            clean_text = clean_text.replace("No gene mutation data were provided.", "")
        if " No gene mutations were identified in this case." in clean_text:
            clean_text = clean_text.replace("No gene mutations were identified in this case.", "")
        if " No gene mutations are reported for this sample." in clean_text:
            clean_text = clean_text.replace("No gene mutations are reported for this sample.", "")
        if " no gene mutations are reported." in clean_text:
            clean_text = clean_text.replace("no gene mutations are reported.", "")
        if "No gene mutations were reported." in clean_text:
            clean_text = clean_text.replace("No gene mutations were reported.", "")
        if " no gene mutations were reported." in clean_text:
            clean_text = clean_text.replace("no gene mutations were reported.", "")
        if " No gene mutations were reported for this sample." in clean_text:
            clean_text = clean_text.replace("No gene mutations were reported for this sample.", "")
        clean_text = clean_text.replace("\u2011"," ")
        clean_text = clean_text.replace("\u202f","-")
        return clean_text.strip()

    except Exception as e:
        return f"ERROR_PROCESSING: {str(e)}"


def is_valid_result(res: str) -> bool:
    """判断已有的结果是否有效（不是错误信息且不为空）"""
    if not res:
        return False
    # 如果结果以 ERROR 开头，视为处理失败，需要重新检索处理
    if isinstance(res, str) and res.strip().startswith("ERROR"):
        return False
    return True

# ================= 主程序 =================
# 155055
def main():
    # 1. 命令行参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-process all IDs in range")

    # parser.add_argument("--start", type=int, default=0, required=False, help="Start index (inclusive)")
    # parser.add_argument("--end", type=int, default=40000, required=False, help="End index (exclusive)")
    # parser.add_argument("--part", type=str, default="part1", required=False, help="Suffix for output file (e.g., 'part1')")

    # parser.add_argument("--start", type=int, default=40000, required=False, help="Start index (inclusive)")
    # parser.add_argument("--end", type=int, default=80000, required=False, help="End index (exclusive)")
    # parser.add_argument("--part", type=str, default="part2", required=False, help="Suffix for output file (e.g., 'part1')")

    parser.add_argument("--start", type=int, default=80000, required=False, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=120000, required=False, help="End index (exclusive)")
    parser.add_argument("--part", type=str, default="part3", required=False, help="Suffix for output file (e.g., 'part1')")

    # parser.add_argument("--start", type=int, default=120000, required=False, help="Start index (inclusive)")
    # parser.add_argument("--end", type=int, default=160000, required=False, help="End index (exclusive)")
    # parser.add_argument("--part", type=str, default="part4", required=False, help="Suffix for output file (e.g., 'part1')")

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    if not os.path.exists(INPUT_JSON):
        print(f"[ERROR] Input file not found: {INPUT_JSON}")
        return

    # 2. 读取并排序原始数据 (确保多机运行时 ID 顺序一致)
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    all_items = sorted(raw_data.items())
    
    # 3. 确定当前分片范围
    start_idx = args.start
    end_idx = min(args.end, len(all_items)) if args.end else len(all_items)
    batch_items = all_items[start_idx:end_idx]

    # 4. 确定输出路径
    if args.part:
        current_output_json = OUTPUT_JSON_BASE.replace(".json", f"_{args.part}.json")
    else:
        current_output_json = OUTPUT_JSON_BASE
    current_tmp_json = current_output_json + ".tmp"

    # 5. 【核心增强】加载已有结果并进行对比检索
    print(f"\n[SCAN] Checking existing progress in: {current_output_json}")
    full_results = load_existing_results(current_output_json)
    
    todo_list = []
    already_done_count = 0
    
    for sample_id, combined_text in batch_items:
        # 如果不在结果中，或者结果是错误信息，或者开启了 --force
        if args.force or sample_id not in full_results or not is_valid_result(full_results[sample_id]):
            todo_list.append((sample_id, combined_text))
        else:
            already_done_count += 1

    print(f"==========================================")
    print(f"[INFO] Range: {start_idx} to {end_idx}")
    print(f"[INFO] Total IDs in this range: {len(batch_items)}")
    print(f"[INFO] Already processed (valid): {already_done_count}")
    print(f"[INFO] Remaining to process: {len(todo_list)}")
    print(f"==========================================\n")

    if not todo_list:
        print("[FINISH] All IDs in this range are already processed. Nothing to do.")
        return

    newly_processed_count = 0

    try:
        # 6. 只处理待处理列表
        for i, (sample_id, combined_text) in enumerate(todo_list):
            
            # 计算当前的全局索引位置，方便观察
            # 找到该 ID 在原始全量数据中的位置（可选，用于显示进度）
            current_idx_display = i + 1

            clinical_text, gene_mutation = parse_entry(combined_text)
            user_prompt = USER_TEMPLATE.format(clinical_text=clinical_text, gene_mutation=gene_mutation)

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    stop=[">>>>"], 
                    temperature=0.2,
                    max_tokens=8196,
                    frequency_penalty=1.2,
                )
                raw_content = resp.choices[0].message.content
                answer_full = extract_final_response(raw_content)
                
            except Exception as e:
                print(f"[WARN] {sample_id}: {e}")
                answer_full = f"ERROR: {e}"

            # 更新内存
            full_results[sample_id] = answer_full
            newly_processed_count += 1
            
            print(f"[{current_idx_display}/{len(todo_list)}] ID: {sample_id} done.")

            # 定时保存
            if newly_processed_count % SAVE_INTERVAL == 0:
                save_file_atomic(current_output_json, current_tmp_json, full_results)
                print(f"--- Progress Saved ({newly_processed_count} new) ---")

    except KeyboardInterrupt:
        print("\n[STOP] User interrupted. Saving progress...")
    except Exception as e:
        print(f"\n[CRITICAL] Error: {e}")
    finally:
        if newly_processed_count > 0:
            save_file_atomic(current_output_json, current_tmp_json, full_results)
            print(f"[FINALIZE] Saved {newly_processed_count} items. Total count in file: {len(full_results)}")
        else:
            print("No new data was processed.")

if __name__ == "__main__":
    main()
