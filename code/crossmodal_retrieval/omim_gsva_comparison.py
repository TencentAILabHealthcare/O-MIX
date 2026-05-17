import json
from pathlib import Path
import sys
sys.path.append("../")
import pandas as pd
from scipy.stats import zscore

with open('../../data/cellwhisper/human_disease/OMIM_gene_score.json', 'r') as f:
    omim_gene_sets = json.load(f)

used_gene_sets = omim_gene_sets.keys()

# 读取 parquet 文件
df = pd.read_parquet('../../data/cellwhisper/human_disease/gsva.parquet')

gene_set_col = df.columns[0] 
df = df.set_index(gene_set_col)

valid_cancer_sets = [gs for gs in used_gene_sets if gs in df.index]

print(f"最终在 Parquet 中匹配到的 Cancer 基因集数量: {len(valid_cancer_sets)}")

# 提取这些行(187, 14113)
cancer_df = df.loc[valid_cancer_sets]



save_path = "../../data/cellwhisper/human_disease_20260115/disease_similarity_matrix_youtu_omics_frozen_omix_t_pretrain_model_e5.pt.csv"


example_disease = 'wilms tumor' 
paneld_file = './figures/gsva_compare_similarity_omics_frozen_panel_d.png'
cdf_path = f'./figures/gsva_cdf_{example_disease.replace(" ", "_")}_omics_frozen.png'
scatter_path = f'./figures/gsva_scatter_{example_disease.replace(" ", "_")}_omics_frozen.png'

# 187, 14113
similarity_df_ori = pd.read_csv(save_path)

# 处理 similarity_df 的索引 (read_csv 默认第一列可能是 row name)
if 'Unnamed: 0' in similarity_df_ori.columns:
    similarity_df_ori = similarity_df_ori.set_index('Unnamed: 0')

similarity_df = pd.DataFrame(
    zscore(similarity_df_ori, axis=1),
    index=similarity_df_ori.index,
    columns=similarity_df_ori.columns
)
# 检查一下：现在应该有正有负了
print(f"归一化前范围: [{similarity_df_ori.iloc[0].min():.4f}, {similarity_df_ori.iloc[0].max():.4f}]")
print(f"归一化后范围: [{similarity_df.iloc[0].min():.4f}, {similarity_df.iloc[0].max():.4f}]")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import ks_2samp, pearsonr

# ==============================================================================
# 1. 数据加载与对齐 (Data Loading & Alignment)
# ==============================================================================

# 假设你之前的代码已经运行，有了 cancer_df (GSVA) 和 similarity_df (AI)
# 这里我们做关键的对齐检查，防止列名错位



# 1. 取交集：确保行(疾病)和列(样本)完全一致
common_diseases = cancer_df.index.intersection(similarity_df.index)
common_samples = cancer_df.columns.intersection(similarity_df.columns)

print(f"对齐前形状: GSVA {cancer_df.shape}, AI {similarity_df.shape}")

# 2. 重排数据：保证顺序严格一致
df_gsva = cancer_df.loc[common_diseases, common_samples].astype(float)
df_ai = similarity_df.loc[common_diseases, common_samples].astype(float)

print(f"对齐后形状: {df_gsva.shape} (Samples: {len(common_samples)})")

# ==============================================================================
# 2. 定义统计指标计算函数
# ==============================================================================

def calculate_signed_ks(ai_scores, gsva_scores):
    """
    计算 Signed KS-statistic。
    逻辑：将样本分为 AI_score > 0 (Selected) 和 <= 0 (Others)，比较两组 GSVA 分数分布。
    """
    # 分组
    mask_pos = ai_scores > 0
    gsva_pos = gsva_scores[mask_pos] # Selected samples
    gsva_neg = gsva_scores[~mask_pos] # All other samples
    
    if len(gsva_pos) == 0 or len(gsva_neg) == 0:
        return 0.0, 1.0 # 无法计算
    
    # 计算 KS 统计量 (距离)
    ks_stat, p_val = ks_2samp(gsva_pos, gsva_neg)
    
    # 确定符号：如果 Positive 组的 GSVA 中位数更大，则是正相关
    sign = np.sign(np.median(gsva_pos) - np.median(gsva_neg))
    if sign == 0: sign = 1
    
    return ks_stat * sign, p_val

# ==============================================================================
# 3. 绘制 Panel b (单个疾病详情: 散点图 + CDF)
# ==============================================================================

def plot_panel_b(disease_name, df_ai, df_gsva):
    """
    完美复现 Extended Data Fig. 2b
    1. 散点图颜色根据 >0 划分 (红/蓝)
    2. CDF 图添加 KS statistic 垂直双向箭头
    """
    if disease_name not in df_ai.index:
        print(f"Error: {disease_name} 不在数据中")
        return

    display_name = "Wilms tumor" if disease_name.lower() == "wilms tumor" else disease_name

    # 1. 提取数据
    vec_ai = df_ai.loc[disease_name].values
    vec_gsva = df_gsva.loc[disease_name].values
    
    # 2. 计算指标
    pcc, p_val_pcc = pearsonr(vec_gsva, vec_ai)
    
    # 分组
    mask_pos = vec_ai > 0
    gsva_pos = vec_gsva[mask_pos]   # Red group
    gsva_neg = vec_gsva[~mask_pos]  # Blue group
    
    ks_stat, p_val_ks = ks_2samp(gsva_pos, gsva_neg)
    
    # 确定 Signed KS 符号
    sign = np.sign(np.median(gsva_pos) - np.median(gsva_neg))
    signed_ks = ks_stat * (sign if sign != 0 else 1)

    # ==========================================
    # 图 1: 散点图 (Scatter Plot)
    # ==========================================
    plt.figure(figsize=(6, 5))  # 单个图的大小
    
    # 定义颜色：>0 为红色, <=0 为蓝色
    colors = np.where(vec_ai > 0, '#d62728', '#1f77b4')
    
    # 画点
    plt.scatter(vec_gsva, vec_ai, c=colors, s=5, alpha=0.5)
    
    # 辅助线和标签
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('GSVA score for gene set', fontsize=20)
    plt.ylabel('O-MIX score', fontsize=20)
    plt.title(f'PCC for {display_name}: {pcc:.2f}', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 保存散点图
    
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close() # 关闭当前画布，防止重叠
    print(f"散点图已保存至: {scatter_path}")

    # ==========================================
    # 图 2: CDF 图 (CDF Plot)
    # ==========================================
    plt.figure(figsize=(7, 5)) # 单个图的大小
    
    # 画 ECDF 线
    sns.ecdfplot(gsva_neg, color='#1f77b4', label='Negative (AI score ≤ 0)', linewidth=2)
    sns.ecdfplot(gsva_pos, color='#d62728', label='Positive (AI score > 0)', linewidth=2)
    
    # --- 计算箭头位置 ---
    all_points = np.sort(np.concatenate([gsva_pos, gsva_neg]))
    cdf_pos = np.searchsorted(np.sort(gsva_pos), all_points, side='right') / len(gsva_pos)
    cdf_neg = np.searchsorted(np.sort(gsva_neg), all_points, side='right') / len(gsva_neg)
    
    diff = np.abs(cdf_pos - cdf_neg)
    max_idx = np.argmax(diff)
    
    arrow_x = all_points[max_idx]
    arrow_y_min = min(cdf_pos[max_idx], cdf_neg[max_idx])
    arrow_y_max = max(cdf_pos[max_idx], cdf_neg[max_idx])
    
    # 画双向箭头
    plt.annotate(
        '', 
        xy=(arrow_x, arrow_y_max), 
        xytext=(arrow_x, arrow_y_min),
        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, linestyle='--')
    )
    
    # --- 文字标注 ---
    text_y = (arrow_y_min + arrow_y_max) / 2
    x_min, x_max = plt.xlim() # 获取当前x轴范围
    
    # 动态决定文字放左边还是右边
    if arrow_x > (x_max + x_min) / 2:
        ha_align = 'right'
        text_x = arrow_x - (x_max - x_min) * 0.05 
    else:
        ha_align = 'left'
        text_x = arrow_x + (x_max - x_min) * 0.05

    plt.text(text_x, text_y, 
             f'Signed KS-statistic: {signed_ks:.2f}', 
             fontsize=18, color='black', 
             ha=ha_align, va='center')
    
    # 标签和图例
    plt.xlabel('GSVA score for gene set', fontsize=18)
    plt.ylabel('Proportion', fontsize=18)
    plt.title(f'Separation of GSVA scores in {display_name}', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=2, label='negative'),
        Line2D([0], [0], color='#d62728', lw=2, label='positive')
    ]
    plt.legend(handles=legend_elements, title='O-MIX score', loc='upper left', fontsize=14, title_fontsize=16)
    
    # 保存CDF图
    
    plt.tight_layout()
    plt.savefig(cdf_path, dpi=300)
    plt.close()
    print(f"CDF图已保存至: {cdf_path}")

# ==============================================================================
# 4. 绘制 Panel d (所有疾病的 PCC 累积分布)
# ==============================================================================

def plot_panel_d(df_ai, df_gsva):
    """
    复现 Extended Data Fig. 2d
    修正：自动调整 X 轴范围，确保曲线从 0 开始
    """
    all_pccs = []
    disease_names = df_ai.index.tolist()
    
    for disease in disease_names:
        vec_ai = df_ai.loc[disease].values
        vec_gsva = df_gsva.loc[disease].values
        
        if np.std(vec_gsva) == 0 or np.std(vec_ai) == 0:
            continue 
            
        r, _ = pearsonr(vec_gsva, vec_ai)
        all_pccs.append({'disease': disease, 'pcc': r})
    
    res_df = pd.DataFrame(all_pccs)
    res_df = res_df.sort_values('pcc')
    
    n = len(res_df)
    # Y轴：从 1/N 到 1
    res_df['cumulative_prop'] = np.arange(1, n + 1) / n
    
    # --- 计算 x=0 处的统计量 ---
    # 计算有多少比例的 PCC 是小于等于 0 的 (负相关比例)
    prop_neg = (res_df['pcc'] <= 0).sum() / n
    prop_pos = 1 - prop_neg
    
    # --- 开始绘图 ---
    plt.figure(figsize=(6, 5)) 
    
    # 画主线
    plt.plot(res_df['pcc'], res_df['cumulative_prop'], color='#444444', linewidth=2.5)
    
    # 标记特定疾病 (如果有的话)
    # targets = ['wilms tumor', 'lung cancer', 'leukemia', 'alzheimer disease']

    def format_disease_name(name):
        special_names = {
            'wilms tumor': 'Wilms tumor',
            'lung cancer': 'Lung cancer',
            'alzheimer disease': 'Alzheimer disease',
        }
        return special_names.get(name.lower(), name)


    targets = ['wilms tumor', 'lung cancer', 'alzheimer disease']
    # for target in targets:
    #     match = res_df[res_df['disease'].str.contains(target, case=False)]
    #     if not match.empty:
    #         row = match.iloc[0]
    #         plt.scatter(row['pcc'], row['cumulative_prop'], color='black', marker='x', s=80, zorder=5, linewidth=2)
    #         # 为了防止字重叠，稍微调整位置
    #         plt.text(row['pcc'] - 0.02, row['cumulative_prop'], f'"{target}"', 
    #                  ha='right', va='center', color='#008CBA', fontsize=14, fontweight='bold')
    for target in targets:
        match = res_df[res_df['disease'].str.contains(target, case=False, na=False)]
        if not match.empty:
            row = match.iloc[0]
            display_target = format_disease_name(target)

            plt.scatter(
                row['pcc'], row['cumulative_prop'],
                color='black', marker='x', s=80, zorder=5, linewidth=2
            )

            # 为了防止字重叠，稍微调整位置
            plt.text(
                row['pcc'] - 0.01,
                row['cumulative_prop'],
                f'"{display_target}"',
                ha='right', va='center',
                color='#008CBA', fontsize=14, fontweight='bold'
            )

    # --- 辅助线 (你的需求) ---
    # 1. 垂直线 X=0
    plt.axvline(0, color='gray', linestyle='-', linewidth=0.8)
    
    # 2. 水平线 Y = prop_neg (即曲线与 X=0 的交点)
    plt.axhline(prop_neg, color='#d62728', linestyle='--', linewidth=1)
    
    # 3. 添加比例标注文本
    # 在红线上方写 "Positive"
    plt.text(-0.15, prop_neg + 0.05, f'Positive Correlation: {prop_pos:.1%} (n={n})', 
             color='#d62728', fontsize=14, fontweight='bold')
    
    # 在红线下方写 "Negative"
    # plt.text(0.1, prop_neg - 0.05, f'Negative Correlation: {prop_neg:.1%}', 
    #          color='gray', fontsize=10)

    # 标注总数
    # plt.text(0.55, prop_neg, f'n={n}', fontsize=12, va='bottom')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Correlation coefficient', fontsize=16)
    plt.ylabel('Cumulative proportion', fontsize=16)
    plt.title('PCC for disease gene sets', fontsize=18)
    
    # 修正 X 轴范围
    plt.xlim(-0.15, 0.7) 
    plt.ylim(0, 1.02)
    # plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    plt.savefig(paneld_file, dpi=300)
    
    return res_df # 返回结果供查看

def calculate_significance_count(df_ai, df_gsva):
    """
    统计有多少个疾病满足 "Positive and Significant"
    标准: Signed KS > 0 且 P-value < 0.05
    """
    results = []
    
    # 确保只计算共有的疾病 (即 187 个)
    target_diseases = df_gsva.index.intersection(df_ai.index)
    print(f"开始计算 Signed KS，共 {len(target_diseases)} 个疾病基因集...")

    for disease in target_diseases:
        # 提取当前疾病的向量
        vec_ai = df_ai.loc[disease].values
        vec_gsva = df_gsva.loc[disease].values
        
        # 1. 你的代码中已经包含了 calculate_signed_ks 函数，直接调用
        # 注意：这里 vec_ai 需要是经过 z-score 归一化的，这样 >0 和 <=0 才有意义
        # 你的代码前面已经做了 zscore 处理，这里直接用即可
        ks_stat, p_val = calculate_signed_ks(vec_ai, vec_gsva)
        
        results.append({
            'disease': disease,
            'signed_ks': ks_stat,
            'p_value': p_val
        })
    
    res_df = pd.DataFrame(results)
    
    # --- 核心判断逻辑 ---
    # Positive: Signed KS > 0
    # Significant: P-value < 0.05
    mask_significant_positive = (res_df['signed_ks'] > 0) & (res_df['p_value'] < 0.05)
    
    count_sig_pos = mask_significant_positive.sum()
    total = len(res_df)
    
    print(f"\n=== 统计结果 (对应 Extended Data Fig. 2c) ===")
    print(f"总基因集数量: {total}")
    print(f"显著正相关 (Positive & Significant) 的数量: {count_sig_pos}")
    print(f"比例: {count_sig_pos/total:.2%}")
    
    # 打印前几个显著的结果看看
    print("\n显著正相关的前 5 个疾病示例:")
    print(res_df[mask_significant_positive].sort_values('signed_ks', ascending=False).head(5))
    
    return res_df

# ==============================================================================
# 执行绘图
# ==============================================================================

# 1. 画 Panel b (以 Breast cancer 为例，如果你的列表里有的话)
# 也可以换成 'colorectal cancer'

# 检查是否存在，如果不存在取第一个
if example_disease not in df_gsva.index:
    example_disease = df_gsva.index[0]

print(f"正在绘制 Panel b: {example_disease} ...")
plot_panel_b(example_disease, df_ai, df_gsva)

# 2. 画 Panel d (所有疾病)
print("正在绘制 Panel d ...")
correlation_results = plot_panel_d(df_ai, df_gsva)

# 看看前几名是谁（预测最准的病）
print("\n预测相关性最高的 5 个疾病:")
print(correlation_results.tail(5))




per_disease_pcc = df_gsva.corrwith(df_ai, axis=1)

mean_pcc = per_disease_pcc.mean()
median_pcc = per_disease_pcc.median()

print(f"=== 方法 1: 按疾病平均 (Macro-average) ===")
print(f"平均 PCC (Mean):   {mean_pcc:.4f}")
print(f"中位数 PCC (Median): {median_pcc:.4f}")
print(f"最差的疾病: {per_disease_pcc.idxmin()} ({per_disease_pcc.min():.4f})")
print(f"最好的疾病: {per_disease_pcc.idxmax()} ({per_disease_pcc.max():.4f})")


# ==========================================
# 方法 2: Global PCC (全局相关性)
# 逻辑：把 (187, 14113) 的矩阵拉直成 (2639131,) 的向量，算一次总的 PCC
# ==========================================

# flatten() 将矩阵展平为一维数组
flat_gsva = df_gsva.values.flatten()
flat_ai = df_ai.values.flatten()

global_pcc, _ = pearsonr(flat_gsva, flat_ai)

print(f"\n=== 方法 2: 全局相关性 (Global) ===")
print(f"Global PCC: {global_pcc:.4f}")


sig_results = calculate_significance_count(df_ai, df_gsva)
