# EVE 作业结果分析报告

蛋白: P53_HUMAN

## 结果总览

在 157 个带 ClinVar 标签且与评分文件有重叠的突变上，5 组方法的 AUC 排名如下:

| 方法 | AUC | 相对 bayes_z50 的差值 |
|---|---:|---:|
| msa_direct | 0.9818 | +0.0348 |
| bayes_z50 | 0.9470 | 0.0000 |
| nonbayes_z50 | 0.9279 | -0.0191 |
| bayes_z100 | 0.9154 | -0.0316 |
| bayes_z10 | 0.8493 | -0.0977 |

结论先行:
1. 在这组 P53 单点突变任务上，直接 MSA 频率比值打分表现最好。
2. EVE 的最优隐空间维度是 z=50，过小(z=10)和过大(z=100)都下降。
3. 贝叶斯 Decoder 优于非贝叶斯 Decoder。
4. EVE 与 PSSM 的位点保守性呈中等相关，但存在一批显著不一致位点。

---

## 1) 下载并运行 EVE（baseline）

### 结果
1. 端到端流程已跑通: 训练 VAE -> 计算 evol indices -> GMM/EVE 打分 -> 产出图和表。
2. 基线打分文件为 all_EVE_scores_hw_bayes_z50.csv。
3. 100% retained 分类统计:
	- Pathogenic: 5195
	- Benign: 1057
4. uncertainty 列的均值约为 0.2438。

### 分析
1. 分类分布明显偏向 Pathogenic，符合“全单点突变枚举”场景中大多数氨基酸替换会破坏功能的常见现象。
2. baseline 作为后续对照是有效的，且其 AUC(0.9470)已经达到较高水平。

### 对应文件
1. results/EVE_scores/all_EVE_scores_hw_bayes_z50.csv
2. results/evol_indices/P53_HUMAN_300_samples_bayes_z50.csv
3. results/homework/auc_summary.csv

---

## 2) 改动隐空间维度并对比（Bayesian z=10/50/100）

### 结果
1. bayes_z10: AUC=0.8493
2. bayes_z50: AUC=0.9470
3. bayes_z100: AUC=0.9154

### 分析
1. 从 z=10 到 z=50，AUC 提升 +0.0977，说明 z=10 容量偏小，表示能力不足。
2. 从 z=50 到 z=100，AUC 下降 -0.0316，说明更大隐空间并未带来泛化收益，可能引入冗余自由度。
3. 因此在本任务下呈现“中间最优”的趋势，z=50 是较优折中。

### 训练日志侧证
1. bayes_z10 最终 BCE=372.445
2. bayes_z50 最终 BCE=402.038
3. bayes_z100 最终 BCE=408.701

说明: 训练目标中的 ELBO 由多项组成（尤其贝叶斯参数 KL 很大），BCE 的绝对值不直接等价于分类 AUC，最终仍以标签评估(AUC)作为主判断标准。

### 对应文件
1. results/homework/auc_summary.csv
2. logs/P53_HUMAN_hw_bayes_z10_losses.csv
3. logs/P53_HUMAN_hw_bayes_z50_losses.csv
4. logs/P53_HUMAN_hw_bayes_z100_losses.csv

---

## 3) 根据 MSA 计算 PSSM，并与 EVE 位点保守性比较

### 结果
1. Pearson 相关系数: 0.4880
2. Spearman 相关系数: 0.6119
3. 对齐位点数: 329
4. 标准化后平均绝对差: 0.2244
5. 中位绝对差: 0.2182

### 共同点
1. Spearman 高于 Pearson，说明两者在“排序趋势”上较一致。
2. 两种指标都能识别一批高保守区域。

### 差异点
1. EVE 前列位点(示例): 130, 171, 141, 245, 282。
2. PSSM 前列位点(示例): 280, 176, 238, 281, 242。
3. 分歧最大的位点(前几名): 347, 94, 345, 171, 119（abs_diff 约 0.52~0.55）。

### 解释
1. PSSM 主要基于单位点频率/熵，反映局部位点保守性。
2. EVE 的位点分数来自模型化后的突变效应均值，间接包含上下文与高阶依赖信息。
3. 因此“中等相关 + 显著分歧位点”的结果是合理的，分歧位点值得做结构或功能位点复核。

### 对应文件
1. results/homework/conservation_comparison.csv
2. results/homework/top_eve_conserved_positions.csv
3. results/homework/top_pssm_conserved_positions.csv
4. results/homework/top_conservation_disagreement_positions.csv

---

## 4) 改代码直接用 MSA 输入并对比效果

### 实现方式
在 compute_evol_indices.py 中新增 msa_direct 模式，不依赖 VAE checkpoint，直接用加权 MSA 频率计算突变分数:

score = sum over substitutions of log((f_wt + eps)/(f_mut + eps))

其中 f_wt 和 f_mut 为该位点 WT/突变氨基酸在重加权 MSA 中的频率。

### 结果
1. msa_direct AUC=0.9818
2. bayes_z50 AUC=0.9470
3. 提升 +0.0348

### 分析
1. 在 P53 单突变数据上，标签与“位点/替换保守性”的一致性较强，直接频率比值可得到很强判别能力。
2. msa_direct 不涉及采样近似和深层生成建模，方差更低、流程更直接。
3. 但该方法对“多突变耦合效应”建模能力有限，泛化到组合突变或跨蛋白任务时不一定继续领先。

### 对应文件
1. compute_evol_indices.py
2. results/evol_indices/P53_HUMAN_1_samples_msa_direct.csv
3. results/homework/auc_summary.csv

---

## 5) 贝叶斯 Decoder vs 非贝叶斯 Decoder

### 实验设置
1. 固定 z_dim=50，仅切换 decoder_parameters.bayesian_decoder。
2. 其余主要超参数保持一致（hidden layers、dropout、卷积输出、温度缩放等）。

### 结果
1. Bayesian z50: AUC=0.9470
2. Non-Bayesian z50: AUC=0.9279
3. 贝叶斯版本提升 +0.0191

### 分析
1. 贝叶斯 Decoder 对解码器参数分布进行采样与正则，提升了鲁棒性与不确定性建模能力。
2. 日志中 non-Bayesian 的 KLD_decoder_params_norm 为 0（无解码器参数 KL 项），训练更快，但泛化表现略弱。
3. 对当前任务，Bayesian Decoder 是更稳健选择。

### 对应文件
1. EVE/VAE_decoder.py
2. results/homework/params/bayes_z50.json
3. results/homework/params/nonbayes_z50.json
4. logs/P53_HUMAN_hw_bayes_z50_losses.csv
5. logs/P53_HUMAN_hw_nonbayes_z50_losses.csv

---

## 最终结论

1. 作业 5 项任务已全部完成并形成可复现实验链路。
2. 若目标是当前 P53 单突变标签判别，最佳结果来自 msa_direct(AUC=0.9818)。
3. 若目标是保持 EVE 的生成式框架与可拓展性，推荐 bayes_z50 作为默认配置。
4. EVE 与 PSSM 的一致与分歧并存，建议优先关注分歧最大的位点做进一步生物学解释。
