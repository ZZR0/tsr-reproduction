# 单份数据 TSR 最佳组：M2_ge_filter

本目录归档 2026-11-04 单份数据训练链中，四组已完成配置里 AlpacaEval LC 最高的一组：**future 用 `>=` 替换、去掉 chosen/rejected 同分偏好对**。结果是 **LC 25.3549、Win 28.8063、805 题**。2026-11-07 从共享盘导出。

这些是本轮实际保存的实验数据；仓库根目录原有报告中的历史实验保持原样。这里的 Mf0、M1、Mf1 和 M2 均使用单份 DPO 数据，不是修复前重复加载两份数据的训练链。

## 四组结果

| 模型 | future 替换条件 | chosen/rejected 同分 | LC (%) | Win (%) | 平均字符数（leaderboard） |
|---|---|---|---:|---:|---:|
| **M2_ge_filter** | **≥** | **删除** | **25.3549** | **28.8063** | **2320** |
| M2 | ≥ | 保留 | 23.5838 | 26.7984 | 2332 |
| M2_gt_filter | > | 删除 | 22.8726 | 22.9997 | 1979 |
| M2_gt_keep | > | 保留 | 22.8132 | 22.8340 | 1975 |

各组均评测 805 题。完整精度见 [四组比较](provenance/comparison-four-variants.json)。本目录保存最佳组与其前置模型的原始数据；其他三组只保存比较结果。

## 最常用的文件

- [最佳组 AlpacaEval 805 条原始回答](alpaca_eval/M2_ge_filter/model_outputs.json)
- [最佳组逐题评分](alpaca_eval/M2_ge_filter/annotations.json.gz)
- [最佳组 leaderboard](alpaca_eval/M2_ge_filter/leaderboard.csv)
- [冻结的 805 条 reference 回答](alpaca_eval/reference_outputs.json)（从已存评分中的 reference 字段提取，顺序与模型回答一致）
- [实际 M2 训练配置](configs/M2_ge_filter.yaml)
- [实际 D2 训练数据](data/four-settings/M2_ge_filter/D2/train.json.gz)
- [每个原始文件的来源、SHA-256 和存储路径](manifest.json)
- [文字与表格版规则说明 PDF](docs/TSR-rules-and-DPO-config.pdf)

## 数据范围

| 目录 | 内容 |
|---|---|
| `inputs/sft_data/` | 实际使用的 IFT、EFT 训练文件及 SFT 测试集 |
| `inputs/SelfPlayBeginData/` | qa1、qa2 各 5,000 条原始问题 |
| `data/iter0/` | M0 和 Mf0 的候选回答、M0 的自评分、D1/D2 偏好对及统计 |
| `data/iter1/` | M1、M0、Mf1 的候选回答，M1 对三者的自评分，训练 Mf1 的 D1 |
| `data/four-settings/M2_ge_filter/D2/` | 最佳组实际训练的 4,770 对偏好数据及统计 |
| `configs/` | M0、Mf0、M1、Mf1、M2_ge_filter 的实际训练 YAML |
| `alpaca_eval/` | 前置模型与最佳组的回答、生成诊断、评分缓存、逐题评分、annotator 配置和 leaderboard；另含 SFT 的 seed 1/2 |
| `model_metadata/` | 仍可读取的前置模型配置、tokenizer、trainer_state、训练结果及 TensorBoard 记录；不含权重分片 |
| `logs/` | 本轮生成、评分、训练和评测完整日志；复用 M0 的原始日志单独存放 |
| `provenance/` | 实际加载条数、训练完成记录、源代码修复记录、原始输入哈希、Mf1 来源审计、四组结果 |
| `source_snapshot/` | 导出时实际使用的复现脚本和上游关键源码、补丁、依赖列表与许可证 |
| `scripts/original_run/` | 实际运行时的调度、构对检查和数据加载检查脚本；保留原服务器路径作为运行记录 |

大于 1 MiB 的文件使用 gzip 压缩，但 `model_outputs.json` 和 `reference_outputs.json` 保持未压缩，方便直接下载评分。完全相同的文件只保存一份；`manifest.json` 的 `identical_to` 记录全部别名和原始路径，包括训练输入快照与 official 校验副本。去重只改变归档存储，不改变文件内容或数据行数。

**权重缺口：** M2_ge_filter 的权重与模型目录元数据原先存放在旧容器 `/opt/tsr-reproduction-aa7200a/tsr-singlecopy-20261104/models/M2_ge_filter`。新 CPU 服务器上该目标不存在，因此本归档无法包含最终 checkpoint 或原始 M2 `trainer_state.json`。共享盘保留的完整 DPO 日志、训练完成记录、输入哈希以及 11 月 5 日的来源审计均已收录。前置 M0/Mf0/M1/Mf1 的权重仍在共享盘，本 Git 数据归档不包含这些权重分片。

## 核验与解压

在本目录执行，Python 3.11 标准库即可：

```bash
python3 verify_and_unpack.py
python3 verify_and_unpack.py --output /absolute/path/to/tsr-best-data
```

第一条命令逐文件验证压缩前后的 SHA-256；第二条将唯一文件解压到空目录。需要复原所有相同副本的目录结构时，增加 `--restore-duplicates`。这会同时恢复历史 official 校验副本；训练仍应使用修复后的 `src/make_cfg.py` 隔离加载唯一的根目录 `train.json`。

YAML 和原始日志保留当时的绝对路径。重跑时应将模型、数据和输出路径映射到新机器；不要直接运行保留旧 GPU 分配及旧路径的调度脚本。`source_snapshot/reproduction/src/build_pairs.py` 可直接从已解压的评分文件重新生成偏好对；最佳组的参数为 `future` 模式和 `--filter`，不使用 `--strict`。归档前已重新构建两轮 D1/D2 并核对字节级 SHA-256，核验结果见 `provenance/archive-validation.json`。

## 本轮实际规则与超参数

| 项目 | 设置 |
|---|---|
| SFT | Llama-3.1-8B；IFT 5,000 + EFT 1,871；3 epochs；lr 2e-6；不 mask system/user；长度 2048；seed 42 |
| iter0 | M0 生成当前回答并统一评分；Mf0 从 M0 在 D1 上训练；M1 从 M0 在 D2 上训练 |
| iter1 | M1/M0/Mf1 各生成每题 7 个回答；全部由 M1 评分；Mf1 从 M1 在 D1 上训练 |
| 最佳组 chosen | 比较 M1 与 Mf1 的最高分；Mf1 分数 ≥ M1 时使用 Mf1；模型内部同分取生成顺序中第一条 |
| 最佳组 rejected | 比较 M1 与 M0 的最低分；同分优先 M1；Mf1 不参与 rejected 候选池 |
| 最佳组过滤 | 仅保留 chosen 分数严格大于 rejected 的样本；5,000 → 4,770，删除 230 对 |
| 最佳组初始化 | M2_ge_filter 与冻结 reference 均取 M1，不从 Mf1 初始化 |
| DPO 数据/步数 | Mf0、M1、Mf1 各 5,000 对、156 步；M2_ge_filter 4,770 对、149 步；各 1 epoch、数据单份加载 |
| DPO 优化 | sigmoid DPO；beta 0.1；AdamW；lr 5e-7；cosine；warmup 0.1；seed 42 |
| DPO 并行 | 每阶段 2 张 A800 80 GB；每卡 1 对；梯度累积 16；global batch 32；BF16；ZeRO-3 |
| DPO 长度 | prompt + 单个回答上限 2048；prompt 配置上限 512，保留末尾；补 BOS 后实际可达 513；只累计回答 token 的 log-prob |
| 训练候选生成 | vLLM，T=1.0，top_p=1.0，max_tokens=1024，K=7 |
| AlpacaEval 生成 | vLLM，T=1.0，top_p=1.0，max_tokens=1024，seed=0；另外保存 SFT seed 1/2 |
| AlpacaEval 评分 | gpt-4o-2024-08-06；reference=gpt4_1106_preview；官方 clf 模板与 logprob 加权；805 题 |
| 关键依赖 | torch 2.5.1、transformers 4.46.3、TRL 0.12.2、DeepSpeed 0.15.4、accelerate 1.1.1、vLLM 0.6.4.post1、alpaca_eval 0.6.6 |

更多行为细节见 PDF。11 月 5 日审计文件包含当日其他实验的状态快照；当前四组完成结果以 `comparison-four-variants.json` 为准。

## 来源与使用条件

运行基于复现代码 `aa7200a`，DPO 重复加载修复对应本仓库 `1f80f17`；上游 TSR 为 `020266a`。来源文件路径、内容哈希和去重关系全部记录于 manifest。配置中的密码、API key、token、私钥和环境文件不在归档范围内，导出时也检查了已配置凭据是否出现在所选文件中。

代码许可证见仓库及源码快照中的 LICENSE。训练数据保留上游来源和适用条款；模型生成内容受 Llama 3.1 Community License 约束。本归档不改变原始数据或模型输出的使用条件。
