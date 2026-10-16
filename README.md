# 不二鸟 · 华为存储杯多模态检索问答

从 PDF 的文本、表格和图像中检索证据，给出可追溯回答；证据不足时保留拒答。这里公开的是用户确认已提交的 **`最终发布准备_20261015_r58`** 源码，使用 **本地 Transformers + Qwen3-VL-8B-Instruct、BGE-M3、BGE reranker**。

![r58 系统流程：来源导入、检索重排、Reader 和支持检查](docs/assets/architecture-r58.png)

本次源码来自 r58 官网正式 ZIP，与配套全量包的 **1,909 个共同非清单文件逐字节一致**。两个包的 CRC、官网 1,909 项及全量 3,929 项清单哈希均已重新核对。没有混入后续 API 适配候选。原有 Git 历史和实验记录保留，[来源与公开范围](docs/SOURCE_AND_SCOPE.md)列出包身份和排除项。

## 看一次真实输入与输出

下面来自 2026-10-15 的 r58 真实单文档新建库与 HTTP 查询记录。图片是原始 JSON 的阅读视图，**不是运行界面截图**。

![r58 真实 HTTP 问题、回答和耗时的日志阅读图](docs/assets/r58-http-log.png)

```json
{"question":"According to the BDC quarterly economic outlook, how did the average exchange rate change between the first quarter of 2012 and the fourth quarter of 2013?"}
```

真实返回的 `answer`：

> The average exchange rate for BDC increased from 161.6 in Q1-12 to 167.9 in Q4-13.

![根据原文数值重新绘制的汇率证据摘要，并非原始截图](docs/assets/finance-evidence-summary.png)

本次编辑核对定位到 **PDF 物理第 16 页、Figure 11、BDC 行**。原始响应没有页码，不能把本次补充定位写成系统当时已返回页码。原始问题的文档称谓保留不改；实际文档题名为 *FY-2013 Economic Report and 2014 Outlook*，发布者 Watershed Capital Limited。受限原图没有再分发，上图为自绘数据摘要。原始问题、回答、图片编号和记录哈希见[精选真实记录](evidence/demos/r58-http-smoke.json)。

## 更多真实案例

以下两例来自历史 r58 开发集冻结记录，本次没有重新推理。审计编号仅用于定位历史记录，不发送给模型。

| 场景 | 实际问题 | 实际最终输出 | 来源定位 |
|---|---|---|---|
| 能源正文取值 | How much energy from biomass is anticipated to be supplied to the UK's total primary energy under the DECC scenario referred to as SBF, when measured in terawatt-hours? | `1062` | `energy_2357218`，物理第 13 页 |
| 图像问题拒答 | Which machinery is usually employed to clear snow on River Street during December 2008 as depicted? | 空字符串 `""` | 历史审阅上下文为 `construction_1676343` 第 12 页；没有成功回答引用 |

能源案例对应原文片段：

![原始论文第13页的两行正文裁片，含1062 TWh](docs/assets/energy-source-p13-excerpt.png)

来源：Patrick E. Allen、Geoffrey P. Hammond，*Bioenergy utilization for a low carbon future in the UK: the evaluation of some alternative scenarios and projections*，BMC Energy 1, 3 (2019)。[论文与 DOI](https://doi.org/10.1186/s42500-019-0002-9)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。上图由实际 PDF 物理第 13 页裁切、放大渲染，未改文字和数值。[图像归属记录](evidence/demos/energy-image-attribution.json)与[案例原值及哈希](evidence/demos/r58-selected-cases.json)可供核对。拒答示例展示真实限制，不补造正确答案或原因。

## 结果与验证范围

| 记录 | 实际结果 | 解释 |
|---|---|---|
| 历史 r58 发布接口开发 30 题 | 本地同模型判分 **26/30（86.67%）** | 开发记录；不是独立、留出或官方成绩 |
| 同作者原文完整支持审阅 | **25/30（83.33%）** | 另有 1 题覆盖不完整、4 题拒答 |
| 2026-10-15 单文档真实冒烟 | 建库 **30.428353183 s**，查询 **107.3451 s** | 真实新建库和 question-only HTTP；不外推全量性能 |
| 本次公开整理 | 见[本轮验证记录](evidence/publication-r58-20261016-validation.json) | 软件检查与来源核对；没有新增模型推理、建库或比赛评测 |

历史 381 项测试、本次提交包的 385 项测试和公开整理期间的测试记录分别保留。文件名 `*_vllm.py` 是赛事入口兼容名，**不能据此声称本实现使用 vLLM 或已验证昇腾部署**。历史图像资格失败、内部 Reader 契约错误等限制见[设计与局限](docs/DESIGN.md)和[历史记录](docs/HISTORY.md)。

## 运行

准备有权使用的成对 PDF / MinerU 解析文件、固定模型资源和匹配的 Python 3.12 GPU 环境。然后从仓库根目录运行：

```sh
cp .env.example .env
# 编辑 .env，填写本机模型和输入路径；不要把真实凭据提交到 Git。
export MRAG_PYTHON=/path/to/qualified-environment/bin/python
"$MRAG_PYTHON" -I -B scripts/verify_release_runtime.py --role client
"$MRAG_PYTHON" -I -B scripts/verify_release_runtime.py --role worker
DOTENV_FILE="$PWD/.env" bash rag-anything-experiments/run.sh run_rag_build
DOTENV_FILE="$PWD/.env" bash rag-anything-experiments/run.sh run_rag_server
```

构建目录必须是新的不存在目录，真实构建完成后再启动查询。模型、完整环境和索引不在 Git 中；模型身份、平台依赖、数据获取与复现限制见[运行说明](docs/RUNNING.md)和[模型资源](docs/MODELS.md)。

```sh
curl --fail --header 'Content-Type: application/json' \
  --data '{"question":"What does the supplied document say about its main subject?"}' \
  http://127.0.0.1:8080/query
```

上面的通用问题只是接口示例，不是新的测试结果。推理仅接收 `question`；标准答案与评分信息留在评测侧。

## 源码导航

| 目录 | 内容 |
|---|---|
| [src/evidence_lab](src/evidence_lab) | 完整应用核心：检索、证据绑定、Reader、支持检查与恢复 |
| [rag-anything-experiments](rag-anything-experiments) | r58 构建、HTTP 服务、原 benchmark / judge / score 接口与依赖加载器 |
| [scripts](scripts)、[tests](tests) | 构建工具、运行预检和合成软件测试 |
| [configs](configs)、[environment](environment) | 冻结算法配置、依赖和模型身份契约 |
| [docs/DESIGN.md](docs/DESIGN.md) | 面向公开阅读的设计说明，去除签名和受限截图 |
| [raganything](raganything)、[reference](reference) | 原始库与官方参考入口，保留第三方许可；不是 r58 实际推理后端 |

第三方归属见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库公开完整必要源码与精选脱敏演示，不分发真实凭据、个人签名、完整非公开数据集、权重、运行环境、超大索引、全量预测、参考答案或 judge 日志。
