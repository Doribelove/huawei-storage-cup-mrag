# r58 公开设计说明

本文根据用户确认的 r58 官网提交包源码与已有验收记录整理，是公开阅读版设计说明。原始 Word/PDF 设计材料中的签名、账号资料和受限第三方图片不复制到仓库。

## 数据如何进入系统

PDF 与对应 MinerU `*_content_list.json` 以文档 stem 配对，原文哈希、页码、区域及来源成员共同构成证据身份。构建重新生成来源记录，不把参考答案、预计算预测或评测页码混入证据。输入校验会拒绝越界路径、未知或改变的来源绑定等情况。

`prepare_extracted_corpus.py` 处理来源记录，`build_full_index.py` 构建 BGE-M3 索引，`collect_native_image_features.py` 和 `configure_inventory.py` 整理图像资格及运行配置。赛事入口在 `rag-anything-experiments/adapter.py` 组织这些过程，拒绝覆盖既有构建目录。

## 从问题到候选证据

`retrieval_backend.py` 提供 BM25、dense、RRF 和 rerank 路由；发布配置结合 BGE-M3 与 BGE reranker。`agent.py` 管理有界检索与邻域补充，来源校验贯穿检索、融合、重排和后续输入。源码限定候选池与调用预算，不能把检索命中本身等同于回答正确。

## Reader 与支持检查

`reader.py` 使用固定本地 Qwen3-VL-8B-Instruct Transformers 后端，保留原文本和绑定图像。后续支持、事实、参数、图表与来源规则决定候选能否发布。空回答是允许的拒答。引用图片不意味着模型一定进行了视觉观察；真实冒烟响应明确记录 `no model observation was requested`，公开演示保留这一区别。

推理 HTTP 请求只有 `question`。历史样例 ID 和评测参考答案属于客户端/审计侧，不进入推理请求。

## 部署边界

本次确认的是本地 Transformers 版本，不使用后续 HTTP 模型 API 适配候选。历史 `*_vllm.py` 文件名仅保持赛事相对路径兼容。

依赖加载器优先使用目标解释器已安装且匹配平台的依赖；旧 vendor 仅在原 Linux aarch64 CPython 3.12 ABI 条件满足时作为末位后备。公开源码不携带已安装 vendor 树。`verify_release_runtime.py` 分 client/worker 核对依赖和设备条件。GPU Torch/CUDA 由目标平台单独准备。

## 已知限制

历史发布接口开发30题本地判分26/30，同作者原文完整支持25/30，不能当成官方或独立准确率。原生图像853条中825成功、28条几何资格失败；1条无效解析区域保留。历史 Reader 契约错误不宣称为零；失败的补充提示跳过不代表原文或整篇向量被删除。公开整理不修复、重评或改写这些旧结果。

真实单文档冒烟验证了指定环境中的新建库和 HTTP 查询；没有证明所有目标平台可用。原始 RAG-Anything 库需要独立依赖，完整原库导入及干净 GPU 安装仍需接收方验证。BGE 固定转换资源的再现条件见 [MODELS.md](MODELS.md)。
