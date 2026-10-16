# 固定模型资源与复现条件

公开仓库不携带模型权重、已安装环境或平台二进制。原始模型的许可和使用条件由发布者决定：

| 用途 | 发布者与模型 | r58 固定 revision |
|---|---|---|
| Reader | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| 检索 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | `5617a9f61b028005a4858fdac845db406aefb181` |
| 重排 | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

在单独获准的资源准备环境中，可通过发布者仓库按指定 revision 获取原始资源。例如已安装 `huggingface_hub` 时：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    local_dir="models/Qwen3-VL-8B-Instruct",
)
```

这只是单独资源准备指令，本次公开整理没有执行下载。不要把令牌写入脚本或提交到Git。

**BGE-M3 的原始下载不等于 r58 已资格资源。** 本实现要求固定 safetensors 权重及真实 `conversion-manifest.json`；reranker 要求 `source-manifest.json`。精确文件哈希见 [external-models.json](../environment/external-models.json)，已验证参考清单见 [model-provenance-reference](../environment/model-provenance-reference)。当前源码不提供重新生成该历史转换工件的独立转换脚本，因此应取得同一已验证工件，或另行审查并复现转换；不能手造清单、重命名 `.bin` 或放宽哈希校验来冒充兼容。这个资源准备条件尚未在全新机器上独立验证。

将三个模型目录配置到 `MRAG_QWEN`、`MRAG_BGE`、`MRAG_RERANKER`。按 [PLATFORM_DEPENDENCIES.md](../environment/PLATFORM_DEPENDENCIES.md)准备目标平台的原生依赖，并使用 `verify_release_runtime.py` 检查。历史AGX环境记录不代表Windows、x86、vLLM或昇腾已通过资格验证。
