# 平台依赖及资源准备

`MRAG_PYTHON` 必须选择目标平台已单独配置、已通过 Torch/CUDA 资格验证的 Python 3.12。启动和工作器都使用该同一解释器及 `-I -B`；不能只依赖交互 shell 的 PYTHONPATH、用户 site-packages 或另一个 Python。

`launch.py` 与 `adapter.py` 通过 `dependency_loader.py` 优先使用解释器已安装的匹配平台包。旧 vendor 只有在 CPython3.12、Linux/aarch64、GNU SOABI=cpython-312-aarch64-linux-gnu 同时匹配时才追加为后备，绝不置于已安装包之前。其他平台完全不加载该 vendor；必须准备本平台依赖，不能复制 aarch64 的 pydantic_core/jiter 二进制到 x86 或 Windows。

在目标合格解释器中先做只读验证：

```sh
"$MRAG_PYTHON" -I -B scripts/verify_release_runtime.py --role client
"$MRAG_PYTHON" -I -B scripts/verify_release_runtime.py --role worker
```

新目标平台应在独立专用环境中按 `client-requirements-lock.txt` 准备客户端包（现在包含 requests==2.32.5），按 `non-gpu-constraints.txt`、`pyproject.toml` 和 `qualified-runtime.json` 核对非 GPU 依赖及语义依赖。GPU Torch/torchvision/CUDA 必须由目标设备的合格环境提供，不能由通用 pip 命令替换。

以下仅为目标平台离线包准备示例，本轮未执行下载或安装；wheelhouse 必须由相同操作系统、架构、Python ABI 的资源准备者提供，并核验其哈希和许可证：

```sh
# 在已获准的目标平台资源准备环境中准备匹配的客户端 wheels：
python -m pip download --only-binary=:all: --dest wheelhouse -r environment/client-requirements-lock.txt
# 在已获准的独立部署环境中安装客户端依赖，不触碰系统 Python 或 GPU 包：
"$MRAG_PYTHON" -m pip install --no-index --find-links=wheelhouse -r environment/client-requirements-lock.txt
```

历史 `client-wheels-lock.json` 是 aarch64 原始文件身份记录，不是跨平台 wheel 锁。新平台实际预检、依赖冲突和 GPU 验收仍需另行完成。本轮只实际验证既有 AGX Linux/aarch64 环境，不声称已验收 x86 GPU、Windows GPU、vLLM 或昇腾。

## 三个模型的离线交付契约

模型权重不包含在官网包或全量索引包。评测组是否已有资源、是否要求另供权重及接收位置均未得到确认，不能宣称资源已就绪。

1. Qwen3-VL-8B-Instruct 固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，四个 safetensors 分片及配置/Tokenizer 必须与 `external-models.json` 的精确文件身份一致。
2. BGE-M3 固定 revision `5617a9f61b028005a4858fdac845db406aefb181`；原始权重 SHA256 为 `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`，本实现合格 safetensors SHA256 为 `002003b14dd71c1a3df0a66cdf89b4a492bd64dcd594d856991a2e45fdf49ff1`。模型根目录必须同时包含对应真实转换产生的 `conversion-manifest.json`。本包保留该已验证清单的逐字节参考副本；不能手造清单、把 .bin 改名或改变哈希验证来绕过身份要求。若接收方只有上游 .bin，需取得这份现成已验证资源，或另行审查并复现转换，本轮没有重做转换。
3. bge-reranker-v2-m3 固定 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，权重 SHA256 为 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`，根目录必须含对应 `source-manifest.json`。精确上游文件身份及已验证参考清单分别见 `external-models.json` 与 `model-provenance-reference/`。

只能从模型发布方的指定 revision 按其许可获取资源，或通过组委会确认的私有交付目标接收提交者已有的已验证资源。将模型目录分别配置到 MRAG_QWEN、MRAG_BGE、MRAG_RERANKER。三个模型均使用本地文件和严格身份检查，不自动联网下载、不执行远程模型代码。
