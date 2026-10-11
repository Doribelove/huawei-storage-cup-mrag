# Existing-runtime execution

The application supplies BM25/dense fusion, BGE reranking, original PDF rendering and source-coordinate binding, a pinned Qwen visual Reader, conservative support gates and bounded source recovery. Reading source identity or receiving a true model support verdict alone does not prove a correct answer.

The existing AGX evaluation used Python 3.12.14, Torch 2.3.1, torchvision 0.18.1 and Transformers 4.57.6. Its GPU dynamic-library environment is external to this source export. Reuse an authorized, qualified runtime rather than treating the desktop optional-dependency declaration as a GPU deployment recipe. No models or dependencies were downloaded for this export.

For a separately authorized future run, the original batch helper takes explicit file paths:

```sh
export EVIDENCE_LAB_AUTHORIZATION="/path/to/existing-user-approved-authorization.json"
PYTHONPATH=src python scripts/run_agx_batch.py   --data-root /path/to/authorized-data-export   --model-dir /path/to/existing-pinned-qwen-model   --bge-dir /path/to/existing-verified-bge-m3   --reranker-dir /path/to/existing-verified-reranker   --questions /path/to/question-only-input.json   --config configs/agx-source-facts-v14.json   --native-image-inventory /path/to/verified-original-image-search-inventory.json   --output /path/to/local-results
```

These paths document the interface. The runtime needs its existing source-map/index formats, verified originals and native inventory. The command has not been reproduced on Lenovo and must not be launched alongside the AGX owner's active computation. It does not include an authorization grant. Historical supervision/build helpers remain for reproducibility and should not be used to alter host environments without separate approval.

The exact Qwen revision is `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; the evaluated retrieval route is `rrf_rerank`. Model and original-source manifest hashes are retained in provenance; hashes alone do not include the excluded inputs or make the evaluation independently reproducible.
