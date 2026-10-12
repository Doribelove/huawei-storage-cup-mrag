# Running the frozen r58 source

These commands document the interface; publication does not execute model inference. Run compute only when the target host's sole execution owner has assigned a serial slot under the existing host policy. Do not stop unrelated tasks, bypass guards or assume that an absent lock is a handoff.

## Required inputs and existing environment

Obtain the official PDFs and corresponding MinerU `*_content_list.json` files from the organizer through authorized access. Pair files by document stem: a PDF may end in `_origin.pdf`; the parsed list uses the same stem plus `_content_list.json`. `DOC_FILE_PATH` can contain nested `auto/` directories. Do not put reference answers, precomputed predictions or evaluator annotations into parsed source records. All source evidence and source binding are regenerated from the actual files during the build.

Use separately provisioned, fixed local model directories: Qwen/Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, BAAI/bge-m3 revision `5617a9f61b028005a4858fdac845db406aefb181`, and BAAI/bge-reranker-v2-m3 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Read `environment/external-models.json` and `src/evidence_lab/semantic.py` for required file/provenance pins, including the BGE conversion manifest. Ordinary raw upstream files are not claimed to be interchangeable with the qualified converted BGE runtime. Reuse already verified local artifacts where available; this source does not supply or fetch them.

The historical AGX runtime used Python3.12.14, Torch2.3.1, torchvision0.18.1, Transformers4.57.6 and CUDA11.4. Preserve its existing dynamic-library environment and qualified interpreter; a system Python executable alone is not sufficient proof of compatibility. `environment/qualified-runtime.json`, `runtime-packages-historical.txt`, `non-gpu-constraints.txt`, `client-requirements-lock.txt` and `release-interface-requirements.txt` document dependencies. No dependencies or binaries are installed by these source-export checks. The client wheel manifest describes historical Linux aarch64 artifacts; it is not portable to Windows. Provision native target-platform packages separately, and do not change GPU components based on client requirements.

## Official relative paths

From the repository root, copy `.env.example` to a private `.env`, then edit the following paths for your authorized inputs:

```dotenv
MRAG_QWEN=/path/to/verified-Qwen3-VL-8B
MRAG_BGE=/path/to/verified-bge-m3
MRAG_RERANKER=/path/to/verified-bge-reranker-v2-m3
DOCS_PATH=/path/to/authorized-pdfs
DOC_FILE_PATH=/path/to/paired-mineru-lists
RAG_STORAGE=/path/to/a-new-output-index
RAG_IP=127.0.0.1
RAG_PORT=8080
RAG_BINDING_HOST=http://127.0.0.1:8080/query
BENCH_QUERIES=/path/to/authorized-question-only.json
BENCHMARK_NAME=r58_local
```

Select the qualified interpreter in the shell and perform a **fresh** build into a nonexistent storage directory; the code refuses to overwrite an old build:

```sh
export MRAG_PYTHON=/path/to/qualified-environment/bin/python
DOTENV_FILE="$PWD/.env" bash rag-anything-experiments/run.sh run_rag_build
DOTENV_FILE="$PWD/.env" bash rag-anything-experiments/run.sh run_rag_server
```

The server uses serial local processing. From a separate client, send only a question:

```sh
curl --fail --header 'Content-Type: application/json'   --data '{"question":"What does the supplied document say about its main subject?"}'   http://127.0.0.1:8080/query
```

The response has an `answer` string and `query_time_s`; an empty answer is a permitted refusal, not proof of correctness. Full historical index reuse requires the excluded original build storage and verified source inputs, provided separately; it is not included in GitHub.

For the unchanged official benchmark client:

```sh
DOTENV_FILE="$PWD/.env" bash rag-anything-experiments/run.sh run_rag_unidocbench
```

That client retains evaluator metadata locally but sends only `sample["question"]` to inference. `examples/question-only.json` is a synthetic one-question JSON array, not the private development30 set. Judge execution requires actual evaluator-supplied labels and a separately authorized judge service. Set judge host/key and true baseline times privately; the repository supplies neither. No official score or paid call is implied by the commands.

## Static and synthetic checks

With the necessary non-model dependencies already installed:

```sh
PYTHONPATH=src:tests python -B -m unittest test_released_interface
```

This six-test fixture suite checks source pairing, tamper rejection, label quarantine, no-overwrite behavior and question-only HTTP/refusal contracts. It does not load Qwen/BGE or establish model accuracy. `scripts/run_tests.py` discovers the entire suite, which needs additional qualified dependencies; do not describe the selected six tests as a full-suite rerun.

The original `raganything` library is separately available in this repository but needs `environment/requirements-raganything-original.txt`, including LightRAG and MinerU. Missing those dependencies does not show that the active r58 adapter is broken.
