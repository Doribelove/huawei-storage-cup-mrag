# Multimodal Evidence Lab — r58

This public source snapshot contains the frozen r58 offline PDF retrieval and visual answering pipeline and its competition-compatible relative entry paths. It is exported from the **r58 technical-declaration revision r2** package, not from an active research working tree. The native backend uses Qwen3-VL-8B CUDA FP16 eager, BGE-M3 retrieval and a local BGE reranker. The historical `*_vllm.py` filenames are interface compatibility names; they do not claim a vLLM backend or Ascend deployment.

## Historical measurement and its limits

On the complete development30 historical AGX released-interface run, the same-model local grader marked **26/30 final answers correct (86.67%)**. Original-source review by the same development author found **25/30 completely supported released answers (83.33%)**, one partially supported answer with missing coverage, and four withheld answers. Raw/source-bound/final local scores were 18/30, 26/30 and 26/30. Batch fatal errors and judge failures were zero; preserved internal Reader contract faults are not claimed to be zero.

These are development results, not independent, held-out or official competition accuracy. They do not establish accuracy above 90%, a competition score, a prize or full deployment qualification. Historical execution included a fresh 100-document build and the original HTTP benchmark client. The 381-test record belongs to the historical freeze and byte-identical runtime proof, not a fresh test run performed during this publication.

The r2 declaration revision only added `requests==2.32.5` and documentation/manifests to the previously frozen package. Its no-model checks used synthetic HTTP fixtures and explicitly substituted existing desktop dependencies for the official client/judge tests. **A genuine GPU build/query smoke of this revised delivery has not been performed.** No model run, software installation, model download or paid judge call is performed for this source publication. See `evidence/publication-validation.json` for the actual publication checks when available.

## Source layout

* `src/evidence_lab/`: all 63 current application/assets files, including source verification, geometry binding, retrieval, Reader, support gates and recovery.
* `scripts/` and `tests/`: all ten frozen helpers and 53 test modules; tests use synthetic fixtures and do not establish benchmark accuracy.
* `rag-anything-experiments/`: release adapter, launcher, shell entries, the original benchmark client, judge and score utility. The build and server paths are `rag_scripts/rag_build_index_vllm.py` and `rag_scripts/rag_uvicorn_server_vllm.py`.
* `raganything/`: all 13 original RAG-Anything library files. This original library requires its own LightRAG/MinerU dependencies; the r58 adapter runs `evidence_lab` and does not import that library.
* `configs/`, `.env.example`, `environment/`: frozen algorithm configuration, editable path examples, package declarations, runtime observations and pinned external model contracts.
* `reference/released-files/`: original organizer code/configuration and its notices, clearly separate from the active adapter.

The current algorithm configuration retains the inherited filename `configs/agx-source-facts-v14.json`. Source and official-interface files copied from r2 are byte-identical. Public export replaces documentation and the real development question example with a synthetic question, and omits runtime/data artifacts. The source hashes and excluded categories are documented in `evidence/provenance-r58.json`.

## Reproduction using separately authorized inputs

Use Python 3.12 with a separately qualified GPU runtime. `environment/qualified-runtime.json` records the historical AGX versions; it is not a universal GPU installation recipe. Do not replace a working Torch/CUDA environment based on this repository. Dependencies and platform-specific clients must already be provisioned for the target platform. The source export does not include the package's Linux aarch64 vendor binaries; normal installed packages supply `python-dotenv`, `requests`, `openai` and other declared client dependencies. The launcher still permits a separately provisioned local `vendor/` directory; this repository contains none.

See [docs/RUNNING.md](docs/RUNNING.md) for build/server/benchmark commands, input pairing, configuration and excluded-input acquisition. Reference answers belong only to evaluator-side judging. Inference accepts **only `question`**; sample IDs, domains, gold answers and precomputed predictions are not inference inputs.

Model weights, PDFs, MinerU data, vectors/indexes, original document inventories, virtual environments, credentials, private answers and the private evaluation set are excluded. Obtain competition inputs from the organizer through authorized access and obtain/use external models only under their applicable upstream terms. Nothing is automatically downloaded. A new build regenerates source provenance from supplied PDFs and parsed lists; the optional old `resources/provenance/source-members.json` archive-fingerprint optimization is intentionally absent.

## Remaining limitations and history

The frozen inventory had 825 collected and 28 failed image features because crop/rotation transforms were not qualified. Failed supplemental hints are skipped; original PDFs and full-document vectors were not deleted. This publication does not repair those transforms or attribute a specific refusal to them. The original library's full import and an independent clean GPU installation remain unverified. The official design template/team/account and formal submission checks are outside this source upload.

The repository stays public with normal `main` history. [docs/HISTORY.md](docs/HISTORY.md) records the preceding r31 snapshot and exact result scope; old commits remain available. Third-party notices are preserved in [THIRD_PARTY.md](THIRD_PARTY.md).
