# Multimodal Evidence Lab — v16-r31

An offline PDF retrieval and visual question-answering application with original-source binding, bounded recovery and support checks. This export selects the latest verified stable candidate from AGX rather than a newer diagnostic version.

## Measured version and limits

The complete development30 run finished on 2026-10-11 at 01:48 UTC. All 30 attempts completed with zero batch runtime errors; 22 answers were released and 8 withheld. The same fixed Qwen model local grader scored raw/source-bound/final outputs at **17/30, 23/30 and 22/30**, respectively. A separate source review by the same development author recorded **20/30 fully supported released answers**, with 2 partially supported released answers. Neither review is an independent or official competition accuracy measurement. Two internal Reader output-contract faults were preserved, despite zero batch fatal errors.

All 30 questions have been used for development. There is no independent document-disjoint holdout for this candidate, and the 24/30 target has not been reached. The batch took 3235.89 seconds on the existing AGX native CUDA FP16/eager route. This repository does not establish NPU/vLLM serving performance, formal submission qualification or an award.

The recorded AGX regression passed 357 tests. That is historical evidence; fresh Lenovo export validation is recorded separately in `evidence/local-validation.json` when available. Model inference and the development30 batch were not repeated during export.

## Contents and provenance

`src/` contains the application, `scripts/` the original helpers, `tests/` synthetic regression fixtures, and `configs/` the evaluated configuration and historical protocols. `configs/agx-source-facts-v14.json` is the actual r31 configuration despite its inherited filename. `evidence/` holds aggregate, sanitized records and source-hash provenance.

The received base Git HEAD is historical, not a commit of r31. The AGX candidate had no `.git` metadata. The export verifies its original file hashes against the evaluation freeze and source/test manifest. All application source remains unchanged; one batch helper's personal authorization-file path is replaced by the required `EVIDENCE_LAB_AUTHORIZATION` environment variable. This portability change is listed explicitly in `evidence/provenance.json` and was not used in the historical model evaluation.

Model weights, original documents, competition/private data, indexes, environments, credentials, private answers, per-case judge payloads and historical task conversations are excluded. Tests contain diagnostic fixtures and do not establish competition accuracy. Reference answers are never supplied as runtime inputs. Oracle diagnostics offered by historical helper scripts are outside the reported retrieval development30 result.

## Running

Use Python 3.12 or newer. For a new, isolated CPU development environment, inspect `pyproject.toml` and the declared requirements before installation. Its optional semantic Torch version differs from the evaluated AGX environment. `requirements-agx-observed.txt` records observed versions and is not a deployment lock or an instruction to rebuild an existing GPU environment.

With the required dependencies already installed:

```sh
PYTHONPATH=src python -m evidence_lab.cli --help
PYTHONPATH=src:tests python -m unittest test_unit_reason_consistency
```

See `docs/RUNNING.md` for explicit path inputs and the existing-runtime batch helper. Full inference requires separately authorized local data, indexes, image inventory and pinned model files. They are not included or automatically downloaded by this export.

## Remaining work

Improve the two partial answers and eight withheld cases; establish independent source review and a document-disjoint holdout; test any newer candidate on a complete frozen same-protocol batch before promotion; verify organizer interfaces and formal packaging rules. Newer r36–r40 diagnostics did not supersede r31; r41 was still a diagnostic in progress when selection was checked.

This source snapshot is published in the public repository `huawei-storage-cup-mrag` with `main` as its default branch, under explicit user authorization. It contains sanitized source and aggregate evidence only. Public visibility is not a claim of competition qualification or a license to redistribute the excluded models and data.

## Export validation limitation

Fresh Lenovo verification matched all 134 evaluated source files and 165 project entries in the original source/test manifest. Selected synthetic regressions ran 13 tests with 0 failures and 2 errors. The original, unmodified historical helper `scripts/build_full_index.py` has an `IndentationError` at line 112; it is retained and recorded rather than silently repaired. The evaluated AGX batch used existing indexes. A complete fresh index build, all tests, model inference, and deployment were not reproduced for this export.

Two requested test modules could not load because the existing Lenovo environment lacks `psutil`; 11 actual synthetic test cases passed. The broader full runtime also lacks NanoVectorDB and model dependencies. No dependency was installed and the historical 357-test AGX result is not presented as fresh Lenovo validation.
