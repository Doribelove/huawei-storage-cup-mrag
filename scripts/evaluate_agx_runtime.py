"""Offline, fail-closed assessment of bounded AGX dependency metadata.

This has no SSH/installation/model action. Its exit status remains 2 for
metadata-only evidence: package versions cannot attest an imported runtime.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def canonical(name):
    return name.lower().replace("_", "-").replace(".", "-")


def version_at_least(value, minimum):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value or "")
    return bool(match and tuple(int(part) for part in match.groups()) >= minimum)


def assess(inventory, plan):
    if inventory.get("schema") != "agx-readonly-dependency-inventory-v1":
        raise ValueError("unsupported inventory schema")
    python = inventory.get("python", {})
    if not all(python.get(name) is True for name in ("isolated", "no_site", "dont_write_bytecode")):
        raise ValueError("isolated no-site no-bytecode collection not confirmed")
    rows = inventory.get("distribution_metadata", [])
    packages = {}
    for row in rows:
        if "name" in row:
            packages.setdefault(canonical(row["name"]), []).append(row)
    torch = packages.get("torch", [])
    transformers = packages.get("transformers", [])
    # No sys.path/import resolution is inferred from metadata directory order.
    torch_cpu_markers = bool(torch) and all(
        row.get("build_markers", {}).get("status") == "static_literal_metadata_only"
        and "cuda" in row["build_markers"].get("fields", {})
        and row["build_markers"]["fields"]["cuda"] is None for row in torch)
    tf_has_builtin = bool(transformers) and all(row.get("qwen3_vl_builtin_source_present") is True for row in transformers)
    tools = inventory.get("tool_versions_and_daemon_runtime_metadata", {})
    runtime = tools.get("docker_runtime_names", {})
    runtime_names = runtime.get("stdout", "").split() if runtime.get("status") == "ok" else None
    files = plan["cpu_proposal"]["files"]
    for item in files:
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("CPU resource missing SHA256")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            raise ValueError("CPU resource missing positive exact byte count")
        if not isinstance(item.get("url"), str) or not item["url"].startswith("https://"):
            raise ValueError("CPU resource is not an HTTPS URL")
    cpu_bytes = sum(item["bytes"] for item in files)
    interpreter = plan["cpu_proposal"]["interpreter"]
    blockers = ["metadata_does_not_attest_import_ABI_CUDA_processor_or_model"]
    if not version_at_least(python.get("version"), (3, 12, 0)):
        blockers.append("observed_interpreter_below_project_Python_3_12")
    if not torch:
        blockers.append("no_torch_metadata_in_observed_standard_locations")
    elif torch_cpu_markers:
        blockers.append("all_observed_torch_build_markers_are_CPU_only")
    elif not all("cuda" in row.get("build_markers", {}).get("fields", {}) for row in torch):
        blockers.append("torch_CUDA_build_marker_unknown")
    if not tf_has_builtin:
        blockers.append("Qwen3_VL_builtin_source_not_confirmed_in_all_observed_Transformers")
    if runtime_names is None:
        blockers.append("Docker_runtime_registration_query_unavailable")
    elif "nvidia" not in runtime_names:
        blockers.append("Docker_does_not_list_named_nvidia_runtime")
    return {
        "schema": "agx-offline-runtime-assessment-v1",
        "observed_python": python["version"],
        "observed_packages": packages,
        "observed_torch_all_CPU_build_markers": torch_cpu_markers,
        "observed_Docker_runtime_names": runtime_names,
        "project_Python_requirement_satisfied_by_observed_interpreter": version_at_least(python.get("version"), (3, 12, 0)),
        "selected_runtime_ready": False,
        "actual_import_ABI_CUDA_or_model_validation_performed": False,
        "readiness_exit_code": 2,
        "blockers": blockers,
        "resource_proposal_not_authorization": {
            "CPU_correctness_gate_wheel_count": len(files),
            "CPU_correctness_gate_wheel_bytes": cpu_bytes,
            "Python_source_bytes": interpreter["bytes"],
            "known_no_weight_CPU_download_total_bytes": cpu_bytes + interpreter["bytes"],
            "GPU_source_build_complete_download_total_bytes": None,
            "container_image_selected": False,
            "new_install_download_or_weight_authorization_granted": False,
            "existing_CPU_packages_are_not_reused_across_CP38_CP312_ABI": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    inventory_bytes = options.inventory.read_bytes()
    plan_bytes = options.plan.read_bytes()
    result = assess(json.loads(inventory_bytes), json.loads(plan_bytes))
    result["input_sha256"] = {"inventory": hashlib.sha256(inventory_bytes).hexdigest(),
                              "plan": hashlib.sha256(plan_bytes).hexdigest()}
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if options.output:
        if options.output.exists():
            raise FileExistsError("refusing to replace an existing assessment")
        options.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")
    return result["readiness_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
