"""Bounded Jetson dependency metadata inventory; Python 3.8 stdlib only.

Run with python3 -I -B -S. This never imports application/ML packages, invokes
package managers to install, starts containers, or writes a report on the host.
Distribution metadata is not an import, ABI, CUDA, or model readiness check.
"""
import ast
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

PACKAGES = (
    "torch", "torchvision", "transformers", "numpy", "Pillow", "tokenizers",
    "safetensors", "huggingface-hub", "accelerate", "sentence-transformers",
    "nanovectordb", "PyMuPDF", "packaging", "pip", "setuptools", "wheel",
    "cmake", "ninja", "astunparse", "six",
)
DEBIAN_PACKAGES = (
    "nvidia-l4t-core", "nvidia-jetpack", "libcudnn8", "libcudnn8-dev",
    "nvidia-container-runtime", "nvidia-container-toolkit", "docker.io",
    "docker-ce", "docker-ce-cli", "containerd.io", "gcc", "g++", "make",
    "build-essential", "libssl-dev", "zlib1g-dev", "libbz2-dev",
    "libreadline-dev", "libsqlite3-dev", "libffi-dev", "liblzma-dev",
    "libopenblas-dev", "libblas-dev", "python3-dev", "python3.12-dev",
    "cuda-toolkit-11-4", "cuda-compiler-11-4", "cuda-cudart-dev-11-4",
)


def normalize(name):
    return name.lower().replace("_", "-").replace(".", "-")


def sanitize(value):
    return str(value).replace(str(Path.home()), "<USER_HOME>")


def torch_build_markers(dist):
    """Read small literal fields from version.py; do not execute/import it."""
    path = Path(dist.locate_file("torch/version.py"))
    if not path.is_file() or path.stat().st_size > 16384:
        return {"status": "bounded_version_file_unavailable"}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fields = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in ("__version__", "cuda", "hip", "debug", "git_version"):
                    try:
                        fields[target.id] = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        fields[target.id] = "not_a_literal"
        return {"status": "static_literal_metadata_only", "fields": fields}
    except (OSError, UnicodeError, SyntaxError) as error:
        return {"status": "metadata_error", "error_type": type(error).__name__}


def command(args, timeout=8):
    executable = shutil.which(args[0])
    if not executable:
        return {"status": "executable_not_on_PATH"}
    try:
        result = subprocess.run(
            [executable] + list(args[1:]), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout}
    except OSError as error:
        return {"status": "os_error", "error_type": type(error).__name__}
    stdout = result.stdout.decode("utf-8", errors="replace")
    # Raw command errors may contain host/user identifiers; never include them.
    status = "ok" if result.returncode == 0 else "nonzero_exit"
    return {"status": status, "exit_code": result.returncode,
            "stdout": sanitize(stdout[:12000]), "stdout_truncated": len(stdout) > 12000,
            "stderr_bytes": len(result.stderr)}


def inventory():
    version = "%d.%d" % sys.version_info[:2]
    locations = [sysconfig.get_path("purelib"), sysconfig.get_path("platlib"),
                 "/usr/lib/python3/dist-packages",
                 "/usr/local/lib/python" + version + "/dist-packages",
                 str(Path.home() / ".local" / "lib" / ("python" + version) / "site-packages")]
    locations = list(dict.fromkeys(p for p in locations if p and Path(p).is_dir()))
    wanted = {normalize(name) for name in PACKAGES}
    distributions = []
    for location in locations:
        try:
            for dist in importlib.metadata.distributions(path=[location]):
                name = dist.metadata.get("Name", "")
                if normalize(name) in wanted:
                    row = {"name": name, "version": dist.version,
                           "requires_python": dist.metadata.get("Requires-Python"),
                           "metadata_location": sanitize(location)}
                    if normalize(name) == "torch":
                        row["build_markers"] = torch_build_markers(dist)
                    if normalize(name) == "transformers":
                        row["qwen3_vl_builtin_source_present"] = Path(dist.locate_file(
                            "transformers/models/qwen3_vl/modeling_qwen3_vl.py")).is_file()
                    distributions.append(row)
        except (OSError, ValueError) as error:
            distributions.append({"metadata_location": sanitize(location),
                                  "status": "metadata_error", "error_type": type(error).__name__})
    python_candidates = {}
    for name in ("python3.8", "python3.9", "python3.10", "python3.11", "python3.12", "python3.13", "python3.14"):
        python_candidates[name] = command([name, "-I", "-B", "-S", "-c",
                                          "import sys; print(sys.version.split()[0])"])
    dpkg = command(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n"] + list(DEBIAN_PACKAGES))
    # dpkg exits 1 if any requested package is absent; retain installed rows,
    # without inferring that every absent query has never been installed.
    dpkg_rows = [line.split("\t") for line in dpkg.pop("stdout", "").splitlines() if line.count("\t") == 2]
    dpkg["package_rows"] = dpkg_rows
    commands = {
        "gcc_version": ["gcc", "-dumpfullversion"],
        "gxx_version": ["g++", "-dumpfullversion"],
        "make_version": ["make", "--version"],
        "nvcc_fixed_existing_toolkit_version": ["/usr/local/cuda-11.4/bin/nvcc", "--version"],
        "docker_client_version": ["docker", "--version"],
        "docker_server_version": ["docker", "version", "--format", "{{.Server.Version}}"],
        "docker_runtime_names": ["docker", "info", "--format", "{{range $key, $value := .Runtimes}}{{$key}} {{end}}"],
        "nvidia_container_runtime_version": ["nvidia-container-runtime", "--version"],
    }
    return {
        "schema": "agx-readonly-dependency-inventory-v1",
        "python": {"version": sys.version.split()[0], "machine": platform.machine(),
                   "isolated": bool(sys.flags.isolated), "no_site": bool(sys.flags.no_site),
                   "dont_write_bytecode": bool(sys.dont_write_bytecode)},
        "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
        "distribution_metadata": sorted(distributions, key=lambda row: (row.get("name", ""), row["metadata_location"])),
        "metadata_search_locations": [sanitize(p) for p in locations],
        "python_candidates_on_PATH": python_candidates,
        "debian_package_metadata": dpkg,
        "tool_versions_and_daemon_runtime_metadata": {name: command(args) for name, args in commands.items()},
        "limits": {"package_imports_performed": False, "CUDA_tensor_or_model_calls_performed": False,
                   "containers_started_or_images_inspected": False,
                   "installer_or_host_configuration_actions_performed": False,
                   "PATH_and_standard_site_metadata_only_no_venv_or_home_scan": True,
                   "missing_on_PATH_is_not_proof_absent_from_entire_host": True},
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), ensure_ascii=True, sort_keys=True, indent=2))
