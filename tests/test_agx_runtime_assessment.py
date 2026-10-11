"""Regressions for misleading deployment readiness from plausible metadata."""
import copy
import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agx_runtime_assessment", ROOT / "scripts/evaluate_agx_runtime.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inventory_spec = importlib.util.spec_from_file_location("agx_runtime_inventory", ROOT / "scripts/agx_runtime_inventory.py")
inventory_module = importlib.util.module_from_spec(inventory_spec)
inventory_spec.loader.exec_module(inventory_module)


class RuntimeAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            "schema": "agx-readonly-dependency-inventory-v1",
            "python": {"version": "3.8.10", "isolated": True, "no_site": True, "dont_write_bytecode": True},
            "distribution_metadata": [
                {"name": "torch", "version": "2.4.1", "build_markers": {
                    "status": "static_literal_metadata_only", "fields": {"cuda": None}}},
                {"name": "transformers", "version": "4.46.3", "qwen3_vl_builtin_source_present": False},
            ],
            "tool_versions_and_daemon_runtime_metadata": {
                "docker_runtime_names": {"status": "ok", "stdout": "runc io.containerd.runc.v2\n"}},
        }
        self.plan = {"cpu_proposal": {"files": [
            {"bytes": 5, "sha256": "1" * 64, "url": "https://example.invalid/diagnostic.whl"}],
            "interpreter": {"bytes": 3}}}

    def test_realistic_cpu_install_and_unregistered_runtime_are_blocked(self):
        result = module.assess(self.inventory, self.plan)
        self.assertFalse(result["selected_runtime_ready"])
        self.assertEqual(result["readiness_exit_code"], 2)
        self.assertIn("all_observed_torch_build_markers_are_CPU_only", result["blockers"])
        self.assertIn("Docker_does_not_list_named_nvidia_runtime", result["blockers"])

    def test_even_plausible_gpu_metadata_cannot_authorize_generation(self):
        item = copy.deepcopy(self.inventory)
        item["python"]["version"] = "3.12.14"
        item["distribution_metadata"][0]["build_markers"]["fields"]["cuda"] = "11.4"
        item["distribution_metadata"][1]["qwen3_vl_builtin_source_present"] = True
        item["tool_versions_and_daemon_runtime_metadata"]["docker_runtime_names"]["stdout"] = "runc nvidia"
        result = module.assess(item, self.plan)
        self.assertEqual(result["blockers"], ["metadata_does_not_attest_import_ABI_CUDA_processor_or_model"])
        self.assertFalse(result["selected_runtime_ready"])
        self.assertEqual(result["readiness_exit_code"], 2)

    def test_missing_cuda_field_is_unknown_instead_of_cpu(self):
        del self.inventory["distribution_metadata"][0]["build_markers"]["fields"]["cuda"]
        result = module.assess(self.inventory, self.plan)
        self.assertFalse(result["observed_torch_all_CPU_build_markers"])
        self.assertIn("torch_CUDA_build_marker_unknown", result["blockers"])

    def test_ambiguous_cpu_and_gpu_metadata_does_not_choose_import_path(self):
        second = copy.deepcopy(self.inventory["distribution_metadata"][0])
        second["build_markers"]["fields"]["cuda"] = "11.4"
        self.inventory["distribution_metadata"].append(second)
        result = module.assess(self.inventory, self.plan)
        self.assertEqual(len(result["observed_packages"]["torch"]), 2)
        self.assertFalse(result["selected_runtime_ready"])

    def test_missing_hash_is_rejected_before_installation_proposal(self):
        self.plan["cpu_proposal"]["files"][0]["sha256"] = None
        with self.assertRaisesRegex(ValueError, "SHA256"):
            module.assess(self.inventory, self.plan)

    def test_nonisolated_inventory_is_rejected(self):
        self.inventory["python"]["no_site"] = False
        with self.assertRaisesRegex(ValueError, "no-site"):
            module.assess(self.inventory, self.plan)

    def test_target_inventory_parses_with_python38_grammar(self):
        ast.parse((ROOT / "scripts/agx_runtime_inventory.py").read_text(encoding="utf-8"), feature_version=(3, 8))

    def test_torch_metadata_code_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version.py"
            path.write_text("cuda = __import__('builtins').exec('raise RuntimeError(123)')\nhip = None\n", encoding="utf-8")
            class Distribution:
                def locate_file(self, unused):
                    return path
            result = inventory_module.torch_build_markers(Distribution())
            self.assertEqual(result["fields"]["cuda"], "not_a_literal")
            self.assertIsNone(result["fields"]["hip"])

    def test_partial_or_prerelease_interpreter_versions_fail_closed(self):
        for value in ("3.12", "3.12.0rc1", None, "unknown"):
            self.assertFalse(module.version_at_least(value, (3, 12, 0)))


if __name__ == "__main__":
    unittest.main()
