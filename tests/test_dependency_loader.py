import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_dependencies', PROJECT/'rag-anything-experiments/dependency_loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class DependencyLoaderTests(unittest.TestCase):
    def exercise(self, compatible, existing):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'vendor').mkdir()
            vendor=str((root/'vendor').resolve())
            original=list(sys.path)
            try:
                sys.path[:]=[vendor, 'qualified_site_packages'] if existing else ['qualified_site_packages']
                with patch.object(loader, 'vendored_runtime_matches', return_value=compatible):
                    result=loader.configure_dependencies(root)
                self.assertEqual(sys.path[0], 'qualified_site_packages')
                self.assertEqual(result['legacy_vendor_enabled'], compatible)
                self.assertEqual(sys.path.count(vendor), int(compatible))
                if compatible:self.assertEqual(sys.path[-1], vendor)
            finally:sys.path[:]=original

    def test_foreign_vendor_not_added(self):self.exercise(False, False)
    def test_foreign_vendor_precedence_removed(self):self.exercise(False, True)
    def test_matching_vendor_appended(self):self.exercise(True, False)
    def test_matching_vendor_never_shadows_installed_package(self):self.exercise(True, True)


if __name__ == '__main__':unittest.main()
