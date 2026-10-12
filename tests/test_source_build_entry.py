"""Build-entry syntax regression; actual CPU/CUDA builds are separate measurements."""
from pathlib import Path
import unittest
from evidence_lab.semantic import LocalBGE


class SourceBuildEntryTests(unittest.TestCase):
    def test_production_python_sources_and_entries_compile(self):
        project = Path(__file__).resolve().parents[1]
        files = sorted((project/'src').rglob('*.py')) + sorted((project/'scripts').glob('*.py'))
        self.assertTrue(files)
        for path in files:
            with self.subTest(file=str(path.relative_to(project))):
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_unsupported_device_cannot_fall_back_to_a_different_runtime(self):
        for device in ('auto', 'cuda:1', 'mps', None, True):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, 'explicit CPU or CUDA'):
                LocalBGE(Path('/nonexistent-device-validation-fixture'), device=device)


if __name__ == '__main__':
    unittest.main()
