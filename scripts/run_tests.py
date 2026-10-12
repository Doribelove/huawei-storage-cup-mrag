"""Run the tests present in this extracted package using its own source tree."""
from pathlib import Path
import sys
import unittest

project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project/'src'), str(project/'tests'), str(project)]
suite = unittest.defaultTestLoader.discover(str(project/'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
