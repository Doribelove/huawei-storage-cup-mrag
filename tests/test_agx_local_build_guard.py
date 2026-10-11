import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location('agx_guard_test', Path(__file__).parents[1] / 'scripts/agx_local_build_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class LocalGuardScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def process(self, pid, parent, group, session, ticks, children=''):
        directory = self.root / str(pid)
        (directory / 'task' / str(pid)).mkdir(parents=True, exist_ok=True)
        fields = ['S', str(parent), str(group), str(session), '0'] + ['0'] * 14 + [str(ticks)]
        (directory / 'stat').write_text(str(pid) + ' (a name with (brackets)) ' + ' '.join(fields))
        (directory / 'task' / str(pid) / 'children').write_text(children)
        return guard.process_snapshot(pid, self.root)

    def test_only_registered_tree_and_session_are_authorized(self):
        root = self.process(910001, 1, 910001, 910001, 123, '910002')
        self.process(910002, 910001, 910001, 910001, 124)
        self.process(910003, 1, 910003, 910003, 125)  # unrelated robot process
        known = guard.observe_descendants({910001: root}, self.root)
        self.assertEqual(set(known), {910001, 910002})
        self.assertEqual(guard.verified_groups(known, {910001}, self.root), [910001])

    def test_pid_reuse_cannot_authorize_a_signal(self):
        captured = self.process(910001, 1, 910001, 910001, 123)
        self.process(910001, 1, 910001, 910001, 999)
        self.assertEqual(guard.verified_groups({910001: captured}, {910001}, self.root), [])

    def test_unregistered_new_child_session_is_excluded(self):
        root = self.process(910001, 1, 910001, 910001, 123, '910002')
        self.process(910002, 910001, 910002, 910002, 124)
        known = guard.observe_descendants({910001: root}, self.root)
        self.assertEqual(guard.verified_groups(known, {910001}, self.root), [910001])

    def test_new_session_requires_observed_ancestry_and_task_location(self):
        root = self.process(910001, 1, 910001, 910001, 123, '910002')
        self.process(910002, 910001, 910002, 910002, 124)
        known = guard.observe_descendants({910001: root}, self.root)
        with patch.object(guard, 'task_location_matches', return_value=False):
            self.assertEqual(guard.register_descendant_sessions(known, {910001}, self.root, self.root), {910001})
        with patch.object(guard, 'task_location_matches', return_value=True):
            self.assertEqual(guard.register_descendant_sessions(known, {910001}, self.root, self.root), {910001, 910002})

    def test_dead_processes_are_pruned_and_not_counted_forever(self):
        captured = self.process(910001, 1, 910001, 910001, 123)
        (self.root / '910001/stat').unlink()
        self.assertEqual(guard.observe_descendants({910001: captured}, self.root), {})

    def test_orphaned_known_native_group_remains_bound(self):
        captured = self.process(910002, 910001, 910002, 910002, 124)
        self.process(910002, 1, 910002, 910002, 124)
        self.assertEqual(guard.verified_groups({910002: captured}, {910002}, self.root), [910002])


if __name__ == '__main__':
    unittest.main()
