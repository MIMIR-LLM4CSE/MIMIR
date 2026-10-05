"""Shared resources, once several sessions use them at the same time.

Everything here was correct while one session lived at a time, and stops being correct
the moment two do. Three distinct shapes of the same mistake:

* **A single-slot pointer** standing in for "the" current thing. The proxy store's
  ``opt_runs/active_session`` named the proxy a nameless ``proxy_eval`` op acts on, so a
  second session initialising its own optimisation retargeted the first one's ops — a run
  meant for one proxy was measured against another.
* **Test-then-create.** ``env_create`` checked ``os.path.exists`` and then created, so two
  sessions asking for the same environment name ran ``venv`` into each other's tree.
* **Read-modify-write, and truncating writes.** ``preferences.json`` loaded, merged and
  wrote atomically — but not the cycle, so the second of two concurrent toggles dropped
  the first. The memory store wrote notes and its index with a plain truncating open, so a
  concurrent reader could load half a file.

What stays shared is deliberate, and the last test says so: the memory store, the Python
environments, the module catalogue and the proxy store belong to the *workspace*. Making
them per conversation would cost minutes and gigabytes on every new one.

Pure-Python + temp dirs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "proxy" / "_lib",
           SERVERS_DIR / "hpc"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class ProxyActiveSessionPointerTests(unittest.TestCase):
    """Which optimisation a conversation is driving is that conversation's own."""

    def setUp(self) -> None:
        from mimir.servers.proxy._lib import store
        self.store = store
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_cache = store._CACHE_DIR
        store._CACHE_DIR = os.path.join(self._tmp.name, "proxy_bench")
        self.addCleanup(setattr, store, "_CACHE_DIR", self._orig_cache)
        env = patch.dict(os.environ, {"MIMIR_STATE_DIR": self._tmp.name}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MIMIR_SESSION_ID", None)

    def test_one_session_init_does_not_retarget_another(self) -> None:
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s1"}):
            self.store._write_active_session("alpha")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s2"}):
            self.store._write_active_session("beta")
            self.assertEqual(self.store._resolve_proxy_name(""), "beta")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s1"}):
            self.assertEqual(self.store._resolve_proxy_name(""), "alpha")

    def test_clearing_one_session_leaves_the_other_pointed(self) -> None:
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s1"}):
            self.store._write_active_session("alpha")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s2"}):
            self.store._write_active_session("beta")
            self.store._clear_active_session()
            self.assertIsNone(self.store._resolve_proxy_name(""))
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s1"}):
            self.assertEqual(self.store._resolve_proxy_name(""), "alpha")

    def test_an_explicit_name_still_wins_over_any_pointer(self) -> None:
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s1"}):
            self.store._write_active_session("alpha")
            self.assertEqual(self.store._resolve_proxy_name("named"), "named")

    def test_a_session_less_end_keeps_the_unsuffixed_pointer(self) -> None:
        """The CLI and the tests: same file name it has always had."""
        self.assertTrue(self.store.active_session_file().endswith("active_session"))

    def test_the_client_resolves_the_session_it_is_acting_for(self) -> None:
        """In the client, N sessions share one environment — the agent must say."""
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "s1"}):
            self.store._write_active_session("alpha")
        self.store._write_active_session("fallback", session_id="s2")
        self.assertEqual(self.store._resolve_proxy_name("", "s1"), "alpha")
        self.assertEqual(self.store._resolve_proxy_name("", "s2"), "fallback")


class EnvCreateRaceTests(unittest.TestCase):
    """The name is claimed with the one operation the filesystem makes atomic."""

    def setUp(self) -> None:
        import server_env
        self.server_env = server_env
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig = self.server_env._ENV_HOME
        self.server_env._ENV_HOME = os.path.join(self._tmp.name, "envs")
        self.addCleanup(setattr, self.server_env, "_ENV_HOME", self._orig)

    def test_the_second_request_for_a_name_is_refused_not_merged(self) -> None:
        # Stand in for the venv build: the claim is what this is about, not the build.
        self.addCleanup(setattr, self.server_env, "_run", self.server_env._run)
        self.server_env._run = lambda argv: {"ok": True, "stderr": "", "returncode": 0}
        first = self.server_env.env_create("shared-name")
        second = self.server_env.env_create("shared-name")
        self.assertEqual(first.get("status"), "ok")
        self.assertEqual(second.get("status"), "error")
        self.assertIn("already exists", second.get("error", ""))

    def test_a_failed_build_does_not_leave_the_name_claimed(self) -> None:
        self.addCleanup(setattr, self.server_env, "_run", self.server_env._run)
        self.server_env._run = lambda argv: {
            "ok": False, "stderr": "boom", "returncode": 1}
        self.assertEqual(self.server_env.env_create("doomed").get("status"), "error")
        self.assertFalse(os.path.exists(os.path.join(self.server_env._ENV_HOME, "doomed")))


class PreferencesConcurrencyTests(unittest.TestCase):
    """Two sessions toggling at once must not drop one of the two changes."""

    def setUp(self) -> None:
        from mimir.client.config import preferences
        self.preferences = preferences
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(preferences, "STATE_DIR", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_concurrent_writers_keep_both_keys(self) -> None:
        barrier = threading.Barrier(2)

        def toggle_servers() -> None:
            barrier.wait()
            for _ in range(40):
                self.preferences.save_disabled({"web"}, set(), set())

        def set_temperature() -> None:
            barrier.wait()
            for _ in range(40):
                self.preferences.save_temperature("some-model", 0.5)

        threads = [threading.Thread(target=toggle_servers),
                   threading.Thread(target=set_temperature)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        payload = self.preferences.load_preferences()
        self.assertEqual(payload.get("disabled_servers"), ["web"])
        self.assertEqual(payload.get("temperatures"), {"some-model": 0.5})


class AtomicWriteTests(unittest.TestCase):
    """A reader must see the old file or the new one, never half of either."""

    def test_the_memory_store_writes_through_a_rename(self) -> None:
        import inspect
        from mimir.servers.agent_state import server_memory
        for fn in (server_memory._write_index, server_memory._save_embeddings):
            self.assertIn("_write_text_atomic", inspect.getsource(fn), fn.__name__)
        self.assertIn("os.replace", inspect.getsource(server_memory._write_text_atomic))

    def test_the_vector_cache_writes_through_a_rename(self) -> None:
        import json as _json
        import vector_cache
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "vectors.json")
            vector_cache.save_vectors(path, {"a": [1.0, 2.0]})
            with open(path) as fh:
                self.assertEqual(_json.load(fh), {"a": [1.0, 2.0]})
            # Nothing half-written left behind for a reader to trip over.
            self.assertEqual(os.listdir(d), ["vectors.json"])


class WhatStaysSharedTests(unittest.TestCase):
    """The deliberate half of the decision, stated so a later change has to argue.

    These belong to the workspace, not to a conversation. Per session they would be
    rebuilt for every new chat — minutes for a venv or a module catalogue, gigabytes for
    sealed reference fields — and the memory store exists precisely to outlive one.
    """

    def test_workspace_goods_resolve_off_the_state_dir_root(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"MIMIR_STATE_DIR": d, "MIMIR_SESSION_ID": "s1"}):
                from state_paths import session_state_dir, state_dir
                self.assertEqual(state_dir(), d)
                # The conversation's own corner is a subdirectory of it, not a
                # replacement for it.
                self.assertTrue(session_state_dir().startswith(d))
                self.assertNotEqual(session_state_dir(), state_dir())

    def test_the_memory_dir_is_not_per_session(self) -> None:
        import inspect
        from mimir.servers.agent_state import server_memory
        source = inspect.getsource(server_memory)
        self.assertIn("state_dir()", source)
        self.assertNotIn("session_state_dir", source)


if __name__ == "__main__":
    unittest.main()


class RevertProtectsConcurrentWorkTests(unittest.TestCase):
    """A revert undoes the diff the user looked at — and only that.

    Snapshots are per agent, but the files on disk are not. With two conversations
    editing one file, reverting in the first wrote its own pre-edit baseline over the
    second's work, which nobody asked to discard. The reviewed digest is the seam: it is
    recorded when the diff is built, which is the one moment what is on disk and what the
    user is being shown are known to be the same thing.
    """

    def _manager(self):
        from mimir.client.guardrails.policy.approval import ApprovalManager
        return ApprovalManager()

    def test_a_file_untouched_since_the_review_may_be_reverted(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.py")
            with open(path, "w") as fh:
                fh.write("edited\n")
            approvals = self._manager()
            approvals.note_reviewed(path, "edited\n")
            self.assertIs(approvals.reviewed_matches(path), True)

    def test_a_file_changed_since_the_review_may_not(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.py")
            with open(path, "w") as fh:
                fh.write("edited\n")
            approvals = self._manager()
            approvals.note_reviewed(path, "edited\n")
            with open(path, "w") as fh:          # another session, or the user's editor
                fh.write("edited, then edited again by someone else\n")
            self.assertIs(approvals.reviewed_matches(path), False)

    def test_a_file_deleted_since_the_review_counts_as_moved(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.py")
            with open(path, "w") as fh:
                fh.write("edited\n")
            approvals = self._manager()
            approvals.note_reviewed(path, "edited\n")
            os.remove(path)
            self.assertIs(approvals.reviewed_matches(path), False)

    def test_an_unrecorded_file_abstains_rather_than_blocking(self) -> None:
        """A guard that cannot tell must not stand in the way of the user's own undo."""
        approvals = self._manager()
        self.assertIsNone(approvals.reviewed_matches("/nowhere/at/all.py"))

    def test_building_the_review_diff_is_what_records_the_digest(self) -> None:
        """The two ends of the seam, so neither can drift from the other."""
        import inspect
        from mimir.client.ui.ws import ws_worker, ws_session
        self.assertIn("note_reviewed", inspect.getsource(ws_worker._AgentWorker._build_batch_status))
        self.assertIn("reviewed_matches", inspect.getsource(ws_session._Session._revert_one))
