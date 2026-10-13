"""Every path that belongs to a conversation names *that* conversation.

MIMIR runs several sessions at the same time, each with its own agent and so its own set
of server subprocesses. A single file — ``<STATE_DIR>/active_session`` — can name only one
of them, so every other session reading it acts for the wrong conversation: it writes to
another's todo list, reads another's plans, announces its blocking run in another's
channel, and — the one that is not merely untidy — inherits and revokes another's approved
paths.

So the session reaches each end the only way that is correct for it:

* **servers** get ``MIMIR_SESSION_ID`` stamped into their environment at spawn. One
  agent owns a server for its whole life, so the session is fixed and the frozen
  environment is the right carrier.
* **the client** passes it explicitly, from the object that knows it. N sessions share
  one ``os.environ`` in that process, so the environment cannot carry it there.

``session_id=None`` means "this end has no session of its own" — the CLI, the benchmark
runner, these tests — and resolves through the pointer and then the state-dir root. That
fallback is what keeps the single-session ends resolving the paths they would without it.

Pure-Python + temp dirs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mimir.servers._shared.state_paths import (
    active_session_id,
    global_state_dir,
    scratch_dir,
    session_state_dir,
)


class _StateDirCase(unittest.TestCase):
    """A throwaway state dir, with no session named anywhere."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = os.path.join(self._tmp.name, "state")
        os.makedirs(self.state)
        # MIMIR_SESSION_ID must be *absent*, not empty: the suite may inherit one.
        env = patch.dict(os.environ, {"MIMIR_STATE_DIR": self.state}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MIMIR_SESSION_ID", None)

    def write_pointer(self, session_id: str) -> None:
        with open(os.path.join(self.state, "active_session"), "w", encoding="utf-8") as fh:
            fh.write(session_id)


class ActiveSessionResolutionTests(_StateDirCase):
    """Which answer wins, and why that order and not the other one."""

    def test_environment_beats_the_pointer(self):
        """The whole point: a server answers for ITS session, not the visible one."""
        self.write_pointer("the-one-on-screen")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "mine"}):
            self.assertEqual(active_session_id(), "mine")

    def test_the_pointer_answers_for_the_single_session_ends(self):
        self.write_pointer("only-session")
        self.assertEqual(active_session_id(), "only-session")

    def test_an_empty_environment_variable_is_not_an_answer(self):
        """Servers of a session-less agent get "" — that must not shadow the pointer."""
        self.write_pointer("only-session")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "   "}):
            self.assertEqual(active_session_id(), "only-session")

    def test_no_session_anywhere_is_the_empty_string(self):
        self.assertEqual(active_session_id(), "")


class SessionStateDirTests(_StateDirCase):
    """The per-conversation directory, and the fallback that preserved behaviour."""

    def test_two_sessions_get_two_directories(self):
        a = session_state_dir(self.state, "aaa")
        b = session_state_dir(self.state, "bbb")
        self.assertNotEqual(a, b)
        self.assertEqual(a, os.path.join(self.state, "sessions", "aaa"))

    def test_no_session_falls_back_to_the_state_dir_itself(self):
        """What keeps the CLI's paths where the CLI expects them."""
        self.assertEqual(session_state_dir(self.state, ""), self.state)
        self.assertEqual(session_state_dir(self.state), self.state)

    def test_none_resolves_through_the_pointer_but_an_explicit_id_does_not(self):
        """``None`` means "I have no session"; "" means "no session, full stop"."""
        self.write_pointer("pointed-at")
        self.assertEqual(
            session_state_dir(self.state), os.path.join(self.state, "sessions", "pointed-at")
        )
        self.assertEqual(session_state_dir(self.state, ""), self.state)

    def test_resolving_a_path_creates_nothing(self):
        """A read-only path check must not materialise directories."""
        session_state_dir(self.state, "ghost")
        self.assertFalse(os.path.exists(os.path.join(self.state, "sessions")))


class ScratchIsolationTests(_StateDirCase):
    def test_two_sessions_get_two_scratchpads(self):
        with patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": "/tmp/scratch-home"}):
            self.assertEqual(scratch_dir(self.state, "aaa"), "/tmp/scratch-home/aaa")
            self.assertEqual(scratch_dir(self.state, "bbb"), "/tmp/scratch-home/bbb")

    def test_an_explicit_id_beats_the_environment(self):
        """The client half: one process, N sessions, one os.environ."""
        with patch.dict(os.environ, {"MIMIR_SCRATCH_DIR": "/tmp/scratch-home",
                                     "MIMIR_SESSION_ID": "someone-else"}):
            self.assertEqual(scratch_dir(self.state, "mine"), "/tmp/scratch-home/mine")


class ApprovedPathsAreNotSharedTests(_StateDirCase):
    """The security half: a grant widens the sandbox of the session that was asked.

    One file per workspace would have a second session *truncate* the first one's
    allowlist mid-run — revoking paths it is actively writing under — and then inherit
    whatever the first was granted.
    """

    def _manager(self, session_id):
        from mimir.client.guardrails.policy.approval import ApprovalManager
        return ApprovalManager(session_id=session_id)

    def _patched_state_dir(self):
        # The client reads STATE_DIR frozen at import, so point it at the temp dir.
        return patch("mimir.client.config.constants.STATE_DIR", self.state)

    def test_each_session_writes_its_own_file(self):
        with self._patched_state_dir():
            a, b = self._manager("aaa"), self._manager("bbb")
            self.assertNotEqual(a.approved_paths_file(), b.approved_paths_file())
            self.assertIn(os.path.join("sessions", "aaa"), a.approved_paths_file())

    def test_a_grant_in_one_session_is_invisible_to_the_other(self):
        with self._patched_state_dir():
            a, b = self._manager("aaa"), self._manager("bbb")
            a.grant_path("/data/elsewhere", always=False)
            self.assertEqual(self._read(a), [os.path.realpath("/data/elsewhere")])
            self.assertEqual(self._read(b), [])

    def test_resetting_one_session_leaves_the_others_grants_standing(self):
        with self._patched_state_dir():
            a, b = self._manager("aaa"), self._manager("bbb")
            a.grant_path("/data/a-needs-this", always=True)
            b.reset_allowed_paths()
            self.assertEqual(self._read(a), [os.path.realpath("/data/a-needs-this")])

    def test_a_session_less_agent_keeps_the_root_file(self):
        with self._patched_state_dir():
            cli = self._manager(None)
            self.assertEqual(
                cli.approved_paths_file(), os.path.join(self.state, "approved_paths.json")
            )

    def _read(self, manager):
        try:
            with open(manager.approved_paths_file()) as fh:
                return json.load(fh)
        except OSError:
            return []


class RunChannelIsolationTests(_StateDirCase):
    """Two conversations can each have a blocking run; a divert must reach its own."""

    def test_the_two_ends_agree_on_one_directory_per_session(self):
        from mimir.client.tool_execution import run_channel as client_side
        # STATE_DIR is bound into this module at import, so patch the name it holds.
        with patch.object(client_side, "STATE_DIR", self.state):
            mine = client_side._dir("bash_run", "aaa")
            theirs = client_side._dir("bash_run", "bbb")
        self.assertNotEqual(mine, theirs)
        self.assertEqual(
            mine, os.path.join(self.state, "sessions", "aaa", "runs", "bash_run")
        )

    def test_the_server_half_resolves_the_same_path_from_its_environment(self):
        """Server and client must name the same file, or a divert reaches nothing."""
        import sys
        sys.path.insert(0, os.path.join("mimir", "servers", "_shared"))
        self.addCleanup(sys.path.pop, 0)
        import importlib
        server_side = importlib.import_module("run_channel")
        with patch.dict(os.environ, {"MIMIR_SESSION_ID": "aaa"}):
            self.assertEqual(
                server_side._dir("bash_run"),
                os.path.join(self.state, "sessions", "aaa", "runs", "bash_run"),
            )


class AgentCarriesItsSessionTests(_StateDirCase):
    """The id reaches the places that used to read the pointer."""

    def test_the_todo_file_named_in_the_system_prompt_is_the_agents_own(self):
        """It goes into the prompt, so the pointer would aim every agent at one list."""
        from mimir.client.agent_core import MimirAgent
        with patch("mimir.client.config.constants.STATE_DIR", self.state), \
             patch("mimir.client.agent_core.STATE_DIR", self.state):
            self.write_pointer("the-one-on-screen")
            agent = MimirAgent(session_id="mine")
            # _get_todo_file is gated on a planning tool being connected.
            with patch("mimir.client.agent_core.names_with_cap", return_value=["todo_write"]):
                self.assertEqual(
                    agent._get_todo_file(),
                    os.path.join(self.state, "sessions", "mine", "todo_list.md"),
                )

    def test_the_approval_manager_inherits_the_agents_session(self):
        from mimir.client.agent_core import MimirAgent
        agent = MimirAgent(session_id="mine")
        self.assertEqual(agent.approvals.session_id, "mine")

    def test_servers_are_told_which_session_they_serve(self):
        """What makes every server-side state path per-conversation."""
        import inspect
        from mimir.client.integration import server_manager
        # The spawn half holds the frozen environment every server is started with.
        source = inspect.getsource(server_manager._spawn_session)
        self.assertIn("MIMIR_SESSION_ID", source)
        self.assertIn('getattr(agent, "session_id", "")', source)


class GlobalStateDirTests(unittest.TestCase):
    """The tier above the workspace: one directory for the whole machine.

    What is tested hardest here is where it must NOT land. This tier holds the memory
    shared by every workspace, so a resolution that reaches the user's real home turns
    any test run into a write to their own memory, and one that reaches ``/tmp``
    directly puts it in a world-writable path shared between users.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # All three must be absent rather than empty: the suite may inherit any of them.
        for var in ("MIMIR_GLOBAL_STATE_DIR", "MIMIR_STATE_HOME", "MIMIR_STATE_DIR"):
            env = patch.dict(os.environ, {}, clear=False)
            env.start()
            self.addCleanup(env.stop)
            os.environ.pop(var, None)

    def test_the_explicit_variable_wins(self) -> None:
        pinned = os.path.join(self._tmp.name, "pinned")
        os.environ["MIMIR_GLOBAL_STATE_DIR"] = pinned
        os.environ["MIMIR_STATE_HOME"] = os.path.join(self._tmp.name, "home")
        self.assertEqual(global_state_dir(), pinned)
        # A passed home does not override what the environment pins.
        self.assertEqual(global_state_dir(os.path.join(self._tmp.name, "other")), pinned)

    def test_a_passed_home_is_used_before_the_environment(self) -> None:
        os.environ["MIMIR_STATE_HOME"] = os.path.join(self._tmp.name, "env-home")
        home = os.path.join(self._tmp.name, "client-home")
        self.assertEqual(global_state_dir(home), os.path.join(home, "global"))

    def test_the_state_home_is_the_next_fallback(self) -> None:
        home = os.path.join(self._tmp.name, "home")
        os.environ["MIMIR_STATE_HOME"] = home
        self.assertEqual(global_state_dir(), os.path.join(home, "global"))

    def test_with_only_a_state_dir_it_stays_inside_it(self) -> None:
        # Not dirname(state_dir): the suite points MIMIR_STATE_DIR at a mkdtemp(), whose
        # parent is /tmp. Collapsing the tier inside the state dir is hermetic instead.
        state = os.path.join(self._tmp.name, "state")
        os.environ["MIMIR_STATE_DIR"] = state
        resolved = global_state_dir()
        self.assertEqual(resolved, os.path.join(state, "global"))
        self.assertNotEqual(resolved, os.path.join(os.path.dirname(state), "global"))

    def test_it_never_reaches_the_users_home_when_a_state_dir_is_set(self) -> None:
        # The regression that would quietly write the developer's own global memory.
        os.environ["MIMIR_STATE_DIR"] = os.path.join(self._tmp.name, "state")
        self.assertFalse(
            global_state_dir().startswith(os.path.realpath(os.path.expanduser("~")) + os.sep)
        )

    def test_with_only_a_files_root_it_stays_inside_the_workspace(self) -> None:
        root = os.path.join(self._tmp.name, "workspace")
        with patch.dict(os.environ, {"MCP_FILES_ROOT": root}, clear=False):
            self.assertEqual(global_state_dir(),
                             os.path.join(root, ".mimir", "global"))

    def test_it_creates_nothing(self) -> None:
        home = os.path.join(self._tmp.name, "home")
        self.assertFalse(os.path.exists(global_state_dir(home)))


if __name__ == "__main__":
    unittest.main()
