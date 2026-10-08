"""Letting the server go on without the window that started it.

The extension spawns it as an ordinary child and holds its stdout, and that pipe is the
real killer: when the extension host dies its end closes, and the next write here takes
a SIGPIPE. A server that survives the window only to die the first time it logs
something has not survived.

Both halves of the fix are things the process does to *itself*, which is what lets the
decision be made when the user is leaving rather than when they connected — the spawn is
never touched. These tests exercise the real system calls: a fake `dup2` would prove
nothing about the thing that actually breaks.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from mimir.client.ui.ws import detach as detach_mod


def _run_child(body: str, tmp: str) -> dict:
    """Run *body* in a fresh interpreter and return the dict it wrote to RESULT.

    A file rather than an inherited descriptor: ``pass_fds`` keeps a descriptor's own
    number, so writing to a hardcoded fd 3 reads whatever happens to be there. A path is
    also what survives this module's whole point, which is re-pointing the child's fds.
    """
    result = os.path.join(tmp, "result.json")
    script = (
        f"import json, os, sys\n"
        f"sys.path.insert(0, {json.dumps(os.getcwd())})\n"
        f"RESULT = {json.dumps(result)}\n"
        f"from mimir.client.ui.ws import detach as d\n"
        + body
    )
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"child failed: {proc.stderr}")
    with open(result, encoding="utf-8") as fh:
        return json.loads(fh.read())


class RedirectTests(unittest.TestCase):
    """The half that decides whether the process lives."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_the_log_is_named_by_pid_so_two_never_collide(self):
        a = detach_mod.log_path(self._tmp.name, 111)
        b = detach_mod.log_path(self._tmp.name, 222)
        self.assertNotEqual(a, b)
        self.assertTrue(a.endswith("server-111.log"))
        self.assertEqual(os.path.dirname(a), os.path.join(self._tmp.name, "logs"))

    def test_output_really_follows_the_descriptors_in_a_child_process(self):
        # In a subprocess, because dup2 on fd 1 is not something to do to the test
        # runner. This is the claim the whole feature rests on: after the redirect,
        # writes through `sys.stdout` land in the file rather than on the old fd.
        result = os.path.join(self._tmp.name, "result.json")
        script = (
            f"import json, os, sys\n"
            f"sys.path.insert(0, {json.dumps(os.getcwd())})\n"
            f"from mimir.client.ui.ws import detach as d\n"
            f'print("before the redirect")\n'
            f"info = d.detach({json.dumps(self._tmp.name)})\n"
            f'print("after the redirect")\n'
            f'sys.stderr.write("stderr too\\n")\n'
            f"sys.stdout.flush(); sys.stderr.flush()\n"
            f"open({json.dumps(result)}, 'w').write(json.dumps(info))\n"
        )
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(result, encoding="utf-8") as fh:
            info = json.loads(fh.read())

        # What the parent's pipe saw: only what was written before the redirect.
        self.assertIn("before the redirect", proc.stdout)
        self.assertNotIn("after the redirect", proc.stdout)

        # And the file has the rest, with the banner that says what it is.
        self.assertTrue(info["redirected"])
        with open(info["log"], encoding="utf-8") as fh:
            logged = fh.read()
        self.assertIn("detached; output continues here", logged)
        self.assertIn("after the redirect", logged)
        self.assertIn("stderr too", logged)

    def test_the_process_survives_writing_after_its_reader_is_gone(self):
        # The failure this exists to prevent, run for real: the parent closes the pipe,
        # and the child keeps writing. Undetached that is a SIGPIPE.
        script = f"""
import os, sys, time
sys.path.insert(0, {json.dumps(os.getcwd())})
from mimir.client.ui.ws import detach as d
d.detach({json.dumps(self._tmp.name)})
# The reader is gone by now; an un-redirected write here would end the process.
for i in range(200):
    print("still here", i)
sys.stdout.flush()
open({json.dumps(os.path.join(self._tmp.name, "survived"))}, "w").write("yes")
"""
        proc = subprocess.Popen([sys.executable, "-c", script],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.stdout.close()          # the extension host going away
        proc.wait(timeout=30)
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(self._tmp.name, "survived")))
        proc.stderr.close()

    def test_a_second_detachment_appends_rather_than_truncating(self):
        # A server can detach, be re-attached to, and detach again; all of it is one
        # run's log.
        script = f"""
import sys
sys.path.insert(0, {json.dumps(os.getcwd())})
from mimir.client.ui.ws import detach as d
d.detach({json.dumps(self._tmp.name)})
print("first")
d.detach({json.dumps(self._tmp.name)})
print("second")
sys.stdout.flush()
"""
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        logs = os.listdir(os.path.join(self._tmp.name, "logs"))
        with open(os.path.join(self._tmp.name, "logs", logs[0]), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("first", text)
        self.assertIn("second", text)

    def test_an_unwritable_log_dir_is_reported_not_raised(self):
        # Reported so the caller can say the detachment did not take, rather than the
        # WS loop dying on it.
        blocked = os.path.join(self._tmp.name, "file-not-a-dir")
        with open(blocked, "w") as fh:
            fh.write("x")
        self.assertFalse(detach_mod._redirect_output(os.path.join(blocked, "x.log")))


class LeaveProcessGroupTests(unittest.TestCase):
    """The insurance half: best effort, and never allowed to cancel the rest."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_setsid_succeeds_for_a_child_that_is_not_a_group_leader(self):
        # Which is exactly what `cp.spawn()` without `detached: true` produces: the
        # child inherits its parent's group, so it is not that group's leader.
        info = _run_child(
            "before = (os.getpid() == os.getpgid(0))\n"
            "ok = d._leave_process_group()\n"
            "open(RESULT, 'w').write(json.dumps("
            "{'was_leader': before, 'setsid': ok, 'sid': os.getsid(0), "
            "'pid': os.getpid()}))\n",
            self._tmp.name)
        self.assertFalse(info["was_leader"], "the fixture did not build a non-leader")
        self.assertTrue(info["setsid"])
        self.assertEqual(info["sid"], info["pid"], "setsid makes it its own session")

    def test_a_group_leader_is_declined_without_raising(self):
        # The normal state for a server started from a shell, and harmless: there is no
        # controlling terminal behind the extension host, so no group-wide SIGHUP is
        # coming either way.
        info = _run_child(
            "os.setsid()                     # become a leader first\n"
            "ok = d._leave_process_group()   # now it must decline, not raise\n"
            "open(RESULT, 'w').write(json.dumps({'setsid': ok}))\n",
            self._tmp.name)
        self.assertFalse(info["setsid"])

    def test_a_declined_setsid_does_not_cancel_the_detachment(self):
        # The essential half is the fds; the group is insurance. A detachment reported
        # as failed because the insurance failed would be a regression.
        # RESULT is written before the redirect takes the fds, so the assertion does
        # not depend on the thing under test having worked.
        info = _run_child(
            "import tempfile\n"
            "os.setsid()\n"
            "info = d.detach(tempfile.mkdtemp())\n"
            "open(RESULT, 'w').write(json.dumps("
            "{'redirected': info['redirected'], 'setsid': info['setsid']}))\n",
            self._tmp.name)
        self.assertFalse(info["setsid"])
        self.assertTrue(info["redirected"])


class ComingBackToARunTests(unittest.IsolatedAsyncioTestCase):
    """What the panel says when it opens onto a run that never stopped.

    The level has to be shown: a worker rebuilt during the absence would otherwise come
    up on the pool-wide record, and a run quietly dropped from `auto_all` to `manual`
    parks at its next sensitive call with nothing said. But showing it is all it is —
    nothing has gone wrong, nothing needs answering, and the only control is a switcher
    already on screen. Drawn as a warning it read as a problem to deal with, at the one
    moment the user is reading for what happened rather than for what to do.
    """

    def setUp(self) -> None:
        from unittest import mock
        from mimir.client.ui.ws import server_registry, ws_session
        from mimir.client.ui.ws.ws_session import _Session
        from mimir.tests._fake_pool import FakePool

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for target in (ws_session, server_registry):
            patcher = mock.patch.object(target, "_MIMIR_DIR_WS", self._tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.sent: list[dict] = []

        class _WS:
            async def send(_self, payload):
                self.sent.append(json.loads(payload))

        class _Worker:
            def __init__(self) -> None:
                self.mode = "manual"

            def set_approval_mode(self, mode):
                self.mode = mode

            def get_approval_mode(self):
                return self.mode

        self.sess = object.__new__(_Session)
        self.sess.ws = _WS()
        self.sess.pool = FakePool(_Worker(), active="s1")
        self.sess._active_session_id = "s1"
        self.sess._apply_setting = lambda name, *args: None

    def _detached_at(self, level: str, sessions: list[str] | None = None) -> None:
        from mimir.client.ui.ws import server_registry
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1,
                                model="m")
        server_registry.claim(sessions or ["s1"], level, log=None)

    def _notice(self) -> dict | None:
        found = [m for m in self.sent if m.get("type") == "command_output"]
        return found[0] if found else None

    async def test_it_says_what_the_run_is_under(self):
        self._detached_at("auto_all")
        await self.sess._restore_detached_autonomy()
        self.assertIn("auto_all", (self._notice() or {}).get("title", ""))

    async def test_it_is_informative_rather_than_a_warning(self):
        # The tone is the whole of this: `warn` draws a badge and a coloured border,
        # which is the vocabulary of something to deal with.
        self._detached_at("auto_all")
        await self.sess._restore_detached_autonomy()
        self.assertEqual((self._notice() or {}).get("tone"), "quiet")

    async def test_it_is_not_kept_in_the_conversation(self):
        # It describes this attachment, not anything that happened in the chat. Stored,
        # it is replayed on every load — and a conversation rejoined twenty times opens
        # on twenty copies of it, each claiming to be now.
        self._detached_at("auto_all")
        await self.sess._restore_detached_autonomy()
        self.assertTrue((self._notice() or {}).get("transient"))

    async def test_it_does_not_explain_a_control_already_on_screen(self):
        self._detached_at("auto")
        await self.sess._restore_detached_autonomy()
        self.assertEqual((self._notice() or {}).get("note", ""), "")

    async def test_the_level_still_reaches_the_panel_and_the_agents(self):
        # The notice is chrome; this is the part that must not be lost with it.
        self._detached_at("auto")
        await self.sess._restore_detached_autonomy()
        modes = [m for m in self.sent if m.get("type") == "approval_mode"]
        self.assertEqual([m["mode"] for m in modes], ["auto"])

    async def test_manual_says_nothing_at_all(self):
        # Nothing was approving anything on its own, so there is nothing to report.
        self._detached_at("manual")
        await self.sess._restore_detached_autonomy()
        self.assertIsNone(self._notice())

    async def test_each_conversation_comes_back_under_its_own_level(self):
        # The claim is per conversation and so is the level. One level for the whole
        # pool would hand a conversation left at `auto` the `auto_all` of another.
        from mimir.client.ui.ws import server_registry
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1,
                                model="m")
        server_registry.claim(["s1"], "auto")
        server_registry.claim(["s2"], "auto_all")
        self.assertEqual(server_registry.claims(), {"s1": "auto", "s2": "auto_all"})

        await self.sess._restore_detached_autonomy()
        modes = [m["mode"] for m in self.sent if m.get("type") == "approval_mode"]
        self.assertEqual(modes, ["auto"], "the level of the conversation on screen")

    async def test_taking_it_back_for_one_leaves_the_others_running(self):
        # The process survives for as long as any conversation claims it. Clearing one
        # flag for the whole workspace made another conversation's run mortal without
        # anybody asking for that.
        from mimir.client.ui.ws import server_registry
        self._detached_at("auto_all", ["s1", "s2"])
        left = server_registry.unclaim(["s1"])
        self.assertEqual(left, {"s2": "auto_all"})
        self.assertTrue(server_registry.read()["detached"])

    async def test_taking_it_back_for_the_last_one_makes_it_mortal(self):
        from mimir.client.ui.ws import server_registry
        self._detached_at("auto_all", ["s1"])
        self.assertEqual(server_registry.unclaim(["s1"]), {})
        self.assertFalse(server_registry.read()["detached"])

    async def test_a_server_that_never_detached_says_nothing(self):
        from mimir.client.ui.ws import server_registry
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1,
                                model="m")
        await self.sess._restore_detached_autonomy()
        self.assertEqual(self.sent, [])


class DetachHandlerTests(unittest.IsolatedAsyncioTestCase):
    """What the `detach` message does besides making the process survivable."""

    def setUp(self) -> None:
        from unittest import mock
        from mimir.client.ui.ws import server_registry, ws_session
        from mimir.client.ui.ws.ws_session import _Session
        from mimir.tests._fake_pool import FakePool

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for target in (ws_session, server_registry):
            patcher = mock.patch.object(target, "_MIMIR_DIR_WS", self._tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)
        # The process-level work is exercised for real in the tests above; here it must
        # not touch the test runner's own descriptors.
        patcher = mock.patch.object(
            ws_session, "detach_process",
            return_value={"log": "/tmp/x.log", "redirected": True, "setsid": True,
                          "sighup_ignored": True, "pid": 4242})
        self.detach_called = patcher.start()
        self.addCleanup(patcher.stop)

        class _Worker:
            def __init__(self) -> None:
                self.mode = "manual"

            def set_approval_mode(self, mode):
                self.mode = mode

            def get_approval_mode(self):
                return self.mode

        self.a, self.b = _Worker(), _Worker()
        self.sent: list[dict] = []

        class _WS:
            async def send(_self, payload):
                self.sent.append(json.loads(payload))

        self.sess = object.__new__(_Session)
        self.sess.ws = _WS()
        self.sess.pool = FakePool(self.a, active="s1",
                                  workers={"s1": self.a, "s2": self.b})
        self.sess._active_session_id = "s1"
        self.applied: list[tuple] = []
        self.sess._apply_setting = lambda name, *args: self.applied.append((name, args))

    async def test_it_replies_with_what_it_managed(self):
        await self.sess._handle_detach({"type": "detach", "autonomy": "auto"})
        reply = [m for m in self.sent if m["type"] == "detached"][0]
        self.assertEqual(reply["autonomy"], "auto")
        self.assertEqual(reply["pid"], 4242)
        self.assertEqual(reply["log"], "/tmp/x.log")
        self.assertTrue(self.detach_called.called)

    async def test_naming_conversations_sets_only_those(self):
        await self.sess._handle_detach(
            {"type": "detach", "autonomy": "auto_all", "session_ids": ["s2"]})
        self.assertEqual(self.b.mode, "auto_all")
        self.assertEqual(self.a.mode, "manual", "a conversation not named was changed")
        self.assertEqual([name for name, _args in self.applied],
                         ["set_non_interactive"],
                         "a per-session detach recorded a level pool-wide")
        self.assertEqual(
            [m for m in self.sent if m["type"] == "detached"][0]["sessions"], ["s2"])

    async def test_naming_none_records_the_level_pool_wide(self):
        # The only form that outlives a worker being rebuilt, since the pool records UI
        # settings per pool rather than per session.
        await self.sess._handle_detach({"type": "detach", "autonomy": "auto"})
        self.assertIn(("set_approval_mode", ("auto",)), self.applied)

    async def test_there_is_no_terminal_whichever_conversations_are_named(self):
        # Recorded pool-wide even for a per-session detach, because it is a fact about
        # the *process*: its stdout is a log file from here on, and that cannot be
        # undone. A detached process goes on building agents — a conversation whose own
        # was released gets it back when a run of its finishes — and one built without
        # this would reach for a terminal that is a log file.
        await self.sess._handle_detach(
            {"type": "detach", "autonomy": "auto_all", "session_ids": ["s2"]})
        self.assertIn(("set_non_interactive", (True,)), self.applied)

    async def test_an_unknown_level_is_refused_and_nothing_is_detached(self):
        await self.sess._handle_detach({"type": "detach", "autonomy": "yolo"})
        self.assertEqual([m["type"] for m in self.sent], ["error"])
        self.assertFalse(self.detach_called.called)

    async def test_a_missing_level_is_the_safe_one(self):
        # Not an error: "continue without me" with nothing said about autonomy means
        # the run survives and parks at its first sensitive tool.
        await self.sess._handle_detach({"type": "detach"})
        self.assertEqual(
            [m for m in self.sent if m["type"] == "detached"][0]["autonomy"], "manual")

    async def test_a_conversation_with_no_agent_is_skipped_quietly(self):
        await self.sess._handle_detach(
            {"type": "detach", "autonomy": "auto", "session_ids": ["never-existed"]})
        self.assertEqual(
            [m for m in self.sent if m["type"] == "detached"][0]["sessions"], [])

    async def test_the_registry_entry_records_the_detachment(self):
        from mimir.client.ui.ws import server_registry
        server_registry.publish(url="ws://127.0.0.1:9", host="127.0.0.1", port=9)
        await self.sess._handle_detach({"type": "detach", "autonomy": "auto"})
        entry = server_registry.read()
        self.assertTrue(entry["detached"])
        self.assertEqual(entry["autonomy"], "auto")
        # The address it was serving on is kept: it was settled at bind time and has
        # not changed.
        self.assertEqual(entry["url"], "ws://127.0.0.1:9")

    async def test_taking_the_decision_back_clears_the_claim(self):
        # Detaching does not disconnect: the socket stays open and the turn goes on in
        # front of the user, so changing their mind has to be possible. What makes a
        # server survive a window closing is that nobody kills it — a decision, not a
        # state of the process.
        from mimir.client.ui.ws import server_registry
        server_registry.publish(url="ws://127.0.0.1:9", host="127.0.0.1", port=9)
        await self.sess._handle_detach({"type": "detach", "autonomy": "auto"})
        self.assertTrue(server_registry.read()["detached"])

        self.sent.clear()
        await self.sess._handle_detach({"type": "detach", "enabled": False})
        self.assertFalse(server_registry.read()["detached"])
        reply = [m for m in self.sent if m["type"] == "detached"][0]
        self.assertIs(reply["detached"], False)

    async def test_taking_it_back_does_not_redo_the_process_work(self):
        # The redirect cannot be undone and does not need to be: fds pointing at a log
        # file are harmless either way.
        await self.sess._handle_detach({"type": "detach", "autonomy": "auto"})
        self.detach_called.reset_mock()
        await self.sess._handle_detach({"type": "detach", "enabled": False})
        self.assertFalse(self.detach_called.called)

    async def test_a_detachment_says_so_explicitly(self):
        # The flag is on the message rather than implied by its arrival, so the two
        # directions cannot be told apart by accident.
        await self.sess._handle_detach({"type": "detach", "autonomy": "auto"})
        reply = [m for m in self.sent if m["type"] == "detached"][0]
        self.assertIs(reply["detached"], True)

    async def test_no_registry_entry_is_not_an_error(self):
        await self.sess._handle_detach({"type": "detach", "autonomy": "auto"})
        self.assertEqual(
            [m["type"] for m in self.sent], ["detached", "command_output"])

    async def test_each_direction_of_the_switch_says_so_in_the_thread(self):
        # The button is one glyph in both positions, so the thread is where the user
        # reads which way it went — and the line names the consequence, since the word
        # alone does not say that closing the window now ends the run.
        from mimir.client.ui.ws import server_registry
        server_registry.publish(url="ws://127.0.0.1:9", host="127.0.0.1", port=9)

        await self.sess._handle_detach({"type": "detach", "autonomy": "auto_all"})
        note = [m for m in self.sent if m["type"] == "command_output"][-1]
        self.assertEqual(note["command"], "detach")
        self.assertEqual(note["tone"], "quiet")
        # Never stored: it describes this moment, not the conversation.
        self.assertTrue(note["transient"])
        self.assertIn("detached", note["title"])
        self.assertIn("auto_all", note["title"])

        self.sent.clear()
        await self.sess._handle_detach({"type": "detach", "enabled": False})
        note = [m for m in self.sent if m["type"] == "command_output"][-1]
        self.assertIn("attached", note["title"])
        self.assertIn("closing it", note["title"])

    async def test_a_manual_detachment_says_it_will_park(self):
        # "under manual" reads as "it will finish"; it will not.
        await self.sess._handle_detach({"type": "detach", "autonomy": "manual"})
        note = [m for m in self.sent if m["type"] == "command_output"][-1]
        self.assertIn("parks at the first call", note["title"])


if __name__ == "__main__":
    unittest.main()
