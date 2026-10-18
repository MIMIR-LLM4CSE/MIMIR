"""The webview's slash list (``SESSION_COMMANDS`` in ``slashUtils.ts``) names exactly the
commands the backend handles; it is mirrored by hand, so this test stops drift.
"""
from __future__ import annotations

import asyncio
import os
import re
import unittest

from mimir.client.config import THINKING_DEPTH_LABELS
from mimir.client.core.settings import read_setting
from mimir.client.session.commands import CommandHost, CommandResult
from mimir.client.ui.cli.chat_commands import CLI_COMMANDS
from mimir.client.ui.ws.connection.commands import WS_COMMANDS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SLASH_UTILS = os.path.join(
    _ROOT, "vscode-extension", "webview", "src", "components", "slashUtils.ts")


def _advertised() -> list[str]:
    """Command names the webview offers in its dropdown."""
    with open(_SLASH_UTILS, encoding="utf-8") as fh:
        src = fh.read()
    block = src.split("SESSION_COMMANDS", 1)[1].split("];", 1)[0]
    return re.findall(r'name:\s*"([^"]+)"', block)


class AdvertisedCommandsExistTests(unittest.TestCase):
    def test_the_list_is_not_empty(self) -> None:
        # A parser that silently matched nothing would make every assertion below vacuous.
        self.assertGreater(len(_advertised()), 5)

    def test_every_advertised_command_is_handled_by_the_session(self) -> None:
        missing = [name for name in _advertised() if WS_COMMANDS.find(f"/{name}") is None]
        self.assertEqual(
            missing, [],
            "the webview offers these but the socket's registry does not handle them — "
            f"they would autocomplete and do nothing: {missing}")

    def test_every_handled_command_is_advertised(self) -> None:
        """Every handled command is advertised; an unlisted one would leave as a plain query for
        the model.
        """
        # Answered from the model's own surface rather than the panel's: the webview has
        # no business knowing these.
        model_facing = {"backend"}
        unrouted = sorted(set(WS_COMMANDS.names()) - set(_advertised()) - model_facing)
        self.assertEqual(
            unrouted, [],
            "the socket handles these but the webview does not route them as commands, "
            f"so they reach the model as plain queries instead: {unrouted}")

    def test_the_housekeeping_commands_are_on_both_surfaces(self) -> None:
        # A command added to one surface and not the other is a command that works in
        # one window and not the next.
        for name in ("memory", "proxy", "modules"):
            self.assertIn(name, _advertised())
            self.assertIsNotNone(CLI_COMMANDS.find(f"/{name}"), f"/{name} is missing from the CLI")
            self.assertIsNotNone(WS_COMMANDS.find(f"/{name}"), f"/{name} is missing from the socket")

    def test_the_old_spellings_still_work_on_both_surfaces(self) -> None:
        for alias, name in (("think", "thinking"), ("thinking-depth", "thinking"),
                            ("stream", "streaming")):
            for registry in (CLI_COMMANDS, WS_COMMANDS):
                self.assertEqual(registry.find(f"/{alias} on")[0].name, name)


class DetachOwnershipTests(unittest.TestCase):
    """Who ends the server: asking to detach stops the kill at once, without waiting for
    the reply, and an attached window ends the server through the registry pid.
    """

    def _extension(self) -> str:
        path = os.path.join(_ROOT, "vscode-extension", "src", "extension.ts")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_the_outbound_request_sets_the_flag(self) -> None:
        src = self._extension()
        send = src[src.index("private _forwardToServer(payload: string): void {"):]
        send = send[:send.index("this._pendingMessages.push(payload)")]
        self.assertIn("serverDetached = true", send,
                      "the kill is only called off once the server answers")

    def _helper_lines(self, lines: list[str]) -> range:
        """The line range of the one function that ends the server."""
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("function endServerUnlessKeptRunning("))
        end = next(i for i, line in enumerate(lines[start:], start=start)
                   if line == "}")
        return range(start, end + 1)

    def test_only_one_place_ends_the_server_and_it_consults_the_flag(self) -> None:
        """A single function signals the server, and it consults the detach flag; the test
        enumerates every kill call.
        """
        lines = self._extension().splitlines()
        inside = self._helper_lines(lines)
        # Real calls only: the spawn comment mentions the same expression to explain
        # why `exec` is used, and a test that counted prose would be satisfied by it.
        sites = [
            i for i, line in enumerate(lines)
            if ("serverProcess.kill()" in line or "serverProcess?.kill()" in line
                or "process.kill(attachedPid" in line)
            and not line.lstrip().startswith(("//", "*", "/*"))
        ]
        self.assertEqual(len(sites), 2,
                         "a child and an attached pid; the kill sites moved")
        for i in sites:
            self.assertIn(i, inside,
                          f"the kill on line {i + 1} is outside the one path out:\n"
                          f"{lines[i]}")
        self.assertIn("serverDetached", "\n".join(lines[i] for i in inside),
                      "the one path out does not consult the flag")

    def test_every_way_out_of_a_window_goes_through_it(self) -> None:
        # The window closing, a connect releasing the previous server, and the panel's
        # own disconnect. Each one is a departure, and a departure that skipped the
        # helper would leave a server nobody is watching and nobody asked to keep.
        src = self._extension()
        for site, opens in (("export function deactivate(): void {", "}"),
                            ("private _teardownServer(): void {", "}"),
                            ("private _disconnect(): void {", "}")):
            body = src[src.index(site):]
            body = body[:body.index("\n  " + opens)]
            self.assertIn("endServerUnlessKeptRunning()", body,
                          f"{site.strip()} does not end the server it is leaving")

    def test_an_attached_server_is_endable_at_all(self) -> None:
        # The pid is the whole handle: a window that spawned nothing has no child to
        # kill, and without this the deliberate departures ended nothing — the run
        # outlived every one of them and was found again on the next connect.
        src = self._extension()
        attach = src[src.index("private async _attachToRunningServer("):]
        attach = attach[:attach.index("\n  /**")]
        self.assertIn("attachedPid = entry.pid", attach,
                      "attaching to a server leaves nothing able to end it")

    def test_the_server_is_spawned_in_its_own_session(self) -> None:
        # Arranging it afterwards depends on the process not already being a group
        # leader, and a detachment that quietly does not hold is worse than none.
        src = self._extension()
        spawn = src[src.index('cp.spawn("bash", ["-c", spawnCmd]'):]
        spawn = spawn[:spawn.index("});")]
        self.assertIn("detached: true", spawn)
        self.assertIn('stdio: "ignore"', spawn)

    def test_its_output_goes_to_a_file_not_a_pipe(self) -> None:
        # A pipe held by this extension host is a tether: when the host goes, the next
        # write takes a SIGPIPE and a server meant to outlive the window dies the first
        # time it logs anything.
        src = self._extension()
        self.assertIn("logPath(", src)
        self.assertIn(">> ${quotedLog} 2>&1", src)


class RoutingTests(unittest.TestCase):
    """A session command must leave the webview as `command`, not as `query`."""

    def test_the_app_routes_session_commands_apart(self) -> None:
        app = os.path.join(_ROOT, "vscode-extension", "webview", "src", "App.tsx")
        with open(app, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("isSessionCommand(text)", src)
        # The routing check must come before anything that could put the text to the
        # model, or a command typed during a run would be queued as a steer and handed
        # over as prose. Read inside the submit handler rather than across the file:
        # what sends a prompt is a function of its own — the panel asks the model too —
        # and where that function is defined says nothing about when it is reached.
        body = src[src.index("const submitQuery = useCallback"):]
        body = body[:body.index("\n  }, [")]
        self.assertIn("isSessionCommand(text)", body)
        self.assertIn("sendPrompt(text)", body)
        self.assertLess(body.index("isSessionCommand(text)"),
                        body.index("sendPrompt(text)"))
        # And the thing it defers to is the one place a prompt reaches the model.
        self.assertEqual(src.count('send({ type: "steer", text });'), 1)

    def test_the_dropdown_offers_both_kinds(self) -> None:
        with open(_SLASH_UTILS, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("filterSlashItems", src)
        self.assertIn("SESSION_COMMANDS", src)


class _Host(CommandHost):
    """A front-end with no agent: settings land in a dict, tool calls answer *payload*."""

    surface = "ws"

    def __init__(self, payload: dict | None = None) -> None:
        self.values: dict = {}
        self.payload = payload or {"status": "ok"}

    def apply(self, name, value) -> str:
        self.values[name] = value
        return ""

    def read(self, name):
        return self.values.get(name, read_setting(None, name))

    async def call_tool(self, tool, arguments) -> dict:
        return self.payload


def _run(text: str, host: CommandHost | None = None) -> CommandResult:
    return asyncio.run(WS_COMMANDS.run(host or _Host(), text))


class CommandAnswersAreRenderedTests(unittest.TestCase):
    """A command answer travels as ``command_output``, which the webview renders, not on the
    dropped ``output`` channel.
    """

    def _reply(self) -> dict:
        from mimir.client.ui.ws.connection.screen import Screen
        from mimir.client.ui.ws.connection.view import Sender, ViewState
        from mimir.tests.support.connections import FakeWS
        from mimir.tests.support.pools import FakePool
        ws = FakeWS()
        screen = Screen(ViewState(), Sender(ws), FakePool())
        asyncio.run(screen.command_reply("memory", "2 memories",
                                         items=[{"label": "a", "detail": "b"}],
                                         note="n", tone="ok"))
        return ws.sent[-1]

    def test_no_answer_uses_the_dropped_output_channel(self) -> None:
        self.assertNotEqual(self._reply()["type"], "output")

    def test_the_answer_is_structured_not_a_formatted_line(self) -> None:
        """A frontend cannot lay out what reaches it as an indented blob of text."""
        self.assertEqual(self._reply(), {
            "type": "command_output", "command": "memory", "title": "2 memories",
            "items": [{"label": "a", "detail": "b"}], "note": "n", "tone": "ok",
            "transient": False})

    def test_the_webview_renders_that_channel(self) -> None:
        reducer = os.path.join(
            _ROOT, "vscode-extension", "webview", "src", "state", "chatReducer.ts")
        with open(reducer, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('case "command_output"', src)
        # And still drops the transient one, or the chatter comes back with it.
        self.assertIn('case "output":', src)

    def test_settings_report_state_not_prose(self) -> None:
        """Every setting reports state; nothing about it is written to the chat.

        A setting has a control that shows its value, so a line about it repeats the
        chrome — and the webview replays its stored settings on connect, so those
        lines greeted each session with changes nobody had just made.
        """
        for text, kind in (("/mode plan", "mode"), ("/thinking deep", "thinking_depth"),
                           ("/thinking-depth 2", "thinking_depth"),
                           ("/streaming off", "streaming"), ("/context compact", "context_mode"),
                           ("/approvals all", "approval_mode"),
                           ("/enforcement off", "enforcement"),
                           ("/subagents build", "subagent_level")):
            with self.subTest(text=text):
                result = _run(text)
                self.assertFalse(result.card)
                self.assertEqual(result.state["type"], kind)
        self.assertFalse(_run("/batch on").card)

    def test_the_reported_value_is_the_one_held(self) -> None:
        # Echoing the requested level back is what let a wrong value be announced.
        host = _Host()
        host.apply = lambda name, value: ""   # the agent refused to move, silently
        result = _run("/thinking deep", host)
        self.assertEqual(result.state["label"], THINKING_DEPTH_LABELS[read_setting(None, "thinking_depth")])

    def test_only_listings_and_erasures_are_cards(self) -> None:
        listing = {"status": "ok", "memory": [{"name": "a", "scope": "workspace"}],
                   "proxies": [], "cleared": 2, "removed": []}
        cards = sorted(
            result.label for text in ("/memory list", "/memory clear", "/memory delete a",
                                      "/proxy list", "/proxy clean p", "/backend ray")
            if (result := _run_patched(text, listing)).card)
        self.assertEqual(cards, ["/backend", "/memory clear", "/memory delete",
                                 "/memory list", "/proxy clean", "/proxy list"])

    def test_an_invalid_value_is_an_error_not_a_change(self) -> None:
        host = _Host()
        for text in ("/mode chat", "/context huge", "/approvals sometimes",
                     "/temperature 9", "/thinking 12", "/streaming maybe"):
            with self.subTest(text=text):
                self.assertTrue(_run(text, host).failed)
        self.assertEqual(host.values, {})

    def test_the_webview_applies_that_state(self) -> None:
        hook = os.path.join(_ROOT, "vscode-extension", "webview", "src", "hooks",
                            "useAgentSettings.ts")
        with open(hook, encoding="utf-8") as fh:
            src = fh.read()
        for handler in ("thinking_depth:", "streaming:", "mode:"):
            self.assertIn(handler, src)

    def test_the_webview_has_a_component_for_it(self) -> None:
        # Rendered as a result card, not as agent prose in a bubble.
        component = os.path.join(
            _ROOT, "vscode-extension", "webview", "src", "components",
            "CommandResult.tsx")
        self.assertTrue(os.path.isfile(component), "CommandResult.tsx is missing")
        with open(component, encoding="utf-8") as fh:
            src = fh.read()
        # Both shapes exist: a line for a setting change, a card for a listing.
        self.assertIn("cmd-line", src)
        self.assertIn("cmd-card", src)


def _run_patched(text: str, payload: dict) -> CommandResult:
    from unittest import mock
    from mimir.client.session.commands import settings
    with mock.patch.object(settings, "switch_backend"):
        return _run(text, _Host(payload))

if __name__ == "__main__":
    unittest.main()


class NudgeCountTests(unittest.TestCase):
    """``/diag nudges`` reads its counts off the journals."""

    def test_reminders_are_counted_by_category_across_sessions(self) -> None:
        from mimir.client.session.journal import TranscriptLog, nudge_counts
        for session_id, categories in (("n1", ["validation", "denial", "validation"]),
                                       ("n2", ["validation"])):
            log = TranscriptLog()
            log.bind(session_id)
            for category in categories:
                log.append({"type": "nudge_injected", "category": category})
            log.append({"type": "tool_call", "name": "read_file"})
        self.assertEqual(nudge_counts(["n1", "n2"]), {"validation": 3, "denial": 1})
        self.assertEqual(nudge_counts(["n2"]), {"validation": 1})
