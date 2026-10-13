"""The webview's slash list must name commands the backend actually handles.

Two kinds of slash share the chat input and travel differently: a SKILL is part of the
query and reaches the model, a SESSION COMMAND is handled by `ws_session._handle_command`
and never does. The webview's dropdown listed only skills, and typed session commands
were sent as `type: "query"` — so "/memory clear" asked the model to clear memory rather
than clearing it, and "/proxy clean foo" was a request for the model to tidy up.

The webview now carries its own list (SESSION_COMMANDS in slashUtils.ts), mirrored by
hand so the dropdown needs no round-trip before the first keystroke completes. Mirrored
lists drift; this test is what stops the drift from becoming a command that autocompletes
and then does nothing.
"""
from __future__ import annotations

import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SLASH_UTILS = os.path.join(
    _ROOT, "vscode-extension", "webview", "src", "components", "slashUtils.ts")
_WS_SESSION = os.path.join(_ROOT, "client", "ui", "ws", "ws_session.py")
_CLI_COMMANDS = os.path.join(_ROOT, "client", "ui", "cli", "chat_commands.py")


def _advertised() -> list[str]:
    """Command names the webview offers in its dropdown."""
    with open(_SLASH_UTILS, encoding="utf-8") as fh:
        src = fh.read()
    block = src.split("SESSION_COMMANDS", 1)[1].split("];", 1)[0]
    return re.findall(r'name:\s*"([^"]+)"', block)


def _ws_handled() -> str:
    with open(_WS_SESSION, encoding="utf-8") as fh:
        return fh.read()


def _handler_body() -> str:
    """Just the command handler, so a match elsewhere in the file cannot stand in."""
    src = _ws_handled()
    start = src.index("async def _handle_command(self")
    end = src.index("async def _send_toggles", start)
    return src[start:end]


class AdvertisedCommandsExistTests(unittest.TestCase):
    def test_the_list_is_not_empty(self) -> None:
        # A parser that silently matched nothing would make every assertion below vacuous.
        self.assertGreater(len(_advertised()), 5)

    def test_every_advertised_command_is_handled_by_the_session(self) -> None:
        handler = _ws_handled()
        missing = [
            name for name in _advertised()
            if f'"/{name} "' not in handler and f'"/{name}"' not in handler
        ]
        self.assertEqual(
            missing, [],
            "the webview offers these but ws_session._handle_command does not handle "
            f"them — they would autocomplete and do nothing: {missing}")

    def test_every_handled_command_is_advertised(self) -> None:
        """The converse of the check above, and the sharper of the two.

        A name advertised but unhandled autocompletes and does nothing. A command the
        session handles but the webview does not know is worse: it is not routed as a
        command at all. It leaves as a plain query, the *model* answers it, and that
        reads as the command working badly rather than never arriving — a `/diag` that
        reports the machine's CPU instead of the event chain's counters.
        """
        # Every ``"/name"`` literal in the handler, rather than the syntactic forms a
        # branch can take. The first version of this test enumerated `==` and
        # `.startswith(`, and missed `text in ("/diag", "/diagnose")` — so it passed
        # while the very command that prompted it was still unrouted.
        handler = _handler_body()
        handled = {name.rstrip() for name in re.findall(r'"/([a-z][a-z-]*)\s?"', handler)}
        # Answered from the model's own surface rather than the panel's: the webview has
        # no business knowing these.
        model_facing = {"backend"}
        advertised = set(_advertised())
        unrouted = sorted(handled - advertised - model_facing)
        self.assertEqual(
            unrouted, [],
            "ws_session handles these but the webview does not route them as commands, "
            f"so they reach the model as plain queries instead: {unrouted}")

    def test_the_two_commands_this_work_added_are_both_there(self) -> None:
        advertised = _advertised()
        self.assertIn("memory", advertised)
        self.assertIn("proxy", advertised)

    def test_they_are_handled_on_the_cli_surface_too(self) -> None:
        # The CLI has its own table; a command added to one surface and not the other is
        # a command that works in one window and not the next.
        with open(_CLI_COMMANDS, encoding="utf-8") as fh:
            cli = fh.read()
        for name in ("memory", "proxy"):
            self.assertIn(f'cmd == "/{name}"', cli, f"/{name} is missing from the CLI")


class DetachOwnershipTests(unittest.TestCase):
    """Who ends the server, and how the host knows it is not its to end.

    Asking to detach is what stops the kill, not hearing back about it: there is a round
    trip between the click and the server having detached, and a host that waits for the
    reply leaves a window in which the user has asked to keep the run going, the badge
    has not changed yet, and closing the editor still ends it. The two outcomes are not
    symmetric — a server left running that failed to detach is findable through the
    registry and endable with the stop command, while one killed after the user asked to
    keep it is work gone.

    The other half is having a handle at all. A window that attached to a server holds no
    child process, so the pid from the registry entry stands in for one; without it the
    departures that are supposed to end a server ended nothing.
    """

    def _extension(self) -> str:
        path = os.path.join(_ROOT, "vscode-extension", "src", "extension.ts")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_the_outbound_request_sets_the_flag(self) -> None:
        src = self._extension()
        send = src[src.index('if (m.type === "ws_send")'):]
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
        """Found by enumerating the calls rather than the ones I knew.

        A guard on some of the paths out is a detachment that holds until the user
        leaves through another one. So there is a single function that signals a
        server — a spawned child and an attached pid alike — and the enumeration is
        the test: a kill written anywhere else fails here rather than in somebody's
        overnight run.
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
                            ('if (m.type === "disconnect") {', "}")):
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
        # The routing check must come BEFORE the busy/steer branch, or a command typed
        # during a run would be queued as a steer and handed to the model.
        self.assertLess(src.index("isSessionCommand(text)"), src.index('type: "steer"'))

    def test_the_dropdown_offers_both_kinds(self) -> None:
        with open(_SLASH_UTILS, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("filterSlashItems", src)
        self.assertIn("SESSION_COMMANDS", src)


class CommandAnswersAreRenderedTests(unittest.TestCase):
    """A command answer must not travel on the transient `output` channel.

    The webview reducer drops `output` on purpose — it is tool-activity chatter — so
    every answer `_handle_command` sent was swallowed: `/memory list` printed nothing,
    and `/memory clear` wiped the store while looking like it had done nothing at all.
    Answers go out as `command_output`, which the reducer renders.
    """

    def _handler_body(self) -> str:
        return _handler_body()

    def test_the_handler_body_is_found(self) -> None:
        # A slice that silently matched nothing would make the assertions below vacuous.
        body = self._handler_body()
        self.assertIn("/memory", body)
        self.assertIn("/proxy", body)

    def test_no_answer_uses_the_dropped_output_channel(self) -> None:
        self.assertNotIn('"type": "output"', self._handler_body())

    def test_answers_use_the_rendered_channel(self) -> None:
        # Answers go through the one helper, which is what puts them on the channel;
        # the handler itself no longer builds a payload by hand.
        self.assertIn("_command_reply(", self._handler_body())
        src = _ws_handled()
        helper = src[src.index("async def _command_reply("):src.index("async def _handle_command(")]
        self.assertIn('"command_output"', helper)

    def test_the_answer_is_structured_not_a_formatted_line(self) -> None:
        """A frontend cannot lay out what reaches it as an indented blob of text."""
        src = _ws_handled()
        helper = src[src.index("async def _command_reply("):src.index("async def _handle_command(")]
        for field in ('"command"', '"title"', '"items"', '"note"', '"tone"'):
            self.assertIn(field, helper)

    def test_the_webview_renders_that_channel(self) -> None:
        reducer = os.path.join(
            _ROOT, "vscode-extension", "webview", "src", "state", "chatReducer.ts")
        with open(reducer, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('case "command_output"', src)
        # And still drops the transient one, or the chatter comes back with it.
        self.assertIn('case "output":', src)

    def test_only_listings_and_erasures_reach_the_transcript(self) -> None:
        """Every setting reports state; nothing about it is written to the chat.

        A setting has a control that shows its value, so a line about it repeats the
        chrome — and the webview replays its stored settings on connect, so those
        lines greeted each session with changes nobody had just made.
        """
        body = self._handler_body()
        replies = re.findall(r'_command_reply\(\s*"([^"]+)"', body)
        self.assertEqual(
            sorted(replies),
            [
                # The one setting that still speaks: it has no control anywhere, and
                # it is not in SESSION_COMMANDS, so the webview never routes it here
                # at all. Switching the LLM in silence would be the worse trade.
                "/backend",
                # Not a setting — an action whose result has nowhere else to appear.
                "/cancel",
                # A listing, and the one place the event chain's own counters are
                # readable. It owns no control, and the question it answers — which
                # link went quiet — has no other surface.
                "/diag",
                "/memory clear", "/memory delete", "/memory list",
                "/proxy clean", "/proxy list",
            ],
            f"a setting is narrating again: {sorted(replies)}",
        )

    def test_settings_that_own_a_control_report_state_not_prose(self) -> None:
        """Thinking and mode have chrome; a transcript line only repeats it.

        The webview replays its stored settings on every connect, so those lines
        greeted each session with a change nobody had just made — and, because the
        replay asserts the panel's own value rather than the agent's, sometimes
        announced a depth the agent was not on.
        """
        body = self._handler_body()
        mode_branch = body[body.index('if text.startswith("/mode ")'):
                           body.index('elif text.startswith("/batch ")')]
        self.assertNotIn("_command_reply", mode_branch)
        self.assertIn('"type": "mode"', mode_branch)

        thinking = body[body.index('elif text.startswith("/thinking ")'):
                        body.index('elif text.startswith("/streaming ")')]
        self.assertNotIn("_command_reply", thinking)
        self.assertEqual(thinking.count("_send_thinking_state()"), 2)  # /thinking + depth

        streaming = body[body.index('elif text.startswith("/streaming ")'):
                         body.index('elif text.startswith("/context ")')]
        self.assertNotIn("_command_reply", streaming)
        self.assertIn("_send_streaming_state()", streaming)

    def test_the_reported_depth_is_the_agents_own(self) -> None:
        # Echoing the requested level back is what let a wrong value be announced.
        src = _ws_handled()
        helper = src[src.index("async def _send_thinking_state("):
                     src.index("async def _handle_command(")]
        self.assertIn("thinking_depth", helper)
        self.assertIn('getattr(agent, "thinking_depth"', helper)

    def test_the_webview_applies_that_state(self) -> None:
        app = os.path.join(_ROOT, "vscode-extension", "webview", "src", "App.tsx")
        with open(app, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('case "thinking_depth"', src)
        self.assertIn('case "streaming"', src)
        self.assertIn('case "mode"', src)

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


if __name__ == "__main__":
    unittest.main()
