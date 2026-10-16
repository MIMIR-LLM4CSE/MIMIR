"""Housekeeping the user does without asking the model: memory, proxies, modules.

Each is one direct tool call (``CommandHost.call_tool``), around the guardrails: the
user invoked it themselves, and the approval and observation machinery exists to judge
what the *model* asked for.
"""
from __future__ import annotations

from .base import Command, CommandHost, CommandResult, Item


def _failure(payload: dict, what: str) -> CommandResult | None:
    """The error to report, if the call failed.

    Read as an empty listing, "no agent is running" or a tool error said "No memories
    stored" — of a store that may well be full.
    """
    if payload.get("status") in (None, "ok"):
        return None
    return CommandResult.error(payload.get("error") or f"{what} failed")


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


class MemoryCommand(Command):
    """The memory tools existed, but only the model could reach them, so "forget what
    you learned about this project" was a request rather than an action — and a stale
    memory keeps being recalled into every prompt until somebody removes it."""
    name = "memory"
    help = (("list", "inspect persistent memory"),
            ("clear", "wipe this workspace's memory (global memory is untouched)"),
            ("delete <name>", "remove one memory"))

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        sub = args[0] if args else ""
        if sub == "list":
            payload = await host.call_tool("memory_list_all", {"scope": "all"})
            if (failed := _failure(payload, "listing")):
                return failed
            entries = payload.get("memory") or []
            return CommandResult(
                _plural(len(entries), "memory", "memories") if entries
                else "No memories stored",
                # The scope rides on each label: a fact reaching every workspace and
                # one reaching only this project look identical without it.
                [Item(f"[{e.get('scope', '?')}] {e.get('name', '?')}",
                      (e.get("description") or "").strip()) for e in entries],
                tone="ok" if entries else "empty", label="/memory list")
        if sub == "delete" and len(args) >= 2:
            payload = await host.call_tool("memory_delete", {"name": args[1]})
            if (failed := _failure(payload, "delete")):
                return failed
            scope = payload.get("scope", "")
            return CommandResult(f"Deleted 1 {scope} memory".replace("  ", " "),
                                 [Item(args[1])], tone="warn", label="/memory delete")
        if sub == "clear":
            # Workspace only, and named as such: the global memory is every other
            # workspace's too, so this command does not wipe it on a bare word.
            # Irreversible, and deliberately typed in full by the person whose memory
            # it is; the count is reported, since a wipe that says nothing reads as a
            # wipe that did not happen.
            payload = await host.call_tool("memory_clear", {"scope": "workspace"})
            if (failed := _failure(payload, "clear")):
                return failed
            n = payload.get("cleared", 0)
            return CommandResult(
                f"Cleared {_plural(n, 'workspace memory', 'workspace memories')}",
                note="This cannot be undone. Global memory is untouched.",
                tone="warn", label="/memory clear")
        return self.usage()


class ProxyCommand(Command):
    """Without these the only way to remove a proxy's runs and optimisation state was
    to delete a store directory whose path nobody has a reason to know — which is how a
    deleted project came back with a finished checklist and an optimisation still
    marked "in progress"."""
    name = "proxy"
    help = (("list", "name the registered proxies"),
            ("clean <name>", "delete a proxy's runs, optimisation state and snapshots; "
                             "reports what it left behind"))

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        sub = args[0] if args else ""
        if sub == "list":
            # "clean <name>" needs a name, and the only way to learn one was to ask the
            # model to list them — an LLM round trip to read a registry file.
            payload = await host.call_tool("proxy_get", {"op": "proxies"})
            if (failed := _failure(payload, "list")):
                return failed
            entries = payload.get("proxies") or []
            return CommandResult(
                f"{_plural(len(entries), 'registered proxy', 'registered proxies')}"
                if entries else "No proxies registered",
                [Item(e.get("name", "?"), (e.get("description") or "").strip())
                 for e in entries],
                tone="ok" if entries else "empty", label="/proxy list")
        if sub == "clean" and len(args) >= 2:
            name = args[1]
            # This one DELETES, so it goes through the tool rather than around it:
            # `clean` is the op that knows what belongs to a proxy and what does not —
            # sealed references and suites are shared, and reported rather than removed.
            payload = await host.call_tool(
                "proxy_manage", {"op": "clean", "name": name, "confirm": True})
            if (failed := _failure(payload, "clean")):
                return failed
            # What survived is the part worth showing: it answers "I deleted everything
            # and it still remembers".
            items = [Item("removed", ", ".join(payload.get("removed") or ["nothing"]))]
            items += [Item("kept", k) for k in (payload.get("kept") or [])]
            return CommandResult(f"Cleaned proxy '{name}'", items, tone="warn",
                                 label="/proxy clean")
        return self.usage()


_NO_CATALOGUE = ("The platform server is not connected, so there is no module "
                 "catalogue. Enable it with: /servers on platform")


def _catalogue_status(payload: dict) -> CommandResult:
    if not payload.get("indexed"):
        return CommandResult(
            "Module catalogue: not indexed",
            [Item("module system", str(payload.get("module_system", "?")))],
            note=payload.get("note", ""), tone="empty")
    digest = payload.get("digest") or {}
    items = [
        Item("host", str(payload.get("hostname", "?"))),
        Item("module system", str(payload.get("module_system", "?"))),
        Item("entries", str(payload.get("count", 0))),
        Item("detail level", f"{payload.get('tier', '?')}"
             + (" (partial)" if payload.get("partial") else "")),
        Item("built", str(payload.get("built_at", "?"))),
        Item("up to date", "yes" if payload.get("signal_fresh")
             else "no — rebuilds on next search"),
    ]
    if payload.get("enriching"):
        items.append(Item("enriching", "in progress"))
    node_types = digest.get("node_types") or []
    if node_types:
        items.append(Item("node types", f"{len(node_types)} distinct hardware signatures"))
    return CommandResult("Module catalogue", items,
                         note="/modules refresh to rebuild, /modules <term> to search.")


def _module_hits(payload: dict) -> CommandResult:
    modules = payload.get("modules") or []
    if not modules:
        return CommandResult("No module matched", tone="empty")
    items = [Item(entry.get("load", "") + (" (default)" if entry.get("default") else ""),
                  (entry.get("description") or "").strip()) for entry in modules]
    note = "Load one with: module load <name>"
    if (payload.get("catalogue") or {}).get("enriching"):
        note = "Descriptions are still being indexed in the background. " + note
    return CommandResult(
        f"{len(modules)} of {payload.get('total_indexed', '?')} indexed modules",
        items, note=note)


def _catalogue_failure(payload: dict) -> CommandResult | None:
    failed = _failure(payload, "catalogue")
    if failed is not None and "not connected" in failed.title:
        return CommandResult.error(_NO_CATALOGUE)
    return failed


class ModulesCommand(Command):
    name = "modules"
    help = (("", "environment-module catalogue: index status"),
            ("refresh", "force a rebuild of the catalogue"),
            ("<term>", "search the catalogue"))

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if not args:
            payload = await host.call_tool("platform_catalogue_status", {})
            return _catalogue_failure(payload) or _catalogue_status(payload)
        if args[0].lower() == "refresh":
            payload = await host.call_tool(
                "platform_search", {"query": "*", "limit": 1, "refresh": True})
            if (failed := _catalogue_failure(payload)):
                return failed
            catalogue = payload.get("catalogue") or {}
            return CommandResult(
                f"Rebuilt: {payload.get('total_indexed', 0)} modules indexed "
                f"(detail level: {catalogue.get('tier', '?')})",
                note=("Descriptions are being filled in in the background."
                      if catalogue.get("enriching") else ""))
        payload = await host.call_tool(
            "platform_search", {"query": " ".join(args), "limit": 20})
        if (failed := _catalogue_failure(payload)):
            return failed
        if payload.get("module_system") == "none":
            return CommandResult(payload.get("note", "No module system on this host."),
                                 tone="empty")
        return _module_hits(payload)


HOUSEKEEPING_COMMANDS: tuple[Command, ...] = (MemoryCommand(), ProxyCommand(), ModulesCommand())
