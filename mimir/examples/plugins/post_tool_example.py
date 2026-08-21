"""Example post-tool hook: run the project's own checks after the agent edits code.

Drop into ``.mimir/plugins/``. This is the seam for evidence the *machine* observes,
next to the verdict the model states about itself — a red suite recorded here makes the
turn-end gates refuse the conclusion, with no blocking mechanism of its own.

Keep a hook cheap: it runs after every matching tool call, inside a 60 s budget, and its
cost is paid on every edit. Filter hard, and prefer the narrowest command that answers
the question.
"""

from mimir.client.extensions import PostToolRule, register_post_tool
from mimir.client.context.capabilities import EDIT, has_cap


async def _project_checks(agent, tool_name, arguments, result, execution_context):
    if not has_cap(tool_name, EDIT, getattr(agent, "tool_caps", {})):
        return ""                                  # not an edit → abstain
    if not str(arguments.get("path", "")).endswith(".py"):
        return ""
    # Goes through the ordinary tool path, so the approval gate still applies: an
    # "always" grant on the command prefix covers the rest of the session.
    out = await agent._run_tool(
        "bash_run", {"command": "pytest -q"}, execution_context=execution_context,
    )
    if '"status": "ok"' in out:
        return ""
    return (
        "\n\nPROJECT_CHECK: the project's test suite is red after this edit. "
        "Read the failure and fix it before moving on."
    )


def register():
    register_post_tool(PostToolRule(name="project_checks", run=_project_checks))


register()
