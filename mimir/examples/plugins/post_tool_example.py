"""Example post-tool hook: run the project's own checks after the agent edits code.

Copy into ``.mimir/plugins/``. Keep a hook cheap: it runs after every matching call,
within 60 s.
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
