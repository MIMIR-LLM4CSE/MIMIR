"""How long a question put to the user stays open before the agent is told to decide for
itself: one number for the server that asks (``interaction``) and the client that waits."""

# Human waits are outside the tool budget, so this is their only wall. Not applied to
# approvals (expiring into "yes" would authorise unseen commands) nor to plan approvals
# (already resolved to "execute nothing").
USER_QUESTION_TIMEOUT_SECS: int = 300
