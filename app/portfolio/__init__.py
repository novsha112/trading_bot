"""Portfolio rules: how confirmed executions change local positions.

Stateless and pure; the state itself lives in the single account-state owner
(``app.execution.account_state``), so there is one lock and one revision.
"""
