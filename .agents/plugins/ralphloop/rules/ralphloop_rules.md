# Ralph Loop Execution Standards

1. **Task Boundary Enforcement:** Never execute multiple tasks in a single cycle. Follow the single-task constraint strictly.
2. **Preflight Verification:** Ensure test harnesses (`pytest`, `npm test`) are passing before and after each task commit.
3. **Artifact Logging:** Record completed task notes in `.agent/logs/LOG.md`.
