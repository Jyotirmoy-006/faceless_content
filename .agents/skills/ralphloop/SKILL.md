---
name: ralphloop
description: "Run, monitor, and configure the Ralph Wiggum autonomous task execution loop (ralph.sh), working through PRD tasks in isolated sandboxes."
---

# Ralph Loop Skill

Ralph Wiggum is an autonomous, long-running agent loop that drives implementation tasks from `.agent/prd/PRD.md` to completion with commit-per-task guarantees.

## Invocation

### Via Bash / WSL / Git Bash:
```bash
./ralph.sh --help
./ralph.sh --once
./ralph.sh --max-iterations 5
```

### Key Configuration Files:
- Task Backlog: `.agent/prd/PRD.md` or `.agent/tasks.json`
- Progress & Activity Log: `.agent/logs/LOG.md`
- Run History: `.agent/history/`

## Ralph Workflow Rules

1. **One Task Per Invocation:** Complete exactly one task from the backlog, verify it thoroughly, commit with Conventional Commits, and pause.
2. **Reuse Before Creating:** Search existing modules before creating new utilities.
3. **File Size Discipline:** Keep modules between 200–300 lines max.
