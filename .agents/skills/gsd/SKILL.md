---
name: gsd
description: "Master orchestrator for the Get Stuff Done (GSD) framework. Manage roadmaps, phase planning, progress tracking, and atomic milestone deliverables."
---

# Get Stuff Done (GSD) Skill

GSD is an autonomous development methodology that organizes complex software development into discrete, verified milestones and phases.

## Structure in Workspace:
- `.planning/PROJECT.md` — Project context and core goals.
- `.planning/ROADMAP.md` — Phased implementation plan with dependency waves.
- `.planning/STATE.md` — Current milestone, active phase, and completed tasks.
- `.planning/REQUIREMENTS.md` — Requirements specifications.
- `.planning/config.json` — Workflow settings.

## Slash Commands:
- `/gsd-new-project` — Initialize new project roadmap and planning directory.
- `/gsd-plan-phase <N>` — Plan an individual milestone phase into discrete tasks.
- `/gsd-execute-phase <N>` — Execute all plans within a phase.
- `/gsd-progress` — Check roadmap completion and current status.
- `/gsd-verify-work` — Interactive acceptance and UAT validation.
- `/gsd-help` — View full GSD command reference.

## Terminal CLI:
You can also run GSD commands directly from PowerShell or Command Prompt:
```powershell
gsd init        # Initialize .planning directory
gsd status      # View current phase and milestone status
gsd help        # Show GSD command guide
```
