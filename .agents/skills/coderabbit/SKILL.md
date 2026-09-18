---
name: coderabbit
description: "Run and inspect CodeRabbit AI code reviews, view actionable review findings, and validate code quality against review standards."
---

# CodeRabbit Skill

Use this skill whenever you need to review code, run CodeRabbit locally, or inspect review findings.

## CLI Usage

CodeRabbit is installed locally at:
`C:\Users\Asus\AppData\Local\Programs\coderabbit\coderabbit.exe`

### Common Commands:

1. **Review Local Working Changes:**
   ```powershell
   & "C:\Users\Asus\AppData\Local\Programs\coderabbit\coderabbit.exe" review
   ```

2. **Show Findings from Previous Local Review:**
   ```powershell
   & "C:\Users\Asus\AppData\Local\Programs\coderabbit\coderabbit.exe" review findings
   ```

3. **Show Review Prompts for AI Agents:**
   ```powershell
   & "C:\Users\Asus\AppData\Local\Programs\coderabbit\coderabbit.exe" review --show-prompts
   ```

4. **Review Only Uncommitted Changes:**
   ```powershell
   & "C:\Users\Asus\AppData\Local\Programs\coderabbit\coderabbit.exe" review --uncommitted
   ```

5. **Review Only Committed Changes Against Main:**
   ```powershell
   & "C:\Users\Asus\AppData\Local\Programs\coderabbit\coderabbit.exe" review --committed --base main
   ```
