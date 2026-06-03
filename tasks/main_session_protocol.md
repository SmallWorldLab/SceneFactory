# Main Session Protocol

The main session PLANS and COORDINATES. It does NOT implement.
Rule: if you're about to read more than 30 lines of code, spawn an agent instead.

---

## How to start a main session

Say this at the start:
> "Read your memory files (see MEMORY.md) and CLAUDE.md. I'm working on SceneFactory workzone.
>  In 3 bullets: what's the current state? Then propose the next task from WORKZONE.md."

---

## How to delegate a task

Say this when you have a task brief:
> "Spawn an Agent to execute tasks/[task_file].md.
>  Ask it to report back: files changed (with line numbers), test command run, pass/fail result, and any blockers."

The Agent tool will run in a fresh context with only the task brief. It will:
1. Read the relevant files mentioned in the brief
2. Make the change
3. Run the acceptance test
4. Report back concisely

---

## How to review and continue

After the agent reports:
> "Update memory with what was found and what changed. What's the next task on the list?"

The main session updates memory (keeps decisions durable), then plans the next task.

---

## Rules for staying lean

| DO | DON'T |
|---|---|
| Ask the agent to read files | Read files yourself in main session |
| Write task briefs before delegating | Debug inline in main session |
| Update memory after each task | Re-explain context to every agent |
| Keep tasks to one acceptance criterion | Bundle multiple unrelated changes |
| End session when task list is empty | Keep session open "just in case" |

---

## Task brief quality checklist

Before spawning an agent, verify the brief has:
- [ ] Problem in ≤2 sentences (symptom + suspected cause)  
- [ ] Relevant files with line number hints  
- [ ] "Do NOT touch" list  
- [ ] Acceptance criterion = one runnable command + expected output  
- [ ] "Already tried" list (prevents repeated dead ends)

---

## Memory update after each task

After an agent finishes, tell the main session:
> "Save a memory entry: [task name] was completed on [date]. What changed: [brief]. Result: [pass/fail]. Key learning: [1 sentence]."

---

## Current task queue (ordered)

1. `tasks/fix_physx_drive.md` — Fix PhysX suspension spring → vehicle reaches 5+ m/s
2. `tasks/workzone_training.md` (not yet written) — Run workzone training, verify convergence
3. `tasks/workzone_eval.md` (not yet written) — Evaluate trained policy on workzone scenes
4. Then: workzone optimization (CEM/safe RL), per-workzone-config scoring
