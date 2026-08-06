# Release tooling

The working repo (`lab`, private) holds the simulator **and** the research
lines.  The public repo holds the simulator only.

## The rule

`origin` is **public**.  It is updated by *export*, never by pushing a branch
from here.  `scripts/release/pre-push` enforces that: it refuses any push to
the public remote that carries `research/`, `tasks/`, `data/` or `logs/`.

Hooks are not versioned by git, so after cloning (or `git worktree add`):

```bash
bash scripts/release/install_hooks.sh
```

## Publishing the backbone

```bash
bash scripts/release/publish.sh                       # review the manifest
bash scripts/release/publish.sh --write /tmp/sf-pub   # stage into a clone
git -C /tmp/sf-pub diff --cached --stat               # read it before pushing
bash scripts/release/publish.sh --write /tmp/sf-pub --push
```

`publish.sh` works from an **allowlist**, never a denylist.  A denylist leaks
the first time someone adds a directory and forgets to exclude it; an allowlist
fails closed.  It aborts outright if the manifest ever resolves to a private
path.

`CLAUDE.md` is deliberately not exported — it is internal working instruction
and it documents the research lines.

## Remotes

| remote | visibility | holds |
|---|---|---|
| `lab` | private | everything; `main` tracks this |
| `origin` | public | backbone release only |
