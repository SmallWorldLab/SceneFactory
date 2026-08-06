#!/usr/bin/env bash
# Install the repo's git hooks. Hooks are not versioned by git itself, so this
# has to be run once per clone (and after `git worktree add`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
HOOKS="$(git rev-parse --git-path hooks)"
install -m 0755 scripts/release/pre-push "$HOOKS/pre-push"
echo "installed $HOOKS/pre-push"
