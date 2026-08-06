#!/usr/bin/env bash
#
# Export the backbone to the PUBLIC repo.
#
# The working repo (remote `lab`, private) holds the simulator AND the research
# lines. The public repo gets the simulator only. This script does that export.
#
# It works from an ALLOWLIST, never a denylist. A denylist leaks the first time
# someone adds a directory and forgets to exclude it; an allowlist fails closed,
# which is the direction you want a mistake to fail in.
#
# Dry run by default -- it prints the manifest and does not touch the remote.
#
#   bash scripts/release/publish.sh                 # show what would be exported
#   bash scripts/release/publish.sh --write DIR     # stage the export into DIR
#   bash scripts/release/publish.sh --write DIR --push
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2

PUBLIC_REMOTE="${PUBLIC_REMOTE:-git@github.com:SmallWorldLab/SceneFactory.git}"
PUBLIC_BRANCH="${PUBLIC_BRANCH:-main}"

# --- the allowlist -----------------------------------------------------------
# Paths are git pathspecs, matched against tracked files only.
ALLOW=(
  'README.md' 'RELEASE.md' 'LICENSE' 'SF_FORMAT.md'
  'requirements.txt' 'quickstart.sh' '.gitignore' 'pyproject.toml'
  'docs/**'
  'src/**'
  'scripts/**'
  'configs/**'
  # The scene pools the documented commands resolve to. These live under
  # generated/ (which is otherwise excluded, see DENY_REGEX) but they ARE the
  # train and eval splits -- each lists its scene files by name -- so shipping
  # them is what makes a split reproducible rather than merely described.
  'configs/scene_factory/generated/scene_factory_256scene_0414_*.yaml'
  'configs/scene_factory/generated/eval_unseen_199scenes_*.yaml'
  'configs/scene_factory/generated/eval_scenediv_*.yaml'
  'checkpoints/**'
  'artifacts/student_vehicle_assets/**'
  'artifacts/student_vehicle_sysid/**'
  # Referenced by src/student_vehicle_multiagent_goal_env.py as the visual proxy.
  'artifacts/low_poly_car_proxy.usd'
  'run_*.sh'
)
# CLAUDE.md is deliberately NOT exported: it is internal working instruction and
# it documents the research lines.
# Removed from the allowlist result. These are paths that MATCH an allow rule
# but must not ship: research-specific configs and runners that live at the repo
# root for historical reasons, and internal tooling.
#
# configs/scene_factory/generated/ is excluded for a different reason: it is a
# scratch directory of machine-generated per-experiment configs, most of them
# written against scene pools that no longer exist, so shipping it wholesale
# hands users 82 configs of which a handful run.
DENY_REGEX='(workzone|cone_finetune|scene226)|^scripts/check_restructure\.sh$|^configs/scene_factory/generated/|^scripts/viz_design_vs_score\.py$'
# Superseded runners and the policies they load. Each resolves an eval config
# that is no longer generated, so every one of them fails at its first file
# check -- and the results they were written to produce have since been redone
# multi-seed by run_paper_table4_eval.sh and run_transfer_eval.sh. Shipping a
# script that cannot run is worse than not shipping it.
DENY_REGEX="$DENY_REGEX"'|^run_v8_(vs_v7_(physics_blind|moderate_wet|heavy_wet)|physx_to_bicycle_transfer)_eval\.sh$'
DENY_REGEX="$DENY_REGEX"'|^run_bicycle_(eval|physx_transfer_eval)\.sh$'
DENY_REGEX="$DENY_REGEX"'|^run_cinematic_demo\.sh$|^scripts/summarize_2x2_eval\.py$'
DENY_REGEX="$DENY_REGEX"'|^checkpoints/v[78]_'
# Eval configs on the retired 64-scene split. That split is not disjoint from the
# current 256-scene training pool -- 41 scenes overlap -- so results from it are
# not held-out results, and the pools they resolve are no longer generated.
DENY_REGEX="$DENY_REGEX"'|^configs/scene_factory/eval_knn_scratch_64scenes_dry\.yaml$'
DENY_REGEX="$DENY_REGEX"'|^configs/scene_factory/eval_(conditioned_16a_friction_matching(_strict)?|waymo_physx_256_friction_groups)\.yaml$'
DENY_REGEX="$DENY_REGEX"'|^configs/scene_factory/eval_(waymo_physx_noweather_16agents_friction_groups|noweather_16a_friction_(seeing|blind|blind_strict))\.yaml$'
DENY_REGEX="$DENY_REGEX"'|^run_eval_friction_sweep\.sh$'
# Two harnesses that exit 0 without producing a valid result, verified on a clean
# checkout 2026-08-06. Both are open bugs in the working repo, not export damage.
#
#   run_braking_validation.sh   -- the 9-world v0-injection variant. Vehicles never
#     reach the 27.78 m/s entry speed, so it prints a full table of zeros. The
#     NHTSA numbers come from run_brake_sweep_replicates.sh, which does work and
#     which the docs now point at.
#   run_physics_validation.sh   -- Phase 1 fails `grounded(z<0.15)`: at full
#     throttle the chassis rides up to z=0.219 m. run_physics_sanity_test.sh and
#     run_traction_probe.sh both pass and cover the same ground.
DENY_REGEX="$DENY_REGEX"'|^run_(braking|physics)_validation\.sh$'

# Exceptions to DENY_REGEX -- paths that match a deny rule but must ship anyway.
# The train and eval scene pools live under generated/ and every documented
# config resolves to one of them through scene_factory.config_path, so denying
# the whole directory silently breaks every command in the README. They are also
# the splits themselves: each pool lists its scene files by name.
#
# KEEP is checked BEFORE DENY, so keep these patterns narrow. The private-path
# guard below still runs regardless, so a mistake here cannot leak research/,
# tasks/, data/ or logs/.
KEEP_REGEX='^configs/scene_factory/generated/(scene_factory_256scene_0414_|scene_factory_256scene_random_|eval_unseen_199scenes_|eval_scenediv_)'

manifest() {
  for pat in "${ALLOW[@]}"; do
    git ls-files -- "$pat"
  done | sort -u | awk -v keep="$KEEP_REGEX" -v deny="$DENY_REGEX" \
    '$0 ~ keep || $0 !~ deny'
}

FILES=$(manifest)
COUNT=$(printf '%s\n' "$FILES" | grep -c . || true)

# --- safety: nothing private may appear -------------------------------------
LEAKS=$(printf '%s\n' "$FILES" | grep -E '^(research|tasks|data|logs)/' || true)
if [[ -n "$LEAKS" ]]; then
  echo "ABORT: allowlist produced private paths:" >&2
  printf '%s\n' "$LEAKS" >&2
  exit 1
fi

MODE="dry"; OUT=""; DO_PUSH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --write) MODE="write"; OUT="${2:?--write needs a directory}"; shift 2 ;;
    --push)  DO_PUSH=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "public remote : $PUBLIC_REMOTE ($PUBLIC_BRANCH)"
echo "files exported: $COUNT"
printf '%s\n' "$FILES" | awk -F/ '{print $1}' | sort | uniq -c | sort -rn | sed 's/^/  /'
echo
echo "excluded by DENY_REGEX: $(git ls-files | grep -Ec "$DENY_REGEX" || true) tracked files"
echo "research/ tasks/ data/ present in export: NO (verified above)"

if [[ "$MODE" == "dry" ]]; then
  echo
  echo "dry run -- nothing written. Re-run with --write <dir> to stage."
  exit 0
fi

mkdir -p "$OUT"
if [[ ! -d "$OUT/.git" ]]; then
  git clone --depth 50 --branch "$PUBLIC_BRANCH" "$PUBLIC_REMOTE" "$OUT" || exit 1
fi
# Wipe tracked content so a file deleted here is deleted there too, then re-copy.
( cd "$OUT" && git rm -rq --ignore-unmatch . )
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  mkdir -p "$OUT/$(dirname "$f")"
  cp -p "$f" "$OUT/$f"
done <<< "$FILES"

# -f matters. The repo's own .gitignore lists `artifacts/`, which are tracked here
# only because they were force-added. Without -f, `git add` silently skips them
# and they stage as DELETIONS -- a push would strip the vehicle USD/URDF and the
# sysid config from the public repo, leaving code that cannot find its car.
( cd "$OUT" && git add -Af . && git status --short | head -20 && echo "..." )

# Fail closed: every manifest file must actually be staged. The failure above was
# silent precisely because nothing checked this.
# Compare against the INDEX, not against the diff: a file identical to the public
# copy cancels its own staged deletion and shows up in no diff at all, so a diff
# check would flag every unchanged file.
MISSING=$(
  cd "$OUT" || exit
  indexed=$(git ls-files --cached | sort -u)
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    grep -qxF "$f" <<< "$indexed" || echo "$f"
  done <<< "$FILES"
)
if [[ -n "$MISSING" ]]; then
  echo "ABORT: manifest files were not staged (check .gitignore in the export):" >&2
  printf '  %s\n' $MISSING >&2
  exit 1
fi

echo
echo "staged in $OUT. Review with: git -C $OUT diff --cached --stat"
[[ $DO_PUSH -eq 1 ]] || { echo "not pushing (pass --push)"; exit 0; }
( cd "$OUT" && git commit -q -m "release: export backbone from the working repo" \
  && git push origin "$PUBLIC_BRANCH" )
