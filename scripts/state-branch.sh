#!/usr/bin/env bash
# Keeps Stoa's durable state on a dedicated branch named stoa-state.
#   state-branch.sh checkout <dir>          check the branch out into <dir>, creating it if needed
#   state-branch.sh commit <dir> <message>  commit and push changes under <dir>/state,
#                                           <dir>/live-state and <dir>/shared
set -euo pipefail

BRANCH="stoa-state"
cmd="${1:?usage: state-branch.sh checkout|commit <dir> [message]}"
dir="${2:?missing directory}"

identity() {
  git -C "$1" config user.name "AddisonTech"
  git -C "$1" config user.email "stalor54@yahoo.com"
}

case "$cmd" in
  checkout)
    if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
      git fetch --depth=1 origin "$BRANCH"
      git worktree add "$dir" FETCH_HEAD
      git -C "$dir" checkout -B "$BRANCH"
    else
      git worktree add --detach "$dir"
      git -C "$dir" checkout --orphan "$BRANCH"
      git -C "$dir" rm -rf --quiet . || true
      printf "Stoa state. Written by the daily workflow. Do not edit while a job runs.\n" > "$dir/README.md"
      identity "$dir"
      git -C "$dir" add README.md
      git -C "$dir" commit --quiet -m "Start Stoa state branch"
    fi
    mkdir -p "$dir/state" "$dir/shared"
    ;;
  commit)
    msg="${3:-update state}"
    identity "$dir"
    for sub in state live-state shared; do
      if [ -e "$dir/$sub" ]; then git -C "$dir" add -A "$sub"; fi
    done
    if git -C "$dir" diff --cached --quiet; then
      echo "state unchanged"
      exit 0
    fi
    git -C "$dir" commit --quiet -m "State: $msg"
    for attempt in 1 2 3; do
      if git -C "$dir" push origin "HEAD:$BRANCH"; then
        exit 0
      fi
      echo "push failed (attempt $attempt), rebasing on the remote branch"
      git -C "$dir" pull --rebase origin "$BRANCH" || true
      sleep $((attempt * 5))
    done
    echo "could not push state" >&2
    exit 1
    ;;
  *)
    echo "unknown command $cmd" >&2
    exit 2
    ;;
esac
