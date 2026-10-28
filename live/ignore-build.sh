#!/usr/bin/env bash
# Vercel's ignored build step for stoa-live (live/vercel.json). Exit 0 skips the build, any
# other exit builds. Runs in live/, the project's Root Directory, so "." is live/.
#
# A push to main that leaves live/ alone is skipped. A redeploy of the commit already live
# always builds, so rotated keys take effect. Vercel clones only the last few commits, so the
# last deployed commit is fetched when the clone does not have it; if that fails, it builds.
set -u

prev="${VERCEL_GIT_PREVIOUS_SHA:-}"
head="${VERCEL_GIT_COMMIT_SHA:-}"
remote="${STOA_IGNORE_REMOTE:-https://github.com/${VERCEL_GIT_REPO_OWNER:-AddisonTech}/${VERCEL_GIT_REPO_SLUG:-Stoa}.git}"

if [ -n "$prev" ] && [ "$prev" = "$head" ]; then
  echo "redeploy of the live commit: build"
  exit 1
fi
base="${prev:-HEAD^}"
if ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
  if [ -z "$prev" ] || ! git fetch --quiet --depth=1 "$remote" "$prev"; then
    echo "cannot find $base: build"
    exit 1
  fi
fi
if git diff --quiet "$base" HEAD -- .; then
  echo "live/ unchanged since $base: skip"
  exit 0
fi
echo "live/ changed since $base: build"
exit 1
