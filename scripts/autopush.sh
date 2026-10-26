#!/usr/bin/env bash
# Commit runs/ and push to $BRANCH every $INTERVAL seconds.
# Pulls by MERGE: --rebase replayed merged-in commits and left the checkout
# mid-rebase for 7 h on 2026-10-26 (plan, Amendments). It never touches git
# while a rebase or merge is in progress, logs every failure, and says
# "pushed" only when the push succeeded.
REPO="$1"; BRANCH="$2"; INTERVAL="${3:-300}"
cd "$REPO" || exit 1
ts() { date -u +%H:%M:%S; }
while true; do
  if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ] || [ -f .git/MERGE_HEAD ]; then
    echo "$(ts)  STUCK: rebase or merge in progress; not touching git"
    sleep "$INTERVAL"; continue
  fi
  git add -A runs/
  if ! git diff --cached --quiet; then
    n=$(git diff --cached --name-only | wc -l | tr -d ' ')
    git commit -q -m "colab: runs checkpoint ($n files) $(date -u +%FT%TZ)" \
      || echo "$(ts)  COMMIT FAILED"
  fi
  if ! out=$(git pull -q --no-rebase --no-edit origin "$BRANCH" 2>&1); then
    echo "$(ts)  PULL FAILED: $out"
    sleep "$INTERVAL"; continue
  fi
  if [ -n "$(git log "origin/$BRANCH..HEAD" --oneline 2>/dev/null)" ]; then
    if out=$(git push -q origin "$BRANCH" 2>&1); then
      echo "$(ts)  pushed through $(git rev-parse --short HEAD)"
    else
      echo "$(ts)  PUSH FAILED: $out"
    fi
  fi
  sleep "$INTERVAL"
done
