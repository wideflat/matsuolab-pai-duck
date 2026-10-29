#!/bin/bash
# Appends machine-derivable state to one file right before a compact.
#
# Why:
#   Compaction summarizes the conversation and you cannot choose what it drops.
#   This hook copies verbatim the facts a summary tends to lose and that are
#   costly to rediscover: git state, uncommitted changes, the latest experiment
#   rows, and the GPU lines in docs/notes.md.
#
# Guarantees:
#   - Reads data from the hook input's `cwd` (CLAUDE_PROJECT_DIR does not follow worktrees).
#   - Appends: several compacts happen per session; overwriting would erase the trail.
#   - Never exits 2: blocking a compact fails the whole request while recovering
#     from the context limit.
#   - No defaults: a missing key in the input JSON fails the hook.
#   - The handoff file name is only ever docs/handoff/HANDOFF.md.
#   - Always write ${VAR} (with braces) directly before a non-ASCII character.
set -u

HANDOFF_REL="docs/handoff/HANDOFF.md"
EXPERIMENTS_REL="docs/experiments.md"
NOTES_REL="docs/notes.md"

INPUT=$(cat)

# `jq '.session_id'` prints null and succeeds on a missing key, so use python (KeyError).
FIELDS=$(printf '%s' "$INPUT" | python3 -c 'import json,sys
d = json.load(sys.stdin)
print(d["session_id"])
print(d["cwd"])
print(d["trigger"])')
if [ $? -ne 0 ]; then
  echo "compact_snapshot: cannot extract session_id / cwd / trigger from input JSON" >&2
  exit 1
fi

SESSION_ID=$(printf '%s\n' "$FIELDS" | sed -n '1p')
CWD=$(printf '%s\n' "$FIELDS" | sed -n '2p')
TRIGGER=$(printf '%s\n' "$FIELDS" | sed -n '3p')

# Validate before building a path from external input.
case "$SESSION_ID" in
  "" | *[!A-Za-z0-9_-]* )
    echo "compact_snapshot: session_id does not match ^[A-Za-z0-9_-]{1,64}$: [${SESSION_ID}]" >&2
    exit 1 ;;
esac
if [ "${#SESSION_ID}" -gt 64 ]; then
  echo "compact_snapshot: session_id longer than 64 characters" >&2
  exit 1
fi
if [ ! -d "$CWD" ]; then
  echo "compact_snapshot: cwd is not a directory: [${CWD}]" >&2
  exit 1
fi

OUT_DIR="$CWD/.claude/compact"
MKDIR_ERR=$(mkdir -p "$OUT_DIR" 2>&1)
if [ $? -ne 0 ]; then
  echo "compact_snapshot: cannot create ${OUT_DIR}: ${MKDIR_ERR}" >&2
  exit 1
fi
OUT="$OUT_DIR/$SESSION_ID.md"

if [ -f "$OUT" ]; then
  N=$(grep -c '^## compact #' "$OUT")
else
  N=0
fi
N=$((N + 1))

# core.quotepath=false: otherwise non-ASCII file names come out octal-escaped.
GIT="git -C $CWD -c core.quotepath=false"

{
  echo ""
  echo "## compact #${N} -- $(date '+%Y-%m-%d %H:%M:%S %z') / trigger=${TRIGGER}"
  echo ""
  echo "**cwd:** \`${CWD}\`"
  echo ""

  # --- distance from the handoff ---------------------------------------------
  GIT_ERR=$(mktemp)
  BRANCH=$($GIT rev-parse --abbrev-ref HEAD 2>"$GIT_ERR")
  if [ $? -ne 0 ]; then
    echo "**git failed (rev-parse):**"
    echo '```'; cat "$GIT_ERR"; echo '```'
  else
    echo "**branch:** \`${BRANCH}\`"
  fi
  echo ""
  HANDOFF_COMMIT=$($GIT log -1 --format=%H -- "$HANDOFF_REL" 2>"$GIT_ERR")
  if [ $? -ne 0 ]; then
    echo "**git failed (log):**"
    echo '```'; cat "$GIT_ERR"; echo '```'
  elif [ -z "$HANDOFF_COMMIT" ]; then
    echo "**${HANDOFF_REL} has no commit yet.**"
  else
    DIST=$($GIT rev-list --count "$HANDOFF_COMMIT..HEAD" 2>"$GIT_ERR")
    if [ $? -ne 0 ]; then
      echo "**git failed (rev-list):**"
      echo '```'; cat "$GIT_ERR"; echo '```'
    else
      echo "**${DIST} commits since ${HANDOFF_REL}** (a proxy for staleness, not staleness itself)"
      if [ "$DIST" -gt 0 ]; then
        echo ""
        echo '```'
        $GIT log --oneline "$HANDOFF_COMMIT..HEAD"
        echo '```'
      fi
    fi
  fi
  rm -f "$GIT_ERR"
  echo ""

  # --- uncommitted -----------------------------------------------------------
  echo "**Uncommitted:**"
  echo ""
  echo '```'
  $GIT status --short 2>&1
  echo '```'
  echo ""

  # --- latest experiment rows ------------------------------------------------
  if [ ! -f "$CWD/$EXPERIMENTS_REL" ]; then
    echo "**${EXPERIMENTS_REL} does not exist.**"
  else
    ROWS=$(grep -E '^\| *[A-Za-z0-9]' "$CWD/$EXPERIMENTS_REL" | grep -v '^| *ID ' | tail -3)
    if [ -z "$ROWS" ]; then
      echo "**${EXPERIMENTS_REL} has no experiment rows yet.**"
    else
      echo "**Last experiment rows:**"
      echo ""
      echo '```'
      printf '%s\n' "$ROWS" | cut -c1-240
      echo '```'
    fi
  fi
  echo ""

  # --- GPU lines (starting/stopping a GPU machine is a loggable event) -------
  if [ ! -f "$CWD/$NOTES_REL" ]; then
    echo "**${NOTES_REL} does not exist.**"
  else
    GPU=$(grep -n 'GPU' "$CWD/$NOTES_REL" | tail -5)
    if [ -z "$GPU" ]; then
      echo "**No GPU line in ${NOTES_REL}.** (Absence is itself worth reporting.)"
    else
      echo "**GPU lines in ${NOTES_REL}:**"
      echo ""
      echo '```'
      printf '%s\n' "$GPU" | cut -c1-200
      echo '```'
    fi
  fi
} >> "$OUT"

# This status is the last command in the block or a failed redirect; it does not
# catch a mid-block write failure.
if [ $? -ne 0 ]; then
  echo "compact_snapshot: cannot write ${OUT}" >&2
  exit 1
fi
exit 0
