#!/bin/bash
# Right after a compact, points at what to read next.
#
# Why:
#   This hook's stdout is injected into context verbatim and the injection window
#   is only a few KB, so it points at locations instead of pasting content.
#   Headings carry line numbers so a section can be fetched with Read's offset.
#
# Guarantees:
#   - Reads data from the hook input's `cwd` (CLAUDE_PROJECT_DIR does not follow worktrees).
#   - Says "missing" when something is missing; never injects empty content.
#   - No defaults: a missing key in the input JSON fails the hook.
#   - session_id validation and path building match compact_snapshot.sh.
#   - Always write ${VAR} (with braces) directly before a non-ASCII character.
set -u

HANDOFF_REL="docs/handoff/HANDOFF.md"

INPUT=$(cat)

FIELDS=$(printf '%s' "$INPUT" | python3 -c 'import json,sys
d = json.load(sys.stdin)
print(d["session_id"])
print(d["cwd"])')
if [ $? -ne 0 ]; then
  echo "compact_resume: cannot extract session_id / cwd from input JSON" >&2
  exit 1
fi

SESSION_ID=$(printf '%s\n' "$FIELDS" | sed -n '1p')
CWD=$(printf '%s\n' "$FIELDS" | sed -n '2p')

case "$SESSION_ID" in
  "" | *[!A-Za-z0-9_-]* )
    echo "compact_resume: session_id does not match ^[A-Za-z0-9_-]{1,64}$: [${SESSION_ID}]" >&2
    exit 1 ;;
esac
if [ "${#SESSION_ID}" -gt 64 ]; then
  echo "compact_resume: session_id longer than 64 characters" >&2
  exit 1
fi
if [ ! -d "$CWD" ]; then
  echo "compact_resume: cwd is not a directory: [${CWD}]" >&2
  exit 1
fi

SNAP="$CWD/.claude/compact/$SESSION_ID.md"
HANDOFF="$CWD/$HANDOFF_REL"

echo "[After compact]"
echo "Before acting, check three things: (1) git state (branch, uncommitted changes)"
echo "(2) tests / smoke run (3) whether a GPU machine is still running and billing"
echo "(Colab / cloud: forgetting to stop one costs real money)."
echo ""

if [ ! -f "$HANDOFF" ]; then
  echo "${HANDOFF_REL} is missing (${CWD})."
else
  echo "## ${HANDOFF_REL} -- first 4 lines and headings (left = line number; read sections with Read offset)"
  echo ""
  sed -n '1,4p' "$HANDOFF" | cut -c1-200
  HEADS=$(grep -n '^##' "$HANDOFF" | head -60 | cut -c1-160)
  if [ -z "$HEADS" ]; then
    echo "No headings at all."
  else
    printf '%s\n' "$HEADS"
  fi
fi
echo ""

if [ -f "$SNAP" ]; then
  echo "## Snapshot taken just before compact (machine-written; read it with Read)"
  echo "${SNAP}"
else
  echo "No snapshot (${SNAP}) -- PreCompact did not run."
fi
exit 0
