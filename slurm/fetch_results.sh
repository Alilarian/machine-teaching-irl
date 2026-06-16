#!/bin/bash
# Fetch experiment results from CHPC scratch to local results_chpc/
#
# CHPC uses Duo 2FA, so direct rsync can't authenticate on its own.
# This script uses SSH ControlMaster: it opens an authenticated session
# once (you complete Duo there), then rsync reuses it without re-prompting.
#
# Usage:
#   ./fetch_results.sh [--host=chpc-king] [--user=u1472210] [--dry-run]
#
# --host accepts either the SSH alias from ~/.ssh/config (e.g. chpc-king)
# or a full hostname (kingspeak.chpc.utah.edu).
# Results land in:  <repo_root>/results_chpc/

set -euo pipefail

CHPC_USER="${CHPC_USER:-u1472210}"
CHPC_HOST="${CHPC_HOST:-chpc-king}"
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --user=*)  CHPC_USER="${arg#*=}" ;;
    --host=*)  CHPC_HOST="${arg#*=}" ;;
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

REMOTE_PATH="/scratch/general/nfs1/${CHPC_USER}/paper_results/"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PATH="${SCRIPT_DIR}/../results_chpc/"
CTRL_SOCK="/tmp/chpc-ctrl-${CHPC_USER}"

mkdir -p "$LOCAL_PATH"

# ── Step 1: open a ControlMaster connection (triggers Duo 2FA once) ──────────
if [[ ! -S "$CTRL_SOCK" ]]; then
  echo "Opening SSH ControlMaster to ${CHPC_HOST} (complete Duo 2FA when prompted)..."
  ssh -fNM \
      -o ControlMaster=yes \
      -o ControlPath="$CTRL_SOCK" \
      -o ControlPersist=10m \
      -o IdentitiesOnly=yes \
      "${CHPC_HOST}"
  echo "Connection established. Socket: ${CTRL_SOCK}"
else
  echo "Reusing existing ControlMaster socket: ${CTRL_SOCK}"
fi

# ── Step 2: rsync over the existing authenticated connection ──────────────────
SSH_CMD="ssh -o ControlMaster=no -o ControlPath=${CTRL_SOCK} -o IdentitiesOnly=yes"

RSYNC_ARGS="-avz --progress"
[[ "$DRY_RUN" == true ]] && RSYNC_ARGS="$RSYNC_ARGS --dry-run"

echo ""
echo "Fetching ${CHPC_HOST}:${REMOTE_PATH}"
echo "   into  ${LOCAL_PATH}"
[[ "$DRY_RUN" == true ]] && echo "(dry-run)"
echo ""

# shellcheck disable=SC2086
rsync $RSYNC_ARGS \
  -e "$SSH_CMD" \
  --include="*/" \
  --include="*.json" \
  --exclude="*" \
  "${CHPC_HOST}:${REMOTE_PATH}" \
  "$LOCAL_PATH"

echo ""
echo "Done. JSON files are in: ${LOCAL_PATH}"
