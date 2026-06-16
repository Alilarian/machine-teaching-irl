#!/bin/bash
# Fetch experiment results from CHPC scratch to local results_chpc/
#
# Usage:
#   ./fetch_results.sh [--user=u1472210] [--host=kingspeak.chpc.utah.edu] [--dry-run]
#
# Defaults match the standard CHPC kingspeak setup.
# Results land in:  <repo_root>/results_chpc/

set -euo pipefail

CHPC_USER="${CHPC_USER:-u1472210}"
CHPC_HOST="${CHPC_HOST:-kingspeak.chpc.utah.edu}"
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --user=*)    CHPC_USER="${arg#*=}" ;;
    --host=*)    CHPC_HOST="${arg#*=}" ;;
    --dry-run)   DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

REMOTE_PATH="/scratch/general/nfs1/${CHPC_USER}/paper_results/"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PATH="${SCRIPT_DIR}/../results_chpc/"

mkdir -p "$LOCAL_PATH"

RSYNC_ARGS="-avz --progress"
[[ "$DRY_RUN" == true ]] && RSYNC_ARGS="$RSYNC_ARGS --dry-run"

echo "Fetching from ${CHPC_USER}@${CHPC_HOST}:${REMOTE_PATH}"
echo "       into ${LOCAL_PATH}"
[[ "$DRY_RUN" == true ]] && echo "(dry-run)"
echo ""

# shellcheck disable=SC2086
rsync $RSYNC_ARGS \
  --include="*/" \
  --include="*.json" \
  --exclude="*" \
  "${CHPC_USER}@${CHPC_HOST}:${REMOTE_PATH}" \
  "$LOCAL_PATH"

echo ""
echo "Done. JSON files are in: ${LOCAL_PATH}"
