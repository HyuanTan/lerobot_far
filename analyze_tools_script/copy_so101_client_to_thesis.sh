#!/usr/bin/env bash
# Copy client-side eval outputs from eval_thesis_client into the matching
# eval_thesis server-side directories so both sit under one tree.
#
# Mapping (paths relative to repo root):
#   outputs/eval_thesis_client/so101_pi05/so101/<method>/<policy>/<param>/
#       -> outputs/eval_thesis/so101/<method>/<policy>/<param>/
#
# Items copied per leaf dir:
#   client_timing/    (directory)
#   trajectories/     (directory)
#   queue.png         (file)
#   client_*.log      (files)
#
# Usage:
#   bash analyze_tools_script/copy_so101_client_to_thesis.sh [--dry-run]

set -euo pipefail

SRC_ROOT="outputs/eval_thesis_client/so101_pi05/so101"
DST_ROOT="outputs/eval_thesis/so101"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    --help|-h)
      sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

if [ ! -d "$SRC_ROOT" ]; then
  echo "ERROR: source directory not found: $SRC_ROOT"
  exit 1
fi

echo "Source:      $SRC_ROOT"
echo "Destination: $DST_ROOT"
[ "$DRY_RUN" -eq 1 ] && echo "(dry-run — no files will be written)"
echo

# ── helpers ──────────────────────────────────────────────────────────────────

# Copy a single directory; replaces dst if already present.
_copy_dir() {
  local src_dir="$1"
  local rel_parent
  rel_parent="$(dirname "${src_dir#"$SRC_ROOT"/}")"
  local name
  name="$(basename "$src_dir")"
  local dst_parent="$DST_ROOT/$rel_parent"
  local dst_dir="$dst_parent/$name"

  echo "  dir : $src_dir"
  echo "     -> $dst_dir"

  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$dst_parent"
    rm -rf "$dst_dir"
    cp -a "$src_dir" "$dst_dir"
  fi
}

# Copy a single file; overwrites dst if already present.
_copy_file() {
  local src_file="$1"
  local rel_file="${src_file#"$SRC_ROOT"/}"
  local dst_file="$DST_ROOT/$rel_file"
  local dst_parent
  dst_parent="$(dirname "$dst_file")"

  echo "  file: $src_file"
  echo "     -> $dst_file"

  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$dst_parent"
    cp -a "$src_file" "$dst_file"
  fi
}

# ── client_timing/ ────────────────────────────────────────────────────────────
echo "=== client_timing/ ==="
find "$SRC_ROOT" -type d -name "client_timing" | sort | while read -r d; do
  _copy_dir "$d"
done

# ── trajectories/ ─────────────────────────────────────────────────────────────
echo ""
echo "=== trajectories/ ==="
find "$SRC_ROOT" -type d -name "trajectories" | sort | while read -r d; do
  _copy_dir "$d"
done

# ── queue.png ─────────────────────────────────────────────────────────────────
echo ""
echo "=== queue.png ==="
find "$SRC_ROOT" -type f -name "queue.png" | sort | while read -r f; do
  _copy_file "$f"
done

# ── client_*.log ──────────────────────────────────────────────────────────────
echo ""
echo "=== client_*.log ==="
find "$SRC_ROOT" -type f -name "client_*.log" | sort | while read -r f; do
  _copy_file "$f"
done

echo ""
echo "Done."
