#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="outputs/eval_client/so101_V2.0_0521"
DST_ROOT="outputs/eval/so101_V2.0_0521"

if [ ! -d "$SRC_ROOT" ]; then
  echo "ERROR: source directory not found: $SRC_ROOT"
  exit 1
fi

mkdir -p "$DST_ROOT"

echo "Source:      $SRC_ROOT"
echo "Destination: $DST_ROOT"
echo


# 复制 client_timing 目录
find "$SRC_ROOT" -type d -name "client_timing" | while read -r src_dir; do
  rel_parent="$(dirname "${src_dir#$SRC_ROOT/}")"
  dst_parent="$DST_ROOT/$rel_parent"
  dst_dir="$dst_parent/client_timing"

  mkdir -p "$dst_parent"

  echo "Copy dir:"
  echo "  $src_dir"
  echo "  -> $dst_dir"

  rm -rf "$dst_dir"
  cp -a "$src_dir" "$dst_dir"
done

# 复制 queue.png 文件
find "$SRC_ROOT" -type f -name "queue.png" | while read -r src_file; do
  rel_file="${src_file#$SRC_ROOT/}"
  dst_file="$DST_ROOT/$rel_file"
  dst_parent="$(dirname "$dst_file")"

  mkdir -p "$dst_parent"

  echo "Copy file:"
  echo "  $src_file"
  echo "  -> $dst_file"

  cp -a "$src_file" "$dst_file"
done

echo
echo "Done."