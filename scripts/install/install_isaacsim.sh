#!/bin/bash
# User-global Isaac Sim 6.2.0 standalone install (no sudo).
#
# Why 6.2.0: the node driver (595.58.03) is the version NVIDIA tested for Isaac Sim
# 6.0.1-6.2.0. Isaac Sim 4.x/5.x (incl. GarmentLab's 4.0.0) crash in the RTX
# renderer on R595 drivers (Vulkan maxMemoryAllocationSize reported as UINT64_MAX).
# See reports/deviation_log.md.
set -euo pipefail

VERSION=6.2.0
URL="https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-${VERSION}-linux-x86_64.zip"
MD5=dde88af0b6423dd3135727385a344fb3
DL_DIR=${DL_DIR:-$HOME/opt/downloads}
PREFIX=${ISAACSIM_PREFIX:-$HOME/opt/isaacsim/${VERSION}}
ZIP="$DL_DIR/$(basename "$URL")"

mkdir -p "$DL_DIR" "$PREFIX"
curl -fL -C - --retry 5 -o "$ZIP" "$URL"
echo "$MD5  $ZIP" | md5sum -c -

# Parallel extraction: NFS home directories are metadata-bound, so extract
# disjoint second-level subtrees concurrently.
unzip -Z1 "$ZIP" | awk -F/ '{ if (NF>2 || ($1 ~ /^(exts|extscache|extsDeprecated|extsInternal|kit|data|standalone_examples)$/ && NF>=2 && $2!="")) print $1"/"$2; else print $1 }' \
  | sort -u | xargs -P 16 -I{} sh -c 'unzip -o -q "$0" "$1" "$1/*" -d "$2" >/dev/null 2>&1 || [ $? -le 11 ]' "$ZIP" {} "$PREFIX"

# Sanity check: every archive member present.
expected=$(unzip -Z1 "$ZIP" | grep -v '/$' | wc -l)
found=$(cd "$PREFIX" && find . -type f -o -type l | wc -l)
echo "archive files: $expected, extracted files: $found"
[ "$found" -ge "$expected" ] || { echo "extraction incomplete" >&2; exit 1; }

ln -sfn "$PREFIX" "$HOME/opt/isaacsim/current"
echo "Isaac Sim $VERSION installed at $PREFIX (symlink ~/opt/isaacsim/current)"
