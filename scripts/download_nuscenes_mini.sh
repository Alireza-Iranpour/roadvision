#!/usr/bin/env bash
# Fetch and lay out nuScenes v1.0-mini under data/nuscenes.
#
# nuScenes does not offer a direct anonymous download URL -- you must
# register at https://www.nuscenes.org/sign-up and accept the dataset's
# terms of use, then download the archives from
# https://www.nuscenes.org/download (the "Full dataset (v1.0)" ->
# "Mini" row, plus the "Map expansion" pack used by the BEV segmentation
# targets) to this directory yourself.
#
# Usage:
#   ./scripts/download_nuscenes_mini.sh /path/to/downloaded/archives
#
# Expects to find these files in the given directory:
#   v1.0-mini.tgz
#   nuScenes-map-expansion-v1.3.zip

set -euo pipefail

ARCHIVE_DIR="${1:-}"
DATAROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/nuscenes"

if [[ -z "$ARCHIVE_DIR" ]]; then
  echo "Usage: $0 /path/to/downloaded/archives" >&2
  echo "" >&2
  echo "Download v1.0-mini.tgz and nuScenes-map-expansion-v1.3.zip manually from" >&2
  echo "https://www.nuscenes.org/download (registration required) first." >&2
  exit 1
fi

MINI_ARCHIVE="$ARCHIVE_DIR/v1.0-mini.tgz"
MAP_ARCHIVE="$ARCHIVE_DIR/nuScenes-map-expansion-v1.3.zip"

if [[ ! -f "$MINI_ARCHIVE" ]]; then
  echo "Missing $MINI_ARCHIVE -- download it from https://www.nuscenes.org/download first." >&2
  exit 1
fi
if [[ ! -f "$MAP_ARCHIVE" ]]; then
  echo "Missing $MAP_ARCHIVE -- download it from https://www.nuscenes.org/download first." >&2
  exit 1
fi

mkdir -p "$DATAROOT"

echo "Extracting $MINI_ARCHIVE -> $DATAROOT"
tar -xzf "$MINI_ARCHIVE" -C "$DATAROOT"

echo "Extracting $MAP_ARCHIVE -> $DATAROOT"
unzip -o -q "$MAP_ARCHIVE" -d "$DATAROOT"

echo "Done. Verify with:"
echo "  python -c \"from nuscenes.nuscenes import NuScenes; NuScenes(version='v1.0-mini', dataroot='$DATAROOT', verbose=True)\""
