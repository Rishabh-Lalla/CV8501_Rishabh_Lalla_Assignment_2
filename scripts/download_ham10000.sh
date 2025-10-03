#!/usr/bin/env bash
set -euo pipefail

ROOT="data/ham10000"
mkdir -p "$ROOT"
echo ">>> Downloading HAM10000 via Kaggle CLI..."
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$ROOT"
echo ">>> Unzipping..."
unzip -q "$ROOT/skin-cancer-mnist-ham10000.zip" -d "$ROOT"
echo ">>> Merging image folders..."
mkdir -p "$ROOT/images"
if [ -d "$ROOT/HAM10000_images_part_1" ]; then
  mv "$ROOT/HAM10000_images_part_1"/* "$ROOT/images"/
fi
if [ -d "$ROOT/HAM10000_images_part_2" ]; then
  mv "$ROOT/HAM10000_images_part_2"/* "$ROOT/images"/
fi
echo ">>> Done. Metadata at $ROOT/HAM10000_metadata.csv ; images at $ROOT/images/"
