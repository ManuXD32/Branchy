#!/usr/bin/env bash
# Download the two trained models into models/ . Set BASE to where you published them (see MODELS.md), e.g.
#   BASE=https://github.com/<you>/<repo>/releases/download/v1 bash scripts/get_models.sh
set -e
cd "$(dirname "$0")/.."
BASE="${BASE:?set BASE to the URL that holds the model files (see MODELS.md)}"
mkdir -p models && cd models
for f in branchy-model-sequential-313M.zip branchy-model-branched8-309M.zip tokenizer.json SHA256SUMS; do curl -fL -o "$f" "$BASE/$f"; done
for z in branchy-model-*.zip; do unzip -oq "$z" && rm "$z"; done
sha256sum -c SHA256SUMS
