#!/usr/bin/env bash
# Sets up the isolated venv for the consolidated text-pipeline engine. Safe
# to re-run.
#
# Consolidates 6 previously-separate engines (indic-ner, spacy-ner,
# word-lid, qwen3-phonetic, nemo-text-norm, indic-text-norm) into one venv —
# these stages are commonly used together regardless of synthesis engine/
# voice, so six separate torch/transformers installs was pure duplication.
#
# nemo_text_processing's and indic-text-normalization's pynini dependency has
# no macOS wheel (manylinux-only) — built from source against Homebrew's
# OpenFST below, same dance both packages previously did independently.
#
# ai4bharat/IndicNER is a GATED HuggingFace model — you must:
#   1. Visit https://huggingface.co/ai4bharat/IndicNER and accept its terms.
#   2. Set HF_TOKEN (a HuggingFace access token) in the repo-root .env.
# Without that, indic-ner requests will fail with a 401.
set -euo pipefail
cd "$(dirname "$0")"

# Remove any broken/partial .venv before creating a fresh one — uv refuses
# to overwrite a directory that exists but isn't a valid virtual environment.
if [ -d .venv ] && [ ! -f .venv/bin/python ]; then
    echo "Removing incomplete .venv (no bin/python found) before recreating..."
    rm -rf .venv
fi

uv venv --python 3.11
uv sync

# pynini needs OpenFST's headers/libs to build from source.
if ! brew list openfst >/dev/null 2>&1; then
    echo "Installing OpenFST via Homebrew (required to build pynini from source)..."
    brew install openfst
fi
OPENFST_PREFIX="$(brew --prefix openfst)"
echo "Building pynini against OpenFST at $OPENFST_PREFIX ..."
# nemo_text_processing pins pynini==2.1.6.post1, but that version doesn't
# build against current Homebrew OpenFST (renamed fst::StringJoin ->
# fst::StrJoin) — use a newer pynini, then install both downstream packages
# with --no-deps so they don't fight the exact pin.
CPLUS_INCLUDE_PATH="$OPENFST_PREFIX/include" LIBRARY_PATH="$OPENFST_PREFIX/lib" \
    uv pip install "pynini>=2.1.7"

uv pip install --no-deps "nemo_text_processing"
uv pip install --no-deps "git+https://github.com/kenpath/indic-text-normalization"

echo "text-pipeline venv ready: text-pipeline/.venv"
echo "Reminder: ai4bharat/IndicNER is gated — accept its terms on HuggingFace and set HF_TOKEN in .env."
echo "Qwen/Qwen3.5-0.8B model weights download lazily on first use."
