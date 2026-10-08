# NOTE: this engine depends on pynini (built from source against OpenFST),
# installed via a manual, macOS/Homebrew-specific dance in setup.sh — see
# that file for why (nemo_text_processing/indic-text-normalization's pynini
# pins don't build against modern OpenFST, and pynini itself has no macOS
# wheel on PyPI). This Dockerfile uses Debian's libfst-dev instead; if it
# breaks, prefer running this engine natively via setup.sh over debugging
# the container build.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libfst-dev git wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock setup.sh ./
RUN uv venv --python 3.13 && uv sync --frozen --extra http
RUN uv pip install "pynini>=2.1.7" \
    && uv pip install --no-deps "nemo_text_processing" \
    && uv pip install --no-deps "git+https://github.com/kenpath/indic-text-normalization"

COPY engine.py server.py ./

ENV PORT=8010
EXPOSE 8010

# ai4bharat/IndicNER is gated on Hugging Face — pass HF_TOKEN at runtime:
#   docker run -e HF_TOKEN=... -p 8010:8010 text-pipeline
CMD [".venv/bin/python", "server.py"]
