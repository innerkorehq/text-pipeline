# text-pipeline

A standalone text-processing library for TTS/speech pipelines — 6 tools,
each loaded lazily on first use. `engine.py` is a plain Python module
(`run({"engine": ..., "mode": ..., ...})`, no web framework); `server.py` is
an optional HTTP wrapper (`uv sync --extra http`) for standalone use.
vidgen calls `engine.py` in-process, one warm process per tool (no HTTP,
no ports):

| engine | what it does |
|---|---|
| `indic-ner` | Named-entity recognition on Indic-script text ([ai4bharat/IndicNER](https://huggingface.co/ai4bharat/IndicNER)) — PER/ORG/LOC spans, useful for protecting names from transliteration. **Gated model** — see below. |
| `spacy-ner` | Broader English entity recognition (spaCy `en_core_web_sm`) — ORG/PERSON/GPE/LOC plus PRODUCT/EVENT/LAW/NORP/WORK_OF_ART/FAC/LANGUAGE. |
| `word-lid` | Word-level Hindi/English language identification ([L3Cube HingBERT](https://huggingface.co/l3cube-pune/hing-bert-lid)) — tags each word HI/EN/NE/O in Roman-script code-mixed text. |
| `qwen3-phonetic` | Phonetic respelling of names/terms (`Qwen/Qwen3.5-0.8B`) so a TTS engine pronounces them correctly, e.g. `"Entrackr"` → `"En-track-er"`. |
| `nemo-text-norm` | English semiotic text normalization (numbers, dates, currency) via [NVIDIA NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing). |
| `indic-text-norm` | The same, for 19 Indian languages, via [indic-text-normalization](https://github.com/kenpath/indic-text-normalization). |

These are genuinely independent, generically useful NLP tools — nothing
here is TTS-specific or tied to any particular pipeline; they're bundled
into one service because they're commonly used together and share heavy
dependencies (torch/transformers).

> **Gated model**: `ai4bharat/IndicNER` requires accepting its license on
> Hugging Face and an `HF_TOKEN` with access — see the
> [model page](https://huggingface.co/ai4bharat/IndicNER). The other 5
> engines need no token.
>
> **pynini build**: `nemo-text-norm` and `indic-text-norm` depend on
> `pynini`, which has no macOS wheel on PyPI and must be built from source
> against OpenFST — `setup.sh` handles this (installs OpenFST via Homebrew
> on macOS). See the Dockerfile for the Linux equivalent.

## Run it

```bash
export HF_TOKEN=hf_...   # only needed for indic-ner
bash setup.sh
uv run --extra http server.py          # PORT=8010 by default
```

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /process`

**indic-ner** / **spacy-ner** — named-entity recognition:

```json
{"engine": "indic-ner", "mode": "tag", "text": "राहुल दिल्ली गए"}
```
```json
{"entities": [{"text": "राहुल", "type": "PER", "start": 0, "end": 5, "score": 0.98}]}
```

**word-lid** — word-level language ID:

```json
{"engine": "word-lid", "mode": "tag", "text": "kal main office jaunga"}
```
```json
{"tokens": [{"word": "kal", "tag": "HI", "start": 0, "end": 3}, ...]}
```

**qwen3-phonetic** — phonetic respelling:

```json
{"engine": "qwen3-phonetic", "mode": "rewrite", "spans": ["Entrackr", "K2-FSA"]}
```
```json
{"rewrites": {"Entrackr": "En-track-er", "K2-FSA": "K two F S A"}}
```

**nemo-text-norm** — English number/date/currency normalization:

```json
{"engine": "nemo-text-norm", "mode": "normalize", "text": "It costs $42.50"}
```
```json
{"text": "It costs forty two dollars, fifty cents"}
```

**indic-text-norm** — the same for 19 Indian languages:

```json
{"engine": "indic-text-norm", "mode": "normalize", "text": "...", "lang": "hi"}
```
```json
{"text": "..."}
```

**On error**: an HTTP error status (422 for an unknown engine/mode, 500 for
a processing failure) with a JSON `{"detail": "..."}` body.

## Licenses

Each model is a separate upstream project with its own license — see the
linked model cards/repositories above.
