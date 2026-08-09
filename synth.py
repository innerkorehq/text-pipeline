#!/usr/bin/env python
"""
Consolidated text-preprocessing CLI — IndicNER, spaCy NER, word-level
Hindi/English LID (L3Cube HingBERT), Qwen3.5-0.8B phonetic respelling, NeMo
(English) and indic-text-normalization (Indic) semiotic normalization, all
in ONE venv instead of six. These stages all run on every TTS call
regardless of which synthesis engine/voice is selected (entity detection ->
phonetic rewrite -> number/date normalization; word-lid additionally for
the "hinglish-lid" pipeline's Route B), so splitting them across separate
venvs was pure duplication (separate copies of torch/transformers for
stages that are never optional).

Request (stdin JSON): {"engine": "indic-ner" | "spacy-ner" | "word-lid"
                                 | "qwen3-phonetic" | "nemo-text-norm"
                                 | "indic-text-norm",
                        ...engine-specific fields, unchanged from each
                        engine's own former synth.py contract}

  indic-ner:        {"mode": "tag", "text": "..."}
  spacy-ner:        {"mode": "tag", "text": "..."}
  word-lid:         {"mode": "tag", "text": "..."}
  qwen3-phonetic:   {"mode": "rewrite", "spans": ["Entrackr", ...]}
  nemo-text-norm:   {"mode": "normalize", "text": "..."}
  indic-text-norm:  {"mode": "normalize", "text": "...", "lang": "hi"}

Response (stdout JSON): engine-specific, unchanged — see each section below.
                        {"ok": false, "error": "..."} on failure.
"""
import logging
import os
import sys

from protocol import read_request, succeed, fail, quiet_stdout

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("text-pipeline")


# ── indic-ner: ai4bharat/IndicNER named-entity recognition ──────────────────
#
# Detects PER/ORG/LOC spans in Indic-script narration text so the pipeline
# can protect them from automatic Indic→Roman transliteration. GATED model —
# requires accepting its terms at https://huggingface.co/ai4bharat/IndicNER
# and an HF_TOKEN env var.

_INDIC_NER_MODEL_ID = "ai4bharat/IndicNER"
_indic_ner_pipeline = None


def _load_indic_ner_pipeline():
    global _indic_ner_pipeline
    if _indic_ner_pipeline is not None:
        return _indic_ner_pipeline

    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    logger.info("Loading %s (token configured: %s)…", _INDIC_NER_MODEL_ID, bool(token))

    try:
        tokenizer = AutoTokenizer.from_pretrained(_INDIC_NER_MODEL_ID, token=token, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            _INDIC_NER_MODEL_ID, token=token, local_files_only=True,
        )
    except Exception:
        logger.info("Not fully cached locally yet — downloading %s…", _INDIC_NER_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(_INDIC_NER_MODEL_ID, token=token)
        model = AutoModelForTokenClassification.from_pretrained(_INDIC_NER_MODEL_ID, token=token)

    _indic_ner_pipeline = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    logger.info("IndicNER ready.")
    return _indic_ner_pipeline


def _run_indic_ner(req: dict) -> dict:
    text = req.get("text", "")
    if not text.strip():
        return {"entities": []}

    ner = _load_indic_ner_pipeline()
    results = ner(text)
    entities = [
        {
            "text": r["word"],
            "type": r["entity_group"],
            "start": int(r["start"]),
            "end": int(r["end"]),
            "score": float(r["score"]),
        }
        for r in results
        if r.get("start") is not None and r.get("end") is not None
    ]
    return {"entities": entities}


# ── spacy-ner: broader entity detection (runs alongside IndicNER) ──────────
#
# Tags a broader OntoNotes-5 subset than IndicNER (PRODUCT, EVENT, LAW, NORP,
# WORK_OF_ART, FAC, LANGUAGE in addition to ORG/PERSON/GPE/LOC) — English-
# trained, complements IndicNER's Indic-script training for Hinglish/
# code-mixed narration.

_SPACY_MODEL_ID = "en_core_web_sm"
_SPACY_ALLOWED_TYPES = {
    "ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "LAW", "NORP",
    "WORK_OF_ART", "FAC", "LANGUAGE",
}
_spacy_nlp = None


def _load_spacy_model():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp

    import spacy

    logger.info("Loading %s…", _SPACY_MODEL_ID)
    _spacy_nlp = spacy.load(_SPACY_MODEL_ID, disable=["lemmatizer"])
    logger.info("spaCy NER ready.")
    return _spacy_nlp


def _run_spacy_ner(req: dict) -> dict:
    text = req.get("text", "")
    if not text.strip():
        return {"entities": []}

    nlp = _load_spacy_model()
    doc = nlp(text)
    entities = [
        {"text": ent.text, "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
        if ent.label_ in _SPACY_ALLOWED_TYPES
    ]
    return {"entities": entities}


# ── word-lid: L3Cube HingBERT word-level Hindi/English language ID ─────────
#
# Tags each word in Roman-script code-mixed text as HI/EN/NE/O so the
# "hinglish-lid" pipeline's Route B knows which words to transliterate
# (HI) vs leave alone (EN/NE) — see src/infra/services/text_pipelines.py's
# step_script_aware_normalize. Not gated — unlike ai4bharat/IndicNER, this
# model needs no HF_TOKEN.

_WORD_LID_MODEL_ID = "l3cube-pune/hing-bert-lid"
_word_lid_tokenizer = None
_word_lid_model = None
_word_lid_label_aliases: dict[int, str] = {}

# Maps whatever label strings the model's own id2label reports to our
# normalized 4-tag scheme — built once at load time from the model's actual
# config rather than assumed, since the model card doesn't document an
# exact label list. Confirmed empirically: this checkpoint's id2label is
# just {0: "EN", 1: "HI"} — a flat binary tagger, no BIO/NE/O labels at all
# — which is also why HF's `pipeline(..., aggregation_strategy="simple")`
# is the WRONG tool here: "simple" aggregation merges consecutive
# SAME-LABEL subword tokens into one span regardless of word boundaries, so
# two adjacent English words ("main office") collapse into a single
# multi-word "entity" and word-level granularity is lost. This handler
# instead tags at the subword level directly and merges only WordPiece
# continuations (##-prefixed) back into their own word — never across a
# word boundary — using the first subtoken's label per word.
_LABEL_ALIAS_TABLE = {
    "hi": "HI", "hin": "HI", "hindi": "HI",
    "en": "EN", "eng": "EN", "english": "EN",
    "ne": "NE", "name": "NE", "named_entity": "NE", "ner": "NE",
    "o": "O", "other": "O", "univ": "O", "mixed": "O", "acronym": "O",
    "punct": "O", "punctuation": "O",
}


def _load_word_lid_model():
    global _word_lid_tokenizer, _word_lid_model, _word_lid_label_aliases
    if _word_lid_model is not None:
        return _word_lid_tokenizer, _word_lid_model

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    logger.info("Loading %s…", _WORD_LID_MODEL_ID)
    try:
        tokenizer = AutoTokenizer.from_pretrained(_WORD_LID_MODEL_ID, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(_WORD_LID_MODEL_ID, local_files_only=True)
    except Exception:
        logger.info("Not fully cached locally yet — downloading %s…", _WORD_LID_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(_WORD_LID_MODEL_ID)
        model = AutoModelForTokenClassification.from_pretrained(_WORD_LID_MODEL_ID)

    id2label = getattr(model.config, "id2label", {}) or {}
    logger.info("word-lid id2label: %s", id2label)
    _word_lid_label_aliases = {
        int(idx): _LABEL_ALIAS_TABLE.get(str(raw).lower().lstrip("bi-_"), "O")
        for idx, raw in id2label.items()
    }

    model.eval()
    _word_lid_tokenizer, _word_lid_model = tokenizer, model
    logger.info("word-lid ready.")
    return tokenizer, model


def _run_word_lid(req: dict) -> dict:
    text = req.get("text", "")
    if not text.strip():
        return {"tokens": []}

    import torch

    tokenizer, model = _load_word_lid_model()
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offsets = encoding.pop("offset_mapping")[0].tolist()
    word_ids = encoding.encodings[0].word_ids

    with torch.no_grad():
        logits = model(**encoding).logits[0]
    pred_ids = logits.argmax(dim=-1).tolist()

    # Merge subword tokens sharing the same `word_ids()` word index into one
    # word span, using the FIRST subtoken's predicted label for that word —
    # never merging across a word boundary (unlike aggregation_strategy).
    tokens: list[dict] = []
    current_word_id = None
    for (start, end), wid, pred_id in zip(offsets, word_ids, pred_ids):
        if wid is None or start == end:  # special tokens ([CLS]/[SEP]/padding)
            continue
        if wid != current_word_id:
            tokens.append({
                "word": text[start:end],
                "tag": _word_lid_label_aliases.get(pred_id, "O"),
                "start": start,
                "end": end,
            })
            current_word_id = wid
        else:
            tokens[-1]["end"] = end
            tokens[-1]["word"] = text[tokens[-1]["start"]:end]

    return {"tokens": tokens}


# ── qwen3-phonetic: Qwen/Qwen3.5-0.8B phonetic respelling of entity spans ───
#
# Respells brand/person/place/product names (tagged by the NER stages above)
# in plain Roman-script syllables a TTS engine will pronounce correctly.
# Scoped to entity spans only — generic number/date normalization is the
# nemo-text-norm/indic-text-norm engines' job.

_QWEN3_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_QWEN3_SYSTEM_PROMPT = (
    "You respell a single name or term so a text-to-speech engine reads it "
    "correctly aloud. Reply with ONLY the respelled term in plain Roman "
    "letters — nothing else: no explanation, no markdown, no bullet points, "
    "no surrounding quotes. "
    "If the term is already a common, easy-to-pronounce English word or "
    "name, reply with it completely unchanged, same capitalization. "
    "If it has multiple words, keep them as separate words (do not join "
    "them with hyphens) and keep each word's original capitalization. "
    "If a single word is a made-up brand name, an acronym, or a "
    "non-Latin-script name, break THAT word into intuitive syllables "
    "separated by hyphens so it's read out naturally "
    "(e.g. \"Entrackr\" -> \"En-track-er\", \"K2-FSA\" -> \"K two F S A\")."
)
_qwen3_processor = None
_qwen3_model = None


def _get_qwen3_model():
    global _qwen3_processor, _qwen3_model
    if _qwen3_model is not None:
        return _qwen3_processor, _qwen3_model

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading %s (device=%s)…", _QWEN3_MODEL_ID, device)

    try:
        _qwen3_processor = AutoProcessor.from_pretrained(_QWEN3_MODEL_ID, local_files_only=True)
        _qwen3_model = AutoModelForImageTextToText.from_pretrained(
            _QWEN3_MODEL_ID, dtype=torch.float32, local_files_only=True,
        ).to(device)
    except Exception:
        logger.info("Not fully cached locally yet — downloading %s…", _QWEN3_MODEL_ID)
        _qwen3_processor = AutoProcessor.from_pretrained(_QWEN3_MODEL_ID)
        _qwen3_model = AutoModelForImageTextToText.from_pretrained(_QWEN3_MODEL_ID, dtype=torch.float32).to(device)

    logger.info("Qwen3.5-phonetic ready.")
    return _qwen3_processor, _qwen3_model


def _qwen3_rewrite_one(processor, model, span: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": _QWEN3_SYSTEM_PROMPT},
        {"role": "user", "content": span},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = first_line.lstrip("-*• ").strip("\"'` ")
    if not first_line:
        return span

    if first_line.replace("-", "").lower() == span.replace("-", "").lower():
        return span
    return first_line


def _run_qwen3_phonetic(req: dict) -> dict:
    spans = req.get("spans", [])
    if not spans:
        return {"rewrites": {}}

    rewrites: dict[str, str] = {}
    needs_llm = [s for s in spans if s and s.strip() and " " not in s.strip()]
    for span in spans:
        if span and span.strip() and " " in span.strip():
            rewrites[span] = span

    if not needs_llm:
        return {"rewrites": rewrites}

    processor, model = _get_qwen3_model()
    for span in needs_llm:
        try:
            rewrites[span] = _qwen3_rewrite_one(processor, model, span)
        except Exception as e:
            logger.warning("phonetic rewrite failed for %r: %s", span, e)
            rewrites[span] = span

    return {"rewrites": rewrites}


# ── nemo-text-norm: English semiotic text normalization ─────────────────────
#
# WFST-based (Pynini): https://github.com/NVIDIA/NeMo-text-processing
# Runs alongside indic-text-norm (not instead of it) — English-only.

_nemo_normalizer = None


def _get_nemo_normalizer():
    global _nemo_normalizer
    if _nemo_normalizer is not None:
        return _nemo_normalizer

    from nemo_text_processing.text_normalization.normalize import Normalizer

    logger.info("Building NeMo Normalizer(lang=en)…")
    _nemo_normalizer = Normalizer(input_case="cased", lang="en")
    return _nemo_normalizer


def _run_nemo_text_norm(req: dict) -> dict:
    text = req.get("text", "")
    if not text.strip():
        return {"text": text}
    normalizer = _get_nemo_normalizer()
    return {"text": normalizer.normalize(text)}


# ── indic-text-norm: semiotic text normalization for 19 Indian languages ───
#
# WFST-based (Pynini), extension of NVIDIA NeMo Text Processing:
# https://github.com/kenpath/indic-text-normalization

_INDIC_NORM_SUPPORTED_LANGS = {
    "hi", "bn", "mr", "te", "kn", "bho", "mag", "hne", "mai",
    "as", "brx", "doi", "gu", "ml", "pa", "ta", "en", "ne", "sa",
}
_indic_normalizers: dict = {}


def _get_indic_normalizer(lang: str):
    from indic_text_normalization import Normalizer

    if lang not in _INDIC_NORM_SUPPORTED_LANGS:
        logger.info("Unsupported lang=%r, falling back to 'en'", lang)
        lang = "en"
    if lang not in _indic_normalizers:
        logger.info("Building indic-text-norm Normalizer(lang=%s)…", lang)
        _indic_normalizers[lang] = Normalizer(input_case="cased", lang=lang)
    return _indic_normalizers[lang]


def _run_indic_text_norm(req: dict) -> dict:
    text = req.get("text", "")
    lang = (req.get("lang") or "en").lower()
    normalizer = _get_indic_normalizer(lang)
    return {"text": normalizer.normalize(text)}


# ── Dispatch ──────────────────────────────────────────────────────────────

_HANDLERS = {
    "indic-ner": ("tag", _run_indic_ner),
    "spacy-ner": ("tag", _run_spacy_ner),
    "word-lid": ("tag", _run_word_lid),
    "qwen3-phonetic": ("rewrite", _run_qwen3_phonetic),
    "nemo-text-norm": ("normalize", _run_nemo_text_norm),
    "indic-text-norm": ("normalize", _run_indic_text_norm),
}


def main() -> None:
    req = read_request()
    engine = req.get("engine")
    mode = req.get("mode")

    handler_entry = _HANDLERS.get(engine)
    if handler_entry is None:
        fail(f"Unknown engine: {engine!r} (expected one of {sorted(_HANDLERS)})")
        return
    expected_mode, handler = handler_entry
    if mode != expected_mode:
        fail(f"Unknown mode: {mode!r} for engine {engine!r} (expected {expected_mode!r})")
        return

    try:
        with quiet_stdout():
            result = handler(req)
    except Exception as e:
        logger.exception("%s failed", engine)
        fail(str(e))
        return

    succeed(**result)


if __name__ == "__main__":
    main()
