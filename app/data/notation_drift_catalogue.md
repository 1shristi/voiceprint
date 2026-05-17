# Notation drift catalogue
Comparison of every phoneme in linguamatch's `L1_PHONEMES` table against the symbols `facebook/wav2vec2-lv-60-espeak-cv-ft` emits.
- Snapshot: `app/data/_l1_phonemes_snapshot.txt`
- Alphabet: `app/data/phoneme_alphabet.json` (version 1.0.0)
- Generated: 2026-05-16 by `scripts/notation_audit.py catalogue`

## Methodology
For each L1 phoneme token (e.g. `/θ/`, `/pʰ/`), we strip the surrounding slashes and look up matching vocabulary entries in `phoneme_alphabet.json`. Each match is reported alongside its eSpeak vocab symbol so consumers can see what the model would actually emit. Mismatches are classified:

- **notation-only** — the phoneme is in the model's vocabulary but under a different surface form (e.g. linguamatch wants `/tʃ/`, model emits `tS`).
- **phonological** — no vocabulary entry covers this phoneme; the model cannot emit it and downstream consumers should not expect to see it.
- **direct** — the phoneme matches a vocab entry exactly under its IPA surface form, no translation needed.

## Summary

- direct matches:        273
- notation-only drift:   1
- phonological gaps:     12
- total tokens audited:  286

## Phonological gaps (model cannot emit)

These phonemes appear in `L1_PHONEMES` but no vocabulary entry covers them. Free decoding cannot produce them under any notation; Phase 1/2 consumers should treat them as structurally invisible.

- `/kʼ/`: amharic, aymara, quechua, tigrinya
- `/pʼ/`: amharic, aymara, quechua, tigrinya
- `/tʼ/`: amharic, aymara, quechua, tigrinya

## Notation-only drift (translation layer recommended)

These phonemes are in the model's vocabulary but under a different surface form than `L1_PHONEMES` uses. Loop 3b Phase 1 should normalise via `phoneme_alphabet.json#ipa_alternates`.

- `/ʈʂ/` (emitted as `ts.`): mandarin

## Per-language detail

### amharic

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʼ/` | `pʼ` | — | phonological |
| `/tʼ/` | `tʼ` | — | phonological |
| `/kʼ/` | `kʼ` | — | phonological |
| `/h/` | `h` | `h` | direct |
| `/ʔ/` | `ʔ` | `??`, `ʔ` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |

### arabic

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/q/` | `q` | `q` | direct |
| `/ħ/` | `ħ` | `ħ` | direct |
| `/ʕ/` | `ʕ` | `ʕ` | direct |
| `/θ/` | `θ` | `θ` | direct |
| `/ð/` | `ð` | `ð` | direct |
| `/χ/` | `χ` | `χ` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |

### aymara

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʼ/` | `pʼ` | — | phonological |
| `/tʼ/` | `tʼ` | — | phonological |
| `/kʼ/` | `kʼ` | — | phonological |
| `/q/` | `q` | `q` | direct |
| `/h/` | `h` | `h` | direct |

### basque

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### belarusian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |
| `/ts/` | `ts` | `ts` | direct |

### bengali

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### bosnian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/dʒ/` | `dʒ` | `dZ`, `dʒ` | direct |

### bulgarian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |

### burmese

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |

### cantonese

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |

### catalan

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### croatian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/dʒ/` | `dʒ` | `dZ`, `dʒ` | direct |

### czech

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |

### danish

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### dari

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/q/` | `q` | `q` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/r/` | `r` | `r` | direct |

### dutch

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/x/` | `x` | `X`, `x` | direct |
| `/ʁ/` | `ʁ` | `ʁ` | direct |

### english

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/θ/` | `θ` | `θ` | direct |
| `/ð/` | `ð` | `ð` | direct |
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ɹ/` | `ɹ` | `ɹ` | direct |
| `/h/` | `h` | `h` | direct |

### estonian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### finnish

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### french

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/ʁ/` | `ʁ` | `ʁ` | direct |
| `/ɲ/` | `ɲ` | `ɲ` | direct |

### german

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʁ/` | `ʁ` | `ʁ` | direct |
| `/ç/` | `ç` | `ç` | direct |
| `/x/` | `x` | `X`, `x` | direct |

### greek

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/θ/` | `θ` | `θ` | direct |
| `/ð/` | `ð` | `ð` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/ɣ/` | `ɣ` | `ɣ` | direct |

### gujarati

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### hebrew

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/ʔ/` | `ʔ` | `??`, `ʔ` | direct |
| `/h/` | `h` | `h` | direct |

### hindi

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### hmong

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |

### hungarian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### icelandic

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/θ/` | `θ` | `θ` | direct |
| `/ð/` | `ð` | `ð` | direct |
| `/r/` | `r` | `r` | direct |
| `/h/` | `h` | `h` | direct |

### igbo

- (no L1 phonemes listed)

### indonesian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### italian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/ɲ/` | `ɲ` | `ɲ` | direct |
| `/ʎ/` | `ʎ` | `ʎ` | direct |

### japanese

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |
| `/ts/` | `ts` | `ts` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/dʑ/` | `dʑ` | `dʑ` | direct |

### kannada

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### khmer

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/h/` | `h` | `h` | direct |

### korean

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/h/` | `h` | `h` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |

### lao

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |

### macedonian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |
| `/ts/` | `ts` | `ts` | direct |

### malay

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### malayalam

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### mandarin

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/ʈʂ/` | `ʈʂ` | `ts.` | notation-only |

### marathi

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### mongolian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/h/` | `h` | `h` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |

### nepali

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### norwegian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### pashto

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/q/` | `q` | `q` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |

### persian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/q/` | `q` | `q` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/r/` | `r` | `r` | direct |
| `/h/` | `h` | `h` | direct |

### polish

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |

### portuguese

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### punjabi

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### quechua

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʼ/` | `pʼ` | — | phonological |
| `/tʼ/` | `tʼ` | — | phonological |
| `/kʼ/` | `kʼ` | — | phonological |
| `/q/` | `q` | `q` | direct |
| `/h/` | `h` | `h` | direct |

### romanian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |
| `/ʃ/` | `ʃ` | `S`, `ʃ` | direct |
| `/ts/` | `ts` | `ts` | direct |

### russian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |

### serbian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/dʒ/` | `dʒ` | `dZ`, `dʒ` | direct |

### sinhala

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/h/` | `h` | `h` | direct |

### slovak

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |

### slovenian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |

### spanish

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/ɲ/` | `ɲ` | `ɲ` | direct |
| `/x/` | `x` | `X`, `x` | direct |

### swahili

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### swedish

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### tagalog

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |

### tamil

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### telugu

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/h/` | `h` | `h` | direct |

### thai

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/h/` | `h` | `h` | direct |

### tibetan

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/tʃ/` | `tʃ` | `tS`, `tʃ` | direct |
| `/h/` | `h` | `h` | direct |

### tigrinya

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʼ/` | `pʼ` | — | phonological |
| `/tʼ/` | `tʼ` | — | phonological |
| `/kʼ/` | `kʼ` | — | phonological |
| `/q/` | `q` | `q` | direct |
| `/h/` | `h` | `h` | direct |
| `/ʔ/` | `ʔ` | `??`, `ʔ` | direct |

### turkish

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/h/` | `h` | `h` | direct |

### ukrainian

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |

### urdu

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/pʰ/` | `pʰ` | `ph`, `pʰ` | direct |
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/ʈ/` | `ʈ` | `t^`, `ʈ` | direct |
| `/ɖ/` | `ɖ` | `d^`, `ɖ` | direct |
| `/r/` | `r` | `r` | direct |
| `/ɾ/` | `ɾ` | `ɾ` | direct |
| `/q/` | `q` | `q` | direct |
| `/x/` | `x` | `X`, `x` | direct |
| `/h/` | `h` | `h` | direct |

### vietnamese

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/tʰ/` | `tʰ` | `th`, `tʰ` | direct |
| `/kʰ/` | `kʰ` | `kh`, `kʰ` | direct |
| `/tɕ/` | `tɕ` | `tɕ` | direct |
| `/ɲ/` | `ɲ` | `ɲ` | direct |
| `/h/` | `h` | `h` | direct |

### welsh

| token | IPA | model vocab symbol(s) | category |
|---|---|---|---|
| `/r/` | `r` | `r` | direct |
| `/x/` | `x` | `X`, `x` | direct |

### yoruba

- (no L1 phonemes listed)

### zulu

- (no L1 phonemes listed)

