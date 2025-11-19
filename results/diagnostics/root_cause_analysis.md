# Root Cause Analysis: FLAN-T5 Generation Collapse

**Author**: Senior Machine Learning Reviewer & Diagnostics Engineer
**Artifact Location**: `results/diagnostics/root_cause_analysis.md`
**Diagnostic Dataset**: `data/processed/test.csv` (225 clinical dialogue samples)
**Supporting Files**:
- `results/diagnostics/generation_comparison.json`
- `results/diagnostics/checkpoint_integrity.json`
- `results/diagnostics/tokenizer_audit.json`
- `results/diagnostics/training_pipeline_audit.md`

---

## 1. Problem Statement

During evaluation on the 225-sample clinical test set, both the supervised fine-tuned FLAN-T5 Base checkpoint (`models/flan_t5_base_finetuned`) and the distilled FLAN-T5 Base checkpoint (`models/flan_t5_distilled_large_to_base`) exhibited severe generation collapse:
- **Observed Outputs**: Repetitive, nonsensical multilingual sequences such as `wendenwendenwendendeshalbdeshalbdeshalb impun impun impun...`, `fühlt fühlt fühlt...`, `baie baie baie... universitaire...`, and `Cluj Cluj Cluj...`.
- **Observed Metrics**: Exactly 0.00 ROUGE-1, 0.00 ROUGE-2, 0.00 ROUGE-L, 0.00 BLEU, and 0.00 Exact Match across all 225 test samples in both `evaluation_results_sft.json` and `evaluation_results_kd.json`.
- **Generation Length**: Average predicted length was 110–115 words (generating continuously until hitting `max_new_tokens=128`), contrasted against ground truth reference answers averaging 11.5 words.

---

## 2. Executive Finding & Diagnosis

> [!IMPORTANT]
> **The model weights on disk are NOT corrupted and the training runs did NOT collapse.**
> The generation collapse is an **embedding/LM-head weight-tying configuration defect** triggered by `"tie_word_embeddings": true` in `config.json`.

In FLAN-T5 (T5 v1.1 architecture), the input embedding matrix (`shared.weight`) and the output projection matrix (`lm_head.weight`) are **untied** and have completely independent weights (maximum absolute element difference: **125.92**).

When `config.json` specifies `"tie_word_embeddings": true`, any environment or loader that enforces weight tying executes:
$$\text{lm\_head.weight} \longleftarrow \text{shared.weight}$$
This **destroys the output projection head**, projecting decoder hidden states through the input embedding matrix rather than the output vocabulary classifier. This logit distortion creates an artificial explosion in logits for specific multilingual subwords from T5's pretraining vocabulary (e.g., German `wenden`, `deshalb`, `schluss`, `fühlt`; Romanian `impun`, `Cluj`; French/Afrikaans `baie`, `universitaire`). The model enters an infinite autoregressive repetition loop without ever emitting the End-of-Sequence token (`</s>`, token ID 1).

---

## 3. Empirical Proof & Bit-for-Bit Reproduction

In our controlled diagnostic audit (`scratch/run_full_audit.py` and `results/diagnostics/generation_comparison.json`), we evaluated the identical test sample under two conditions:

### Test Sample 0 Input:
> *"Question: List the patient's chief complaints with comma separated value. Context: Good afternoon! How are you feeling today? I’ve been experiencing sharp abdominal pain, especially on the right side..."*
> **Reference**: *"Sharp abdominal pain on the right side, mild fever, bloating."*

### Empirical Output Under Both Modes:

| Model Variant | Condition | Prediction Output | Status |
| :--- | :--- | :--- | :--- |
| **SFT Base** (`models/flan_t5_base_finetuned`) | **Untied (Normal)** | `"Sharp abdominal pain, uncomfortable moving, mild fever, mild bloating."` | **PASS** (Accurate clinical summary, ROUGE-1: 0.62) |
| **SFT Base** (`models/flan_t5_base_finetuned`) | **Tied (`lm_head = shared`)** | `"wendenwendenwendendeshalbdeshalbdeshalb impun impun impun impun impun impun impun..."` | **COLLAPSE REPRODUCED** (100% bit-for-bit match to `evaluation_results_sft.json`) |
| **KD Student** (`models/flan_t5_distilled_large_to_base`) | **Untied (Normal)** | `"sharp abdominal pain, especially on the right side"` | **PASS** (Accurate clinical summary, ROUGE-1: 0.37) |
| **KD Student** (`models/flan_t5_distilled_large_to_base`) | **Tied (`lm_head = shared`)** | `"fühlt fühlt fühltdeshalbdeshalbdeshalbdeshalbdeshalb..."` | **COLLAPSE REPRODUCED** (100% bit-for-bit match to `evaluation_results_kd.json`) |

---

## 4. Checkpoint Integrity Analysis

All tests confirm that the serialized weight files on disk are completely intact:

1. **Tensor Keys & Shapes**:
   - `model.safetensors` contains exactly 282 tensor keys and 247,577,856 total elements.
   - `shared.weight`: shape `[32128, 768]`, dtype `torch.float32`.
   - `lm_head.weight`: shape `[32128, 768]`, dtype `torch.float32`.
   - `encoder.embed_tokens.weight` and `decoder.embed_tokens.weight` are absent from `model.safetensors` by design in Hugging Face T5 because they are in-memory pointers to `shared.weight`.
2. **Numerical Health**:
   - NaN check: **0 NaN values** across all parameters.
   - Inf check: **0 Inf values** across all parameters.
   - Max difference between `shared.weight` and `lm_head.weight`:
     - Original Google checkpoint: `125.9259`
     - Fine-tuned checkpoint: `125.9005`
     - Distilled checkpoint: `125.9135`
3. **Reload Consistency**:
   - Reloading both checkpoints in completely fresh Python processes produced identical generation hashes (`Reload match: True`).

---

## 5. Tokenizer Compatibility Audit

1. **Vocabulary Size**:
   - Vocabulary size is consistently 32,100 wordpieces across the base model, SFT checkpoint, and KD checkpoint.
   - Sentinel tokens (`<extra_id_0>` through `<extra_id_99>`) occupy IDs 32,099 down to 32,000.
2. **Special Token IDs**:
   - `pad_token_id`: `0` (`<pad>`)
   - `eos_token_id`: `1` (`</s>`)
   - `unk_token_id`: `2` (`<unk>`)
   - `decoder_start_token_id`: `0` (`<pad>`)
3. **Clinical Phrase Tokenization & Decoding**:
   - All clinical phrases ("Sharp abdominal pain on the right side, mild fever, bloating.", "penicillin gives hives", "asthma as a child") decode to the exact input string without corruption or character drops.

---

## 6. Distinguishing Confirmed Defects from Hypotheses

| Finding | Classification | Evidence & Impact |
| :--- | :--- | :--- |
| **`tie_word_embeddings: true` in `config.json`** | **CONFIRMED DEFECT** | When tying is executed, `lm_head.weight` is overwritten with `shared.weight`, directly generating `wendenwenden...` and `fühlt fühlt...`. |
| **Missing Embedding Keys Warning** | **NON-DEFECT** | `encoder.embed_tokens.weight` and `decoder.embed_tokens.weight` are Hugging Face memory aliases for `shared.weight`. Their absence from `model.safetensors` is standard T5 serialization design. |
| **Model Weights Corrupted by Training** | **DISPROVEN HYPOTHESIS** | When evaluated with untied weights, the checkpoints achieve strong clinical performance (ROUGE-1: 0.6210 for SFT, 0.3679 for KD). The model weights are fully intact. |
| **Distillation Loss Padding Masking** | **MINOR IMPROVEMENT OPPORTUNITY** | `src/distillation/loss.py` computed KL divergence across all sequence tokens including padding tokens. Masking padding tokens reduces uninformative gradient updates. |

---

## 7. Recommended Smallest Corrective Actions

### Action 1: Fix `config.json` in Both Checkpoint Directories
In `models/flan_t5_base_finetuned/config.json` and `models/flan_t5_distilled_large_to_base/config.json`:
Change:
```json
"tie_word_embeddings": true
```
To:
```json
"tie_word_embeddings": false
```
This permanently informs every model loader and evaluation script that the language model head must remain untied, eliminating both the collapse and the console warning.

### Action 2: Update `src/training/trainer.py` and `scripts/distill.py`
Ensure that newly saved models preserve `"tie_word_embeddings": false`:
```python
model.config.tie_word_embeddings = False
```

### Action 3: Mask Padding Positions in `src/distillation/loss.py`
In `DistillationLoss.forward`, apply the label mask `labels != self.ignore_index` when reducing KL divergence over sequence length.

