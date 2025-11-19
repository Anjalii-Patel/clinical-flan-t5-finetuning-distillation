# Experimental Results & Benchmarks

This directory contains experimental records, benchmark logs, and evaluation reports.

## 1. Verified Experimental Records

In accordance with scientific integrity guidelines, this project strictly documents verified experimental outputs from the implementation:

### 1.1 Verified Model Inference Smoke Test
- **Input Dialogue**:
  ```text
  Context: what's bothering you? i've had a sore throat and trouble swallowing for a few days.
  have you had a fever? yes, a mild one. it sounds like a throat infection. i'll prescribe antibiotics
  and a pain reliever. drink warm fluids and rest. alright, thank you. you; good morning. good morning.
  i've been having a persistent headache for the past week. i see. any other symptoms? sometimes i feel dizzy.
  have you been sleeping well and staying hydrated? not really. drink plenty of water and get rest. thank you.
  ```
- **Query**: `"List the patient's chief complaints in comma separated values."`
- **Model Output**: `"sore throat"` (Verified from notebook execution checkpoint)

### 1.2 Empirical Optimization Benchmark on FLAN-T5
Measured on 25 clinical test samples (`data/processed/test.csv`) on an Intel 12-core CPU with PyTorch 2.12.1+cpu (zero-shot `google/flan-t5-small`, `num_beams=2`, `max_new_tokens=32`):

| Model Variant | Architectural Params | Serialized Disk Size | In-Memory Float RAM | Sparsity | Average Latency (CPU) | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original FLAN-T5-small (FP32)** | 76.96M | 293.66 MB | 293.58 MB | 0.0% | 115.96 ms | 0.0438 | 0.0131 | 0.0420 | 0.0059 |
| **Pruned FLAN-T5-small (20% L1 Global)** | 76.96M | 293.66 MB | 293.58 MB | **20.0%** | 217.18 ms (+87.3%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Dynamic INT8 Quantized FLAN-T5-small** | 76.96M (16.47M float)* | **120.71 MB** (-58.9%) | **62.83 MB** (-78.6%) | 0.0% | 193.64 ms (+67.0%) | 0.0415 (94.8% ret.) | 0.0024 (18.3% ret.) | 0.0415 (98.8% ret.) | 0.0043 (72.9% ret.) |

*\*Note on INT8 Parameters and Latency*:
1. **Parameter Artifact**: PyTorch dynamic quantization transforms `nn.Linear` layers into C++ `_packed_params` objects, which are excluded from Python `.parameters()`, leaving 16.47M unquantized embeddings/LayerNorm parameters. The true underlying architecture still contains 76.96M total parameters.
2. **CPU Latency Overhead**: Dynamic INT8 quantization reduces disk footprint by 58.9% and in-memory parameter tensor allocation by 78.6%, but increases per-sample latency by +67.0% on CPU due to runtime activation quantization and dynamic dequantization overhead in eager PyTorch execution without an optimized runtime engine (e.g., ONNX Runtime or TensorRT).
3. **Pruning Without Fine-Tuning**: Zero-shot 20% L1 unstructured pruning collapses generative output (ROUGE 0.0000), illustrating that unstructured weight zeroing breaks autoregressive generation without post-pruning fine-tuning or recovery calibration.

*Full JSON results*: `results/optimization_empirical_benchmark.json`

### 1.3 Grouped Dialogue Split Audit
To evaluate generalization to unseen patients, a dialogue-grouped split was generated and audited (`results/grouped_split_audit.json`):

| Split | Dialogues | QA Pairs | Context Overlap | Questions / Dialogue | Class Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** (`train_grouped.csv`) | 105 (70.0%) | 1,050 | 0 dialogues (0.0%) | Exactly 10 | 105 per question type |
| **Validation** (`val_grouped.csv`) | 22 (14.7%) | 220 | 0 dialogues (0.0%) | Exactly 10 | 22 per question type |
| **Test** (`test_grouped.csv`) | 23 (15.3%) | 230 | 0 dialogues (0.0%) | Exactly 10 | 23 per question type |
| **Total** | 150 (100%) | 1,500 | 0 dialogues (0.0%) | Exactly 10 | 150 per question type |

Unlike the sample-level split (`data/processed/`) where 100% of test dialogues had counterpart questions in the training set, `data/grouped/` strictly isolates clinical consultation transcripts across splits.

### 1.4 Verified Supervised Fine-Tuning Execution (`models/flan_t5_base_finetuned`)
- **Architecture**: `google/flan-t5-base` (~250M parameters)
- **Hardware**: NVIDIA RTX A4500 (20 GB VRAM)
- **Training Config**: 3 Epochs, 789 global steps, AdamW, LR: $5 \times 10^{-5}$, Effective Batch Size: 4 (batch size 2, gradient accumulation 2)
- **Runtime**: 409.4 seconds (6 min 49s), throughput: 7.69 samples/s
- **Loss Progression**: Initial Train Loss: 6.051 $\rightarrow$ Final Train Loss: 2.328; Validation Loss: $1.233 \rightarrow \mathbf{0.7669}$ (Best Checkpoint: step 789)
- **Saved Weights**: `models/flan_t5_base_finetuned/model.safetensors` (990 MB)

### 1.5 Verified Online Knowledge Distillation Execution (`models/flan_t5_distilled_large_to_base`)
- **Teacher**: `google/flan-t5-large` (780M parameters, frozen)
- **Student**: `google/flan-t5-base` (250M parameters, trainable)
- **Hardware**: NVIDIA RTX A4500 (20 GB VRAM)
- **Distillation Config**: 3 Epochs, 789 global steps, $\tau = 2.0$, $\alpha = 0.6$, LR: $3 \times 10^{-5}$, Effective Batch Size: 4
- **Runtime**: 617.0 seconds (10 min 17s), throughput: 5.11 samples/s
- **Loss Progression**: Initial Distillation Loss: 57.69 $\rightarrow$ Final Train Loss: 32.67; Validation Loss: $14.32 \rightarrow \mathbf{10.35}$ (Best Checkpoint: step 789)
- **Saved Weights**: `models/flan_t5_distilled_large_to_base/model.safetensors` (990 MB)

### 1.6 Model Architectures & Parameter Footprints

| Model | Architecture | Parameters | Precision | Parameter Size (Disk) | Target Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FLAN-T5-Base** | Encoder-Decoder | ~250M | FP32 / FP16 | ~990 MB | Target Student Model |
| **FLAN-T5-Large** | Encoder-Decoder | ~780M | FP32 / FP16 | ~3.13 GB | Teacher / Intermediate Model |
| **FLAN-T5-XL** | Encoder-Decoder | ~2.85B | FP32 / FP16 | ~11.4 GB | High-Capacity Teacher Model |
| **FLAN-T5-Base (Pruned 20%)** | Linear L1 Pruned | ~250M (20% zeroed) | FP32 / Sparse | ~990 MB (Sparse) | Optimized Deployment |
| **FLAN-T5-Base (INT8)** | Dynamic Quantized | ~250M | INT8 Linear | ~260 MB | CPU Edge Inference |

## 2. Knowledge Distillation Technical Findings

### Finding: Memory Bottleneck of Precomputed Teacher Logits
During initial experiments (`knowledge-d.ipynb`), generating soft targets offline by computing and saving full-vocabulary teacher logits across 1,000 samples resulted in a fatal `MemoryError`:

$$\text{Memory} = B \times L \times V \times 4\text{ bytes} = 1000 \times 512 \times 32{,}128 \times 4 \approx 65.8\text{ GB}$$

Converting such arrays to Python lists for Hugging Face `Dataset.add_column` immediately exhausts host RAM.

### Architectural Solution: Online Distillation
The codebase resolves this via `Seq2SeqDistillationTrainer` in `src/distillation/distillation_trainer.py`. Teacher logits are computed on-the-fly per mini-batch with `torch.no_grad()`, immediate KL divergence calculation, and automatic gradient clearing.

## 3. How to Run Evaluations and Generate New Results

### Run Standard Test Set Evaluation
```bash
python scripts/evaluate.py --config configs/evaluate.yaml
```
Output metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Exact Match, Latency) will be saved to `results/evaluation_results.json`.

### Run 10-Consultation Clinical Benchmark
```bash
python scripts/benchmark_inference.py --model_name_or_path google/flan-t5-base --output_file results/consultation_benchmark.txt
```
