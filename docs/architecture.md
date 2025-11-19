# Architecture & Technical Design

This document details the system architecture, data processing pipeline, knowledge distillation framework, and inference flows.

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        RAW["Raw Doctor-Patient Dialogues<br/>(1,500 Multi-Turn Encounters)"]
        GEN["src.data.generator<br/>(50 Core Dialogues x 10 Questions)"]
        PREP["src.data.dataset<br/>Prompt Formatting & Tokenization"]
        SPLIT["Processed Splits<br/>Train: 1,050 | Val: 225 | Test: 225"]
    end

    subgraph Modeling & Training
        BASE["google/flan-t5-base<br/>(250M Parameters)"]
        LARGE["google/flan-t5-large<br/>(780M Parameters)"]
        XL["google/flan-t5-xl<br/>(2.85B Parameters)"]

        SFT["Supervised Fine-Tuning<br/>src.training.trainer"]
        KD["Online Knowledge Distillation<br/>src.distillation.distillation_trainer"]
    end

    subgraph Optimization Layer
        PRUNE["Weight Pruning (L1 Magnitude)<br/>src.optimization.pruning"]
        QUANT["Dynamic Quantization (INT8)<br/>src.optimization.quantization"]
    end

    subgraph Inference & Applications
        PIPE["src.inference.pipeline<br/>10-Section Clinical Extractor"]
        UI["app.gradio_app<br/>Web Interface"]
        RAG["app.rag_pipeline<br/>Dense Vector Search + T5"]
        EVAL["src.evaluation.evaluator<br/>ROUGE-1/2/L, BLEU, Latency"]
    end

    RAW --> PREP --> SPLIT
    GEN --> PREP
    SPLIT --> SFT
    SPLIT --> KD

    BASE --> SFT
    LARGE --> SFT
    XL -.->|Teacher Logits| KD
    LARGE -.->|Teacher Logits| KD
    BASE -->|Student Updates| KD

    SFT --> PRUNE --> QUANT
    KD --> PRUNE --> QUANT

    QUANT --> PIPE
    SFT --> PIPE
    PIPE --> UI
    PIPE --> EVAL
    PIPE --> RAG
```

## 2. Sequence-to-Sequence Prompt Formulation

FLAN-T5 is an encoder-decoder model operating in text-to-text space. Input dialogues and queries are structured into a standardized template:

```
Question: {Target Clinical Question} Context: {Doctor-Patient Dialogue Transcript}
```

The decoder predicts the concise clinical answer corresponding to one of the 10 target categories:
- Brief History
- Chief Complaints (comma-separated values)
- Past Medical History
- Hospitalization & Surgical History
- Gynecological History
- Lifestyle & Social Activity
- Family History
- Allergies & Hypersensitivities
- Previous Investigations
- Review of Systems (ROS)

## 3. Online Knowledge Distillation Mechanics

```mermaid
sequenceDiagram
    autonumber
    participant D as Mini-Batch DataLoader
    participant T as Teacher (FLAN-T5-XL / Large)
    participant S as Student (FLAN-T5-Base)
    participant L as DistillationLoss (KL + CE)
    participant O as AdamW Optimizer

    D->>T: Input IDs, Attention Mask (no_grad)
    D->>S: Input IDs, Attention Mask, Labels
    T-->>L: Teacher Logits (P_teacher)
    S-->>L: Student Logits (P_student), Labels
    Note over L: Soft Loss = KL(P_student/T || P_teacher/T) * T^2<br/>Hard Loss = CE(P_student, Labels)<br/>Total = alpha * Soft + (1-alpha) * Hard
    L-->>S: Backward gradients
    O->>S: Optimizer Step (Update Student Weights)
```

## 4. Pruning and Dynamic Quantization

1. **L1 Unstructured Pruning**: Zeroes out weights with the smallest absolute values in all `nn.Linear` layers across both encoder and decoder self-attention and feed-forward blocks.
2. **Dynamic INT8 Quantization**: Converts float32 weights of Linear modules into signed 8-bit integers (`qint8`), reducing serialized disk size by 58.9% and in-memory parameter tensor allocation by 78.6%, noting that CPU inference latency in eager PyTorch execution increases (+67.0% in our benchmark) due to per-step dynamic quantization/dequantization overhead.
