# Legacy Experiments Archive

This directory preserves the original exploratory scripts, prototypes, and notebooks created during early research iterations. None of the author's original exploratory work has been deleted.

## File Inventory & Migration Mapping

| Original File | Role / Experiment | Status | Modern Replacement |
| :--- | :--- | :--- | :--- |
| `ANJALI_DATASET.py` | Initial script with 50 doctor-patient dialogues and 10 questions per dialogue. | Functional data creation script | `src/data/generator.py` |
| `flan-base_finetune.py` | FLAN-T5-base fine-tuning script with hardcoded training arguments. | Functional exploratory script | `src/training/` & `scripts/train.py` |
| `new_with1500.py` | Fine-tuning script on 1,500 samples with PyTorch Dataset wrapper. | Functional exploratory script | `src/training/` & `scripts/train.py` |
| `fine-tune(flan-t5).ipynb` | Interactive notebook for fine-tuning FLAN-T5-base on medical QA. | Verified execution; output: `"sore throat"` | `notebooks/01_flan_t5_finetuning.ipynb` |
| `know-dist.py` | Knowledge distillation with PyTorch loop and initial trainer experiments. | Experimental (demonstrated KL logit inversion and loss variations) | `src/distillation/` & `scripts/distill.py` |
| `XL_LARGE.py` | Distillation script transferring knowledge from FLAN-T5-XL to FLAN-T5-large. | Functional distillation script with checkpoint resumption | `src/distillation/` & `scripts/distill.py` |
| `knowledge-d.ipynb` | Distillation notebook exploring offline teacher logits caching vs online loops. | Hit `MemoryError` in cell 9 (65GB logits footprint); prompted online distillation design | `notebooks/02_knowledge_distillation.ipynb` |
| `flan-t5.py` | Inference script querying 10 clinical sections with local Windows paths. | Functional exploratory script | `src/inference/pipeline.py` |
| `TESTING_KD.py` | Inference latency benchmarking over 10 dialogues x 10 questions. | Functional benchmarking script | `scripts/benchmark_inference.py` |
| `multi_processing.py` | Multiprocessing experiments (process per question vs shared model lock). | Proof-of-concept for parallel extraction | `src/inference/parallel.py` |
| `gradio_demo.py` | Initial Gradio demo extracting 10 medical categories. | Functional UI prototype | `app/gradio_app.py` |
| `Flan-T5-XL(RAG).py` | ChromaDB + SentenceTransformers RAG prototype. | Functional RAG prototype | `app/rag_pipeline.py` |
| `final_deploy.py` | Multimodal prototype (Whisper-large audio transcription + BioMistral + Gradio SSL). | Prototype requiring specific GPU and SSL certs | Preserved in legacy archive; modularized components in `app/` |

