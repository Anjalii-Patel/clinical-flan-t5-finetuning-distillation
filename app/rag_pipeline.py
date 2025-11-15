"""
Retrieval-Augmented Generation (RAG) prototype for Clinical Contexts using SentenceTransformers and FLAN-T5.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag")


class ClinicalRAGPipeline:
    """
    RAG pipeline combining dense vector retrieval with FLAN-T5 conditional generation.
    """

    def __init__(
        self,
        generator_model_name: str = "google/flan-t5-base",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Initializing embedding model: %s", embedding_model_name)
        self.embedder = SentenceTransformer(embedding_model_name)

        logger.info("Initializing generator model: %s", generator_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name).to(self.device)
        self.generator.eval()

        self.documents: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: List[str]) -> None:
        """Embed and index a list of text passages or consultation transcripts."""
        self.documents = [d.strip() for d in documents if d.strip()]
        logger.info("Encoding %d documents into vector space...", len(self.documents))
        embeddings = self.embedder.encode(self.documents, convert_to_tensor=False)
        self.doc_embeddings = np.array(embeddings, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """Dense cosine similarity retrieval over indexed documents."""
        if not self.documents or self.doc_embeddings is None:
            raise ValueError("No documents have been indexed yet. Call index_documents() first.")

        query_emb = self.embedder.encode([query], convert_to_tensor=False)
        query_vec = np.array(query_emb, dtype=np.float32)

        # Cosine similarity
        norm_docs = self.doc_embeddings / np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norm_query = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        similarities = np.dot(norm_docs, norm_query.T).squeeze()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.documents[idx] for idx in top_indices]

    def answer(self, query: str, top_k: int = 1) -> str:
        """Retrieve relevant context and generate grounded answer."""
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n".join(retrieved)

        prompt = f"Answer the clinical question strictly based on the context.\nContext: {context}\nQuestion: {query}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        with torch.inference_mode():
            outputs = self.generator.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                num_beams=2,
                early_stopping=True,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical RAG pipeline demo.")
    parser.add_argument("--query", type=str, default="What is the past medical history of the patient?", help="Clinical query.")
    args = parser.parse_args()

    sample_dialogues = [
        "What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache. Past medical history: migraines managed with ibuprofen. Surgeries: cyst removal 3 years ago.",
        "What brings you in today? Persistent dry cough for 3 weeks following a cold. No fever. Mild asthma in childhood. Father has high cholesterol.",
    ]

    rag = ClinicalRAGPipeline()
    rag.index_documents(sample_dialogues)
    ans = rag.answer(args.query)
    print(f"\nQuery: {args.query}")
    print(f"RAG Grounded Answer: {ans}\n")

