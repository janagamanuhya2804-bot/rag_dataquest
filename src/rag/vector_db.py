import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("spendsense.vectordb")

# Try importing chromadb and sentence_transformers
CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception as e:
    logger.warning(f"chromadb package import failed or SQLite3 version mismatch: {e}. Using fallback memory vector database.")

SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    logger.warning(f"sentence-transformers package import failed: {e}. Using fallback TF-IDF vectors.")

class FallbackVectorDB:
    """A lightweight in-memory vector database with JSON persistence for fallback support."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.documents = []
        self.metadatas = []
        self.ids = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.metadatas = data.get("metadatas", [])
                    self.ids = data.get("ids", [])
            except Exception as e:
                logger.error(f"Error loading fallback vector DB: {e}")

    def _save(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                    "ids": self.ids
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving fallback vector DB: {e}")

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.documents[idx] = doc
                self.metadatas[idx] = meta
            else:
                self.ids.append(doc_id)
                self.documents.append(doc)
                self.metadatas.append(meta)
        self._save()

    def query(self, query_texts: List[str], n_results: int = 5) -> Dict[str, Any]:
        """Simple TF-IDF token overlap + word match query scoring."""
        if not self.documents:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
            
        query = query_texts[0].lower()
        query_words = set(re_split(query))
        
        scores = []
        for doc in self.documents:
            doc_lower = doc.lower()
            doc_words = re_split(doc_lower)
            
            # Intersection score
            intersection = query_words.intersection(doc_words)
            overlap_score = len(intersection)
            
            # Simple keyword matching boosts
            boost = 0.0
            if "total" in query and "total" in doc_lower: boost += 1.0
            if "category" in query and "category" in doc_lower: boost += 1.0
            
            total_score = overlap_score + boost
            scores.append(total_score)
            
        # Get top indices sorted descending
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        res_docs = [self.documents[i] for i in top_indices]
        res_metas = [self.metadatas[i] for i in top_indices]
        res_ids = [self.ids[i] for i in top_indices]
        res_distances = [float(1.0 / (scores[i] + 1.0)) for i in top_indices] # distance is inverse score
        
        return {
            "documents": [res_docs],
            "metadatas": [res_metas],
            "ids": [res_ids],
            "distances": [res_distances]
        }

def re_split(text: str) -> List[str]:
    return [w for w in re.split(r'\W+', text) if w]

import re

class VectorDBManager:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = os.path.join(self.project_dir, "spendsense_chroma")
        
        self.use_chroma = CHROMA_AVAILABLE
        self.model = None
        
        if self.use_chroma:
            try:
                self.client = chromadb.PersistentClient(path=self.db_path)
                # Load local embeddings if available
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")
                    class SentenceTransformerEF:
                        def __init__(self, model):
                            self.model = model
                        def __call__(self, input: List[str]) -> List[List[float]]:
                            embeddings = self.model.encode(input)
                            return [e.tolist() for e in embeddings]
                    self.ef = SentenceTransformerEF(self.model)
                else:
                    self.ef = None # Use ChromaDB's default embedding function (onnx MiniLM)
                
                self.collection = self.client.get_or_create_collection(
                    name="spendsense_financials",
                    embedding_function=self.ef
                )
                logger.info("ChromaDB vector database client initialized successfully.")
            except Exception as e:
                logger.error(f"Error starting ChromaDB. Falling back to local memory DB: {e}")
                self.use_chroma = False
                
        if not self.use_chroma:
            fallback_file = os.path.join(self.project_dir, "fallback_vector_db.json")
            self.fallback_db = FallbackVectorDB(fallback_file)
            logger.info("Fallback local vector database client initialized.")

    def add_transaction(self, tx_id: int, merchant: str, category: str, amount: float, date: str, raw_text: str = ""):
        """Add a transaction record into the vector index."""
        doc = f"Transaction: spent ₹{amount:.2f} at {merchant} on category {category} on date {date}."
        if raw_text:
            doc += f" Raw source text: {raw_text}"
            
        metadata = {
            "type": "transaction",
            "id": tx_id,
            "merchant": merchant,
            "category": category,
            "amount": float(amount),
            "date": date
        }
        
        doc_id = f"tx_{tx_id}"
        
        if self.use_chroma:
            self.collection.upsert(
                ids=[doc_id],
                documents=[doc],
                metadatas=[metadata]
            )
        else:
            self.fallback_db.add([doc_id], [doc], [metadata])

    def add_weekly_summary(self, summary_id: int, start_date: str, end_date: str, total_spend: float, breakdown: str, insights: str):
        """Add a weekly summary into the vector index."""
        doc = (
            f"Weekly Summary from {start_date} to {end_date}.\n"
            f"Total spending: ₹{total_spend:.2f}.\n"
            f"Breakdown: {breakdown}.\n"
            f"Insights: {insights}"
        )
        
        metadata = {
            "type": "weekly_summary",
            "id": summary_id,
            "start_date": start_date,
            "end_date": end_date,
            "total_spend": float(total_spend)
        }
        
        doc_id = f"week_{summary_id}"
        
        if self.use_chroma:
            self.collection.upsert(
                ids=[doc_id],
                documents=[doc],
                metadatas=[metadata]
            )
        else:
            self.fallback_db.add([doc_id], [doc], [metadata])

    def add_monthly_summary(self, summary_id: int, month: str, total_spend: float, breakdown: str, insights: str):
        """Add a monthly summary into the vector index."""
        doc = (
            f"Monthly Summary for {month}.\n"
            f"Total spending: ₹{total_spend:.2f}.\n"
            f"Breakdown: {breakdown}.\n"
            f"Insights: {insights}"
        )
        
        metadata = {
            "type": "monthly_summary",
            "id": summary_id,
            "month": month,
            "total_spend": float(total_spend)
        }
        
        doc_id = f"month_{summary_id}"
        
        if self.use_chroma:
            self.collection.upsert(
                ids=[doc_id],
                documents=[doc],
                metadatas=[metadata]
            )
        else:
            self.fallback_db.add([doc_id], [doc], [metadata])

    def add_financial_insight(self, insight_id: int, type_str: str, insight_text: str):
        """Add a coaching advice or financial insight record."""
        doc = f"Financial Coaching Advice ({type_str}): {insight_text}"
        
        metadata = {
            "type": "financial_insight",
            "id": insight_id,
            "insight_type": type_str
        }
        
        doc_id = f"insight_{insight_id}"
        
        if self.use_chroma:
            self.collection.upsert(
                ids=[doc_id],
                documents=[doc],
                metadatas=[metadata]
            )
        else:
            self.fallback_db.add([doc_id], [doc], [metadata])

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search vector database for relevant transactions, summaries or past advice."""
        if self.use_chroma:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            except Exception as e:
                logger.error(f"Error querying ChromaDB: {e}")
                # Fallback to local search if query errors
                if hasattr(self, 'fallback_db'):
                    results = self.fallback_db.query([query], n_results)
                else:
                    return []
        else:
            results = self.fallback_db.query([query], n_results)
            
        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{} for _ in docs]
            ids = results["ids"][0] if "ids" in results else ["" for _ in docs]
            distances = results["distances"][0] if "distances" in results else [0.0 for _ in docs]
            
            for doc, meta, doc_id, dist in zip(docs, metas, ids, distances):
                retrieved.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                })
        return retrieved
