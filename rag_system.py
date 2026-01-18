"""
Ultra Simple RAG System - Just Works!
"""

import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from typing import List, Dict


class UltraSimpleRAG:
    def __init__(self, data_dir: str = "financial_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Storage
        self.documents = []
        self.embeddings = None
        
        # Create sample data and load
        self.create_sample_data()
        self.load_documents()
    
    def create_sample_data(self):
        """Create sample financial data"""
        sample_data = [
            {
                "title": "Tesla Reports Strong Q4 Earnings",
                "content": "Tesla Inc. reported better-than-expected fourth-quarter earnings, with revenue reaching $25.2 billion. The electric vehicle maker delivered 484,507 vehicles in Q4, beating analyst estimates. CEO Elon Musk highlighted the company's progress in autonomous driving technology and energy storage solutions.",
                "company": "Tesla Inc.",
                "date": "2024-01-24"
            },
            {
                "title": "Apple's Services Revenue Hits Record High", 
                "content": "Apple Inc. announced record-breaking services revenue of $23.1 billion in the latest quarter. The growth was driven by App Store sales, iCloud subscriptions, and Apple Pay transactions. The company also reported strong iPhone sales despite market headwinds.",
                "company": "Apple Inc.",
                "date": "2024-01-25"
            },
            {
                "title": "Federal Reserve Signals Potential Rate Cuts",
                "content": "The Federal Reserve indicated it may consider interest rate cuts in 2024 if inflation continues to decline. Fed Chair Jerome Powell stated that the central bank is closely monitoring economic indicators and remains committed to achieving price stability.",
                "company": "Federal Reserve",
                "date": "2024-01-26"
            },
            {
                "title": "Microsoft Cloud Revenue Surges",
                "content": "Microsoft Corporation reported exceptional growth in its cloud computing division, with Azure revenue increasing by 30% year-over-year. The company's AI initiatives and enterprise solutions continue to drive strong demand.",
                "company": "Microsoft Corporation",
                "date": "2024-01-27"
            }
        ]
        
        # Save sample data
        for i, item in enumerate(sample_data):
            with open(self.data_dir / f"news_{i+1}.json", 'w') as f:
                json.dump(item, f, indent=2)
        
        print(f"Created {len(sample_data)} sample documents")
    
    def load_documents(self):
        """Load and process documents"""
        self.documents = []
        texts = []
        
        # Load all JSON files
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    doc = json.load(f)
                    
                    # Combine text fields
                    text = f"{doc.get('title', '')} {doc.get('content', '')} Company: {doc.get('company', '')}"
                    texts.append(text)
                    self.documents.append(doc)
                    
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        if texts:
            # Create embeddings
            print(f"Creating embeddings for {len(texts)} documents...")
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
            print("✅ Documents loaded and embedded!")
        else:
            print("No documents found!")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if not self.documents or self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(similarities[idx]),
                'text': f"{self.documents[idx].get('title', '')} {self.documents[idx].get('content', '')}"
            })
        
        return results
    
    def get_context(self, query: str, max_length: int = 1000) -> str:
        """Get context for AI agent"""
        results = self.search(query, top_k=2)
        
        if not results:
            return "No relevant financial information found."
        
        context_parts = ["Financial Context:"]
        current_length = len(context_parts[0])
        
        for i, result in enumerate(results, 1):
            doc = result['document']
            entry = f"\n\n{i}. {doc.get('title', 'News')} ({doc.get('date', 'Unknown date')})\n{doc.get('content', '')[:300]}..."
            
            if current_length + len(entry) > max_length:
                break
                
            context_parts.append(entry)
            current_length += len(entry)
        
        return "".join(context_parts)


# Simple global instance
_rag_instance = None

def get_financial_context(query: str) -> str:
    """Simple function to get financial context"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = UltraSimpleRAG()
    
    return _rag_instance.get_context(query)


def create_rag_system() -> UltraSimpleRAG:
    """Create a RAG system instance"""
    return UltraSimpleRAG()


# Test
if __name__ == "__main__":
    print("🚀 Testing Ultra Simple RAG")
    print("=" * 40)
    
    rag = UltraSimpleRAG()
    
    queries = [
        "Tesla earnings performance",
        "Apple revenue growth", 
        "Federal Reserve interest rates",
        "Microsoft cloud business"
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        context = rag.get_context(query)
        print(f"📄 Context: {context[:200]}...")
    
    print("\n✅ Test completed!")