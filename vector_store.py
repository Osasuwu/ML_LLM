"""Vector store management using ChromaDB."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from pathlib import Path


class VectorStore:
    """Manage document embeddings and similarity search using ChromaDB."""
    
    def __init__(self, db_path: str, embedding_model_name: str):
        """Initialize the vector store.
        
        Args:
            db_path: Path to store the ChromaDB database
            embedding_model_name: Name of the sentence transformer model
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="corporate_documents",
            metadata={"description": "Corporate internal documents"}
        )
    
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def add_documents(self, documents: List[Dict[str, str]], 
                     chunk_size: int = 1000, chunk_overlap: int = 200,
                     batch_size: int = 5000):
        """Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'content', 'filename', 'filepath'
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Maximum number of chunks to add in a single batch (default: 5000)
        """
        from document_loader import DocumentLoader
        
        loader = DocumentLoader("")
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        chunk_id = self.collection.count()  # Start from existing count
        
        for doc in documents:
            chunks = loader.chunk_text(
                doc['content'], 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1
        
        if all_chunks:
            total_chunks = len(all_chunks)
            print(f"Processing {total_chunks} chunks from {len(documents)} documents...")
            
            # Process in batches to avoid ChromaDB batch size limit
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                batch_chunks = all_chunks[i:batch_end]
                batch_metadata = all_metadata[i:batch_end]
                batch_ids = all_ids[i:batch_end]
                
                # Generate embeddings for this batch
                print(f"  Embedding batch {i//batch_size + 1}: chunks {i+1}-{batch_end} of {total_chunks}...")
                batch_embeddings = self.embed_text(batch_chunks)
                
                # Add to ChromaDB
                print(f"  Adding batch {i//batch_size + 1} to vector store...")
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_chunks,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            
            print(f"✓ Successfully added {total_chunks} chunks from {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents based on a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing relevant chunks and metadata
        """
        # Generate query embedding
        query_embedding = self.embed_text([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.client.delete_collection("corporate_documents")
        self.collection = self.client.get_or_create_collection(
            name="corporate_documents",
            metadata={"description": "Corporate internal documents"}
        )
        print("Vector store cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }
