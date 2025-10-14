"""Main Corporate LLM system integrating all components."""
from typing import List, Dict, Optional
from document_loader import DocumentLoader
from vector_store import VectorStore
from llm_engine import LLMEngine
from config import settings


class CorporateLLM:
    """Corporate LLM system for question answering on internal documents."""
    
    def __init__(self):
        """Initialize the Corporate LLM system."""
        # Initialize components
        self.document_loader = DocumentLoader(settings.documents_path)
        self.vector_store = VectorStore(
            db_path=settings.vector_db_path,
            embedding_model_name=settings.embedding_model
        )
        self.llm_engine = LLMEngine(
            api_key=settings.google_api_key,
            model_name=settings.gemini_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens
        )
    
    def index_documents(self, clear_existing: bool = False):
        """Load and index all documents from the documents directory.
        
        Args:
            clear_existing: Whether to clear existing indexed documents
        """
        if clear_existing:
            self.vector_store.clear()
        
        print("Loading documents...")
        documents = self.document_loader.load_all_documents()
        
        if not documents:
            print("No documents found to index.")
            return
        
        print(f"Indexing {len(documents)} documents...")
        self.vector_store.add_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        stats = self.vector_store.get_stats()
        print(f"Indexing complete. Total chunks in database: {stats['total_chunks']}")
    
    def ask(self, question: str, top_k: Optional[int] = None,
            use_context: bool = True) -> Dict[str, any]:
        """Ask a question and get an answer based on indexed documents.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve (uses settings default if None)
            use_context: Whether to use retrieved context for answering
            
        Returns:
            Dictionary with 'answer', 'sources', and 'context_chunks'
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        # Retrieve relevant context
        context_chunks = []
        if use_context:
            print(f"Searching for relevant context...")
            context_chunks = self.vector_store.search(question, top_k=top_k)
            print(f"Found {len(context_chunks)} relevant chunks")
        
        # Generate answer
        print("Generating answer...")
        answer = self.llm_engine.generate_answer(
            query=question,
            context_chunks=context_chunks,
            use_context=use_context
        )
        
        # Extract unique sources
        sources = []
        if context_chunks:
            seen_sources = set()
            for chunk in context_chunks:
                filename = chunk['metadata'].get('filename', 'Unknown')
                if filename not in seen_sources:
                    sources.append({
                        'filename': filename,
                        'filepath': chunk['metadata'].get('filepath', '')
                    })
                    seen_sources.add(filename)
        
        return {
            'answer': answer,
            'sources': sources,
            'context_chunks': context_chunks
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        return self.vector_store.get_stats()
    
    def clear_index(self):
        """Clear all indexed documents."""
        self.vector_store.clear()
