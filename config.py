"""Configuration management for the Corporate LLM system."""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google Gemini API
    google_api_key: str = Field(..., env='GOOGLE_API_KEY')
    
    # Embedding model
    embedding_model: str = Field(
        # Multilingual model (supports 50+ languages incl. Russian & English). Was 'all-MiniLM-L6-v2'.
        default='paraphrase-multilingual-MiniLM-L12-v2',
        env='EMBEDDING_MODEL'
    )
    
    # Document processing
    chunk_size: int = Field(default=1000, env='CHUNK_SIZE')
    chunk_overlap: int = Field(default=200, env='CHUNK_OVERLAP')
    
    # Storage paths
    vector_db_path: str = Field(default='./vector_db', env='VECTOR_DB_PATH')
    documents_path: str = Field(default='./documents', env='DOCUMENTS_PATH')
    
    # Retrieval settings
    top_k_results: int = Field(default=3, env='TOP_K_RESULTS')
    
    # Gemini model settings
    gemini_model: str = Field(default='gemini-2.0-flash', env='GEMINI_MODEL')
    temperature: float = Field(default=0.3, env='TEMPERATURE')
    max_output_tokens: int = Field(default=2048, env='MAX_OUTPUT_TOKENS')
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


# Global settings instance
settings = Settings()
