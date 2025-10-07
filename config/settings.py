import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class RAGConfig:
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME: str = "rag-qa-index"
    
    # LLM Configuration
    LLM_MODEL_NAME: str = "microsoft/DialoGPT-large"  # Can be changed to Llama when available
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4096
    
    # Database Configuration
    DEFAULT_DB_TYPE: str = "sqlite"
    DB_CONNECTION_TIMEOUT: int = 30
    
    # UI Configuration
    APP_TITLE: str = "RAG-Powered Q&A System"
    PAGE_ICON: str = "🤖"
    LAYOUT: str = "wide"
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = 200
    ALLOWED_EXTENSIONS: list = None
    
    # Search Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    RERANK_TOP_K: int = 3
    
    # Visualization Configuration
    PLOT_THEME: str = "plotly_white"
    FIGURE_WIDTH: int = 800
    FIGURE_HEIGHT: int = 600
    
    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls', '.txt']
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables"""
        return cls()
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if not self.PINECONE_API_KEY:
            issues.append("PINECONE_API_KEY not set")
        
        if self.CHUNK_SIZE <= 0:
            issues.append("CHUNK_SIZE must be positive")
            
        if self.TOP_K_RESULTS <= 0:
            issues.append("TOP_K_RESULTS must be positive")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

# Global configuration instance
config = RAGConfig.from_env()