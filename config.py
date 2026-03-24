"""
Configuration for Agentic KB
All paths, URLs, and settings in one place
"""
import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

# Base directory (where this config file is)
BASE_DIR = Path(__file__).parent

# Database paths
SALES_DB_PATH = BASE_DIR / "sales_db" / "sales_data.db"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
DOCS_PATH = BASE_DIR / "kb"

# Logs directory
LOGS_PATH = BASE_DIR / "logs"

# ============================================================================
# LLM Configuration
# ============================================================================

# OpenAI API key (required for some models/libraries even if not actually used)
os.environ['OPENAI_API_KEY'] = "not_a_real_key"

# Llama.cpp server configuration
LLAMA_SERVER_URL = "http://localhost:8080/v1/"
MODEL_NAME = "Qwen3.5-35B-A3B-Q3_K_M.gguf"


# LLM parameters
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 6000

# ============================================================================
# Frontend API Configuration
# ============================================================================

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = "8000"

# ============================================================================
# RAG Configuration
# ============================================================================

RAG_AVAILABLE = True  
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Supported file types for document ingestion
SUPPORTED_FILE_TYPES = {
    '.txt': 'TextLoader',
    '.pdf': 'PyPDFLoader',
    '.docx': 'Docx2txtLoader',
    '.md': 'UnstructuredMarkdownLoader',
    '.csv': 'CSVLoader',
    '.json': 'JSONLoader',
    '.html': 'UnstructuredHTMLLoader',
    '.htm': 'UnstructuredHTMLLoader',
}

# Text splitting parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
TOP_K = 8  # Number of relevant chunks to retrieve for RAG
MIN_RAG_SCORE = 0.4 # Keep chunks with distance <= this value (lower = better)

# ============================================================================
# Agent Configuration
# ============================================================================

# Debug mode (shows detailed logs) 
DEBUG_MODE = True

#Print SQL queries to console (can be noisy, so separate flag)
SQL_PRINTING_ENABLED = True

# Conversation settings
DEFAULT_THREAD_ID = "default_user"

# Request timeout (seconds)
REQUEST_TIMEOUT = 120

# ============================================================================
# Display Settings
# ============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════╗                          
║                 SalesOS Agent with Tools                 ║
╚══════════════════════════════════════════════════════════╝
"""

# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config() -> bool:
    """
    Validate configuration at startup.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    warnings = []
    
    # Check database
    if not SALES_DB_PATH.exists():
        errors.append(f"Sales database not found: {SALES_DB_PATH}")
        errors.append("  → Run: python setup_sales_db.py")
    
    # Check documents folder
    if not DOCS_PATH.exists():
        warnings.append(f"Documents folder not found: {DOCS_PATH}")
        warnings.append("  → Create folder and add documents, then run: python setup_knowledge_base.py")
    elif not any(DOCS_PATH.iterdir()):
        warnings.append(f"Documents folder is empty: {DOCS_PATH}")
        warnings.append("  → Add documents to kb/ folder")
    
    # Check ChromaDB
    if RAG_AVAILABLE and not CHROMA_DB_PATH.exists():
        errors.append(f"ChromaDB not initialized: {CHROMA_DB_PATH}")
        errors.append("  → Run: python setup_knowledge_base.py")
    
    # Create logs directory if it doesn't exist
    LOGS_PATH.mkdir(exist_ok=True)
    
    # Print warnings
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    # Print errors
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   {error}")
        print()
        return False
    
    return True
