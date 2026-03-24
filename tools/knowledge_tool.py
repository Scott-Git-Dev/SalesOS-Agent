"""
Knowledge Base Tool
Searches local documents using RAG (Retrieval Augmented Generation)
"""
import os
import traceback
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import config
import sys
from pathlib import Path
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, DEBUG_MODE, TOP_K, MIN_RAG_SCORE, CHROMA_COLLECTION_METADATA

sys.path.append(str(Path(__file__).parent.parent))

# Global vectorstore (initialized once)
_VECTORSTORE = None



def _init_vectorstore():
    """Initialize vectorstore if not already done, and return it."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _VECTORSTORE = Chroma(
            persist_directory=str(CHROMA_DB_PATH), 
            embedding_function=embeddings,
            collection_metadata=CHROMA_COLLECTION_METADATA
        )
    return _VECTORSTORE



@tool
def search_local_docs(query: str) -> str:
    """
    Search local knowledge base for relevant information.
    
    Use this tool for questions about:
    - Company policies, procedures, strategies
    - Sales playbooks, competitive intelligence
    - Product information and positioning
    - Customer success stories and case studies
    - Goals, targets, quotas (NOT in sales database)
    
    This tool searches through all documents in the knowledge base using
    semantic similarity (RAG with vector embeddings).
    
    Args:
        query: Search query for the knowledge base
    
    Returns:
        Relevant document excerpts or error message
    """
    
    try:
        # Check if database exists
        if not os.path.exists(CHROMA_DB_PATH):
            return (
                f"Knowledge base not found at {CHROMA_DB_PATH}. "
                f"Please run setup_knowledge_base.py first."
            )
        
        if DEBUG_MODE:
            print(f"[RAG SEARCH] Query: '{query}'") 
        
        # Initialize vectorstore
        vectorstore = _init_vectorstore()
        

        if DEBUG_MODE:
             # Check collection
            try:
                doc_count = vectorstore._collection.count()
                print(f"[RAG SEARCH] {doc_count} documents in collection")
            except Exception as e:
                print(f"[RAG SEARCH] Error checking document count: {e}")
        

        # Search for relevant documents
        docs = vectorstore.similarity_search_with_score(query, k=TOP_K)

        # Filter out low-relevance documents; lower score means more relevant (distance), so keep those <= threshold
        filtered_docs = [(doc, score) for doc, score in docs if score <= MIN_RAG_SCORE]

        if not filtered_docs:
            if DEBUG_MODE:
                print(f"[RAG SEARCH] No docs passed min relevance filter")
            return "No relevant documents found in knowledge base for this query."
        
        # Format results
        results = []
        for i, (doc, score) in enumerate(filtered_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            source_name = Path(source).name if source != 'Unknown' else source
            content = doc.page_content.strip()
            if DEBUG_MODE:
                    print(f"[Document {i}] (Score: {score:.2f}) From {source_name}:\n{content}")
            results.append(f"[Document {i}] (Score: {score:.2f}) From {source_name}:\n{content}")

        return "\n\n---\n\n".join(results)
    
    except Exception as e:
        if DEBUG_MODE:
            traceback.print_exc()
        return f"Error searching knowledge base: {str(e)}"
