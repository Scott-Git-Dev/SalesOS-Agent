"""
Knowledge Base Setup
Handles document ingestion and RAG database updates
"""
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader,
    UnstructuredMarkdownLoader, CSVLoader, 
    JSONLoader, UnstructuredHTMLLoader,
)

from config import (CHROMA_COLLECTION_METADATA, CHROMA_DB_PATH, DOCS_PATH, EMBEDDING_MODEL,CHUNK_SIZE, CHUNK_OVERLAP)
from rag_metadata import RAGMetadataManager


# Map file extensions to loader classes
LOADER_MAP = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.md': UnstructuredMarkdownLoader,
    '.csv': CSVLoader,
    '.json': JSONLoader,
    '.html': UnstructuredHTMLLoader,
    '.htm': UnstructuredHTMLLoader,
}


def get_loader_for_file(file_path: str):
    """Get appropriate document loader for a file"""
    ext = Path(file_path).suffix.lower()
    return LOADER_MAP.get(ext)


def setup_knowledge_base(
    documents_path: str = None,
    db_path: str = None,
    force_rebuild: bool = False
):
    """
    Set up or update the knowledge base with incremental change detection.
    
    Args:
        documents_path: Path to documents folder (defaults to config)
        db_path: Path to vector database (defaults to config)
        force_rebuild: If True, delete and rebuild everything
    
    Returns:
        Dictionary with update statistics
    """
    
    # Use defaults from config if not provided
    if documents_path is None:
        documents_path = str(DOCS_PATH)
    if db_path is None:
        db_path = str(CHROMA_DB_PATH)
    
    print("📚 Knowledge Base Setup")
    print("="*60)
    
    # Paths
    metadata_db_path = os.path.join(db_path, "metadata.db")
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize metadata manager
    metadata_mgr = RAGMetadataManager(metadata_db_path)
    
    # Force rebuild if requested
    if force_rebuild:
        print("🔄 Force rebuild requested - deleting existing database...")
        import shutil
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
        metadata_mgr = RAGMetadataManager(metadata_db_path)
    
    # Check documents folder
    if not os.path.exists(documents_path):
        print(f"❌ Documents folder not found: {documents_path}")
        print("Creating folder...")
        os.makedirs(documents_path, exist_ok=True)
        return {"status": "no_documents"}
    
    # Scan for files
    print(f"\n📂 Scanning: {documents_path}")
    current_files = {}
    supported_exts = list(LOADER_MAP.keys())
    
    for ext in supported_exts:
        for file_path in Path(documents_path).rglob(f"*{ext}"):
            file_path_str = str(file_path)
            file_hash = metadata_mgr.compute_file_hash(file_path_str)
            if file_hash:
                current_files[file_path_str] = file_hash
    
    if not current_files:
        print(f"❌ No supported files found")
        print(f"Supported types: {', '.join(supported_exts)}")
        return {"status": "no_documents"}
    
    print(f"✅ Found {len(current_files)} files") 
    
    # Detect changes
    tracked_files = metadata_mgr.get_all_tracked_files()
    new_files = set(current_files.keys()) - tracked_files
    deleted_files = tracked_files - set(current_files.keys())
    
    changed_files = set()
    for file_path in tracked_files & set(current_files.keys()):
        if current_files[file_path] != metadata_mgr.get_stored_hash(file_path):
            changed_files.add(file_path)
    
    # Stats
    stats = {
        "total": len(current_files),
        "new": len(new_files),
        "changed": len(changed_files),
        "deleted": len(deleted_files),
        "unchanged": len(current_files) - len(new_files) - len(changed_files)
    }
    
    print(f"\n📊 Change Detection:")
    print(f"   New: {stats['new']}")
    print(f"   Changed: {stats['changed']}")
    print(f"   Deleted: {stats['deleted']}")
    print(f"   Unchanged: {stats['unchanged']}")
    
    # Early exit if nothing to do
    if not new_files and not changed_files and not deleted_files:
        print("\n✅ Knowledge base is up to date!")
        return stats
    
    # Initialize vectorstore
    print(f"\n🔧 Loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_metadata=CHROMA_COLLECTION_METADATA
    )
    
    # Handle deletions
    if deleted_files:
        print(f"\n🗑️  Removing {len(deleted_files)} deleted files...")
        for file_path in deleted_files:
            chunk_ids = metadata_mgr.delete_file_metadata(file_path)
            if chunk_ids:
                try:
                    vectorstore.delete(ids=chunk_ids)
                    print(f"   ✓ {Path(file_path).name}")
                except Exception as e:
                    print(f"   ✗ {Path(file_path).name}: {e}")
    
    # Process new/changed files
    files_to_process = list(new_files | changed_files)
    
    if files_to_process:
        print(f"\n📥 Processing {len(files_to_process)} files...")
        
        # Remove old chunks for changed files
        for file_path in changed_files:
            chunk_ids = metadata_mgr.get_chunk_ids(file_path)
            if chunk_ids:
                vectorstore.delete(ids=chunk_ids)
        
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Process each file
        for file_path in files_to_process:
            try:
                # Get loader
                loader_class = get_loader_for_file(file_path)
                if not loader_class:
                    print(f"   ⚠️  Skipping {Path(file_path).name} (unsupported)")
                    continue
                
                # Load and split
              
                if loader_class == TextLoader:
                    loader = loader_class(file_path, encoding="utf-8", autodetect_encoding=True)
                else:
                    loader = loader_class(file_path)

                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                
                if not splits:
                    continue
                
                # Generate chunk IDs
                chunk_ids = [f"{file_path}_{i}" for i in range(len(splits))]
                
                # Add to vectorstore
                vectorstore.add_documents(documents=splits, ids=chunk_ids)
                
                # Update metadata
                file_type = Path(file_path).suffix.lower().replace('.', '')
                metadata_mgr.store_file_metadata(
                    file_path,
                    current_files[file_path],
                    chunk_ids,
                    file_type
                )
                
                status = "✓" if file_path in new_files else "🔄"
                print(f"   {status} {Path(file_path).name} ({len(splits)} chunks)")
                
            except Exception as e:
                print(f"   ✗ {Path(file_path).name}: {e}")
    
    # Final stats
    final_stats = metadata_mgr.get_stats()
    
    print(f"\n{'='*60}")
    print(f"✅ Knowledge base updated!")
    print(f"   Files: {final_stats['file_count']}")
    print(f"   Chunks: {final_stats['total_chunks']}")
    print("="*60 + "\n")
    
    return stats


def main():
    """Run knowledge base setup"""
    setup_knowledge_base()


if __name__ == "__main__":
    main()
