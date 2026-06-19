"""
Local Data Ingestion Script for Indus-Guardian (PRO VERSION).
Run this ONCE on your local machine to populate the cloud Pinecone database.
"""

import os
import glob
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# Force load the local .env file before backend configurations initialize
load_dotenv()

try:
    from backend.engine.document_processor import DocumentProcessor
    from backend.core.config import get_settings
except ImportError:
    print("Error: Make sure you are running this script from the root 'Indus-Guardian' directory.")
    exit(1)

def run_pre_ingestion():
    settings = get_settings()
    docs_folder = "pre_indexed_docs"
    shared_namespace = "global_knowledge_base"
    
    print("=" * 60)
    print("🚀 STARTING PRO KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    # 1. PURGE OLD CORRUPTED VECTORS
    print(f"\n🧹 STEP 1: Wiping old corrupted data from namespace '{shared_namespace}'...")
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        index.delete(delete_all=True, namespace=shared_namespace)
        print("   ↳ Namespace wiped successfully. Clean slate ready.")
        time.sleep(3) # Wait for Pinecone to register the deletion across its global nodes
    except Exception as e:
        print(f"   ↳ Warning: Could not clear namespace (it might already be empty). Proceeding...")

    # 2. Check for document folder
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"📦 Created a new folder named: '{docs_folder}'")
        print("👉 Action Required: Copy your 4-5 core PDFs into that folder and run this script again.")
        return

    # 3. Scan for target files
    supported_files = glob.glob(os.path.join(docs_folder, "*.pdf")) + glob.glob(os.path.join(docs_folder, "*.docx"))
    
    if not supported_files:
        print(f"⚠️ No files found inside the '{docs_folder}/' directory.")
        print("👉 Action Required: Drop your AI/ML PDFs/DOCX files there first.")
        return

    print(f"\n📂 Found {len(supported_files)} files ready for cloud compilation.")
    
    # 4. Initialize Core Processing Component
    try:
        processor = DocumentProcessor()
    except Exception as e:
        print(f"❌ Initialization Failed. Check your local .env file credentials. Error: {e}")
        return

    # 5. Process and Loop files sequentially
    for file_path in supported_files:
        filename = os.path.basename(file_path)
        print(f"\n📄 Parsing Document: [ {filename} ]")
        
        # Detect exact MIME type
        content_type = "application/pdf" if filename.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        try:
            with open(file_path, "rb") as file_buffer:
                raw_bytes = file_buffer.read()
            
            # Step A: Run PDF/DOCX Parser and Semantic Text Chunking
            parsed_data = processor.parse(
                filename=filename,
                content_type=content_type,
                file_bytes=raw_bytes
            )
            print(f"   ↳ Segmented into {len(parsed_data.chunks)} individual content blocks.")

            # Step B: Compute embeddings and stream batches to Pinecone Cloud
            print("   ↳ Streaming vectors to Pinecone cloud infrastructure...")
            processor.index_to_pinecone(session_id=shared_namespace, parsed_doc=parsed_data)
            print(f"   ✅ Successfully indexed: {filename}")

        except Exception as err:
            print(f"   ❌ Failed to process file {filename}. Logged exception: {err}")
            continue

    print("\n" + "=" * 60)
    print("🏁 KNOWLEDGE BASE COMPLETELY DEPLOYED TO PINECONE CLOUD")
    print(f"   Target Namespace: {shared_namespace}")
    print("   Your system is now fully armed with clean, persistent data.")
    print("=" * 60)

if __name__ == "__main__":
    run_pre_ingestion()