import sys
import os

# Add the current directory to path so we can import 'app'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ingest import sync_website
from app.rag import answer

def main():
    print("--- 🤖 AI Assistant Verification ---")
    
    # 1. Sync Website Data
    print("\nStep 1: Syncing website data to Pinecone...")
    try:
        # force=False checks if data already exists to save units
        sync_website(force=False)
    except Exception as e:
        print(f"❌ Ingestion Error: {e}")
        return

    # 2. Test RAG Response
    print("\nStep 2: Testing Chatbot Response...")
    test_question = "give me the linkedin profile of Hassan"
    print(f"Question: {test_question}")
    
    try:
        response = answer(test_question)
        print("\n--- 🗨️ Response ---")
        print(response)
        print("------------------")
        print("\n✅ Verification complete! If the response is accurate, your bot is working perfectly.")
    except Exception as e:
        print(f"❌ RAG Error: {e}")

if __name__ == "__main__":
    main()
