import sys
import os
# Add the parent directory to the Python path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest import sync_website

if __name__ == "__main__":
    print("🔄 Syncing website...")
    sync_website(force=True)
    print("✅ Sync complete.")
