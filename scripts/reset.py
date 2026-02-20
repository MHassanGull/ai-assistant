import sys
import os
# Add the parent directory to the Python path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store import clear_namespace

if __name__ == "__main__":
    confirm = input("⚠️ This will delete all vectors. Type YES to continue: ")

    if confirm == "YES":
        clear_namespace()
        print("✅ Namespace cleared.")
    else:
        print("Cancelled.")
