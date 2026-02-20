import sys
import os
# Add the parent directory to the Python path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import answer
from app.memory import ConversationMemory

memory = ConversationMemory(max_turns=4)

if __name__ == "__main__":
    print("\n" + "="*40)
    print("🤖 PORTFOLIO AI ASSISTANT (Local)")
    print("="*40)
    print("Type 'exit' to quit.\n")

    while True:
        print(f"\n[Memory: {len(memory.get_recent_messages())} messages / {memory.max_turns} turns]")
        question = input("👤 You: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "bye"]:
            print("\n" + "="*40)
            print("🤖 Bot: Goodbye! Have a great day! 👋")
            print("="*40)
            break

        # Adding message to memory handles internal trimming
        memory.add_user_message(question)

        # Get response from RAG
        print("🤖 Thinking...")
        response = answer(question, memory=memory)

        # Store response
        memory.add_assistant_message(response)

        print("\n" + "-"*40)
        print(f"🤖 Bot: {response}")
        print("-"*40)
