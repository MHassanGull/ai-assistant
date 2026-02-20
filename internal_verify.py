from app.rag import clean_response
from app.memory import ConversationMemory

def test_clean_response():
    sample = "<think>I should check the resume.</think>Hassan is a Software Engineer."
    cleaned = clean_response(sample)
    expected = "Hassan is a Software Engineer."
    assert cleaned == expected, f"Expected '{expected}', got '{cleaned}'"
    print("✅ clean_response test passed")

def test_memory_persistence():
    memory = ConversationMemory(max_turns=2)
    memory.add_user_message("Hi")
    memory.add_assistant_message("Hello")
    messages = memory.get_recent_messages()
    assert len(messages) == 2
    assert messages[0]['role'] == 'user'
    assert messages[1]['role'] == 'assistant'
    print("✅ ConversationMemory logic passed")

if __name__ == "__main__":
    test_clean_response()
    test_memory_persistence()
