from typing import List, Dict


class ConversationMemory:
    """
    Lightweight, token-safe short-term memory.
    Keeps only the last N conversation turns.
    """

    def __init__(self, max_turns: int = 4):
        # max_turns = number of user+assistant pairs
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim_memory()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim_memory()

    def get_recent_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def clear(self):
        self.messages = []

    def _trim_memory(self):
        """
        Keep only the last max_turns * 2 messages
        (each turn = user + assistant)
        """
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
