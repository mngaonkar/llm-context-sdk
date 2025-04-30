from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class Session():
    """Session class to manage chat messages."""
    def __init__(self):
        self.chat_history = []
        self.session_id = None
        self.max_chat_history = 2

    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        if len(self.chat_history) > self.max_chat_history:
            self.chat_history = self.chat_history[-self.max_chat_history:]

        self.chat_history.append(ChatMessage(role=role, content=content))
        return self.chat_history
    
    def get_chat_history(self):
        """Get the chat history."""
        return self.chat_history
    
    def get_chat_history_as_string(self):
        """Get the chat history as a string."""
        return "\n".join([f"{msg.role}: {msg.content}" for msg in self.chat_history])
    
    def get_chat_history_length(self):
        """Get the length of the chat history."""
        return len(self.chat_history)
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []
        return self.chat_history