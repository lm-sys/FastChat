import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: str = "###"

    def get_prompt(self):
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ":"
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def append_gradio_chatbot_history(self, history):
        for a, b in history:
            self.messages.append([self.roles[0], a])
            self.messages.append([self.roles[1], b])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep=self.sep)



default_conversation = Conversation(
    system="A chat between a curious human and a knowledgeable artificial intelligence assistant.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hello! What can you do?"),
        ("Assistant", "As an AI assistant, I can answer questions and chat with you."),
        ("Human", "Give me a list of tallest mountains in the world."),
        ("Assistant", "here's a list of the ten tallest mountains in the world based on their height in meters:\n"
         "Mount Everest - 8,848 meters\n"
         "K2 (also known as Mount Godwin-Austen) - 8,611 meters\n"
         "Kangchenjunga - 8,586 meters\n"
         "Lhotse - 8,516 meters\n"
         "Makalu - 8,485 meters\n"
         "Please note that there are differing opinions on the exact height of these mountains "
         "due to factors such as changes in elevation due to natural processes or differences in surveying methods."),
    )
)


if __name__ == "__main__":
    print(default_conversation.get_prompt())
