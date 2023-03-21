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
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Give three tips for staying healthy."),
        ("Assistant",
            "Sure, here are three tips for staying healthy:\n"
            "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
            "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
            "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
            "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
            "activities at least two days per week.\n"
            "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
            "vegetables, whole grains, lean proteins, and healthy fats can help support "
            "your overall health. Try to limit your intake of processed and high-sugar foods, "
            "and aim to drink plenty of water throughout the day.\n"
            "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
            "and mental health. Adults should aim for seven to nine hours of sleep per night. "
            "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
            "help improve the quality of your sleep.")
    )
)


if __name__ == "__main__":
    print(default_conversation.get_prompt())
