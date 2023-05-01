from typing import List, Tuple, Any

class Message:
    def __init__(self, input):
        self.input = input
        self.output = None
        self.is_prompt = True

class Prompt:

    def __init__(self):
        self.stop_str: str = None

    def support_model(model_name):
        return False

    def get_prompt(self, history, message):
        result = ""
        result += self.make_prefix()
        result += self.make_history(history)
        result += self.make_last(message)
        return result
    
    def make_prefix(self):
        return ""
    
    def make_history(self, history):
        result = ""
        for message in history:
            if not message.is_prompt:
                continue
            result += self.make_message(message)
        return result
    
    def make_message(self, message):
        return ""
    
    def make_last(self, message):
        return ""
    
    
class OneShot(Prompt):

    def __init__(self):
        self.stop_str = "###"

    def support_model(self, model_name):
        return True
    
    def make_message(self, message):
        return \
            f'Humain: {message.input}###' +\
            f'Assistant: {message.output}###'
    
    def make_last(self, message):
        return \
            f'Humain: {message.input}###' +\
            f'Assistant: '
    
    def make_prefix(self):
        return \
            "A chat between a curious human and an artificial intelligence assistant. " + \
            "The assistant gives helpful, detailed, and polite answers to the human's questions." + \
                "\n###" + \
            "Human: What are the key differences between renewable and non-renewable energy sources?" + \
                "\n###" + \
            "Assistant: " + \
            "Renewable energy sources are those that can be replenished naturally in a relatively " +\
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. " +\
            "Non-renewable energy sources, on the other hand, are finite and will eventually be " +\
            "depleted, such as coal, oil, and natural gas. Here are some key differences between " +\
            "renewable and non-renewable energy sources:\n" +\
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable " +\
            "energy sources are finite and will eventually run out.\n" +\
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact " +\
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, " +\
            "and other negative effects.\n" +\
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically " +\
            "have lower operational costs than non-renewable sources.\n" +\
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote " +\
            "locations than non-renewable sources.\n" +\
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different " +\
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n" +\
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while " +\
            "non-renewable sources are not, and their depletion can lead to economic and social instability." +\
                "\n###"

class Vicuna_V1_1(Prompt):

    def support_model(self, model_name):
        return "vicuna" in model_name or "output" in model_name
    
    def make_message(self, message):
        return \
            f'USER: {message.input} ' +\
            f'ASSISTANT: {message.output}</s>'
    
    def make_last(self, message):
        return \
            f'USER: {message.input} ' +\
            f'ASSISTANT:'
    
    def make_prefix(self):
        return \
            "A chat between a curious user and an artificial intelligence assistant. " +\
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "


class Koala_V1(Prompt):

    def support_model(self, model_name):
        return "koala" in model_name
    
    def make_message(self, message):
        return \
            f'USER: {message.input} ' +\
            f'GPT: {message.output}</s>'
    
    def make_last(self, message):
        return \
            f'USER: {message.input} ' +\
            f'GPT:'
    
    def make_prefix(self):
        return \
            "BEGINNING OF CONVERSATION: "

_allPrompts = [Vicuna_V1_1(), Koala_V1(), OneShot()]

class Session:

    def __init__(self):
        self.prompts = _allPrompts
        self.currentPrompt: Prompt = self._set_curren_prompt(_allPrompts[-1])
        self.history = []
        self.current: Message = None
        ###
        self.conv_id: Any = None
        self.skip_next: bool = False
        self.model_name: str = None

    def set_model_name(self, model_name):
        self.model_name = model_name
        self._set_curren_prompt(self._get_prompt_by_name(model_name))

    def _set_curren_prompt(self, prompt: Prompt):
        self.currentPrompt = prompt
        self.stop_str = prompt.stop_str

    def _get_prompt_by_name(self, model_name):
        model_name = model_name.lower()
        for prompt in self.prompts:
            if prompt.support_model(model_name):
                return prompt
            
    def append_input(self, text):
        if self.current:
            self.history.append(self.current)
        self.current = Message(text)

    def append_output(self, text):
        self.current.output = text

    def append_fail(self, text):
        self.append_output(text)
        self.current.is_prompt = False

    def get_prompt(self):
        return self.currentPrompt.get_prompt(self.history, self.current)
    
    def to_gradio_chatbot(self):
        return \
            [[message.input, message.output] for message in self.history] + \
            [[self.current.input, None]]
    
if __name__ == '__main__':
    session = Session()
    session.set_model_name("vacuna")
    session.append_input("Hello!")
    session.append_output("Hi!")
    session.append_input("Fuck!")
    session.append_fail("YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN.")
    session.append_input("How are you?")
    print(session.get_prompt())
    print(session.to_gradio_chatbot())
    print(session.stop_str)
