# Tag structure
# - category_tag
#     - criteria_v0.1
#         - specificity
#         - ...
#     - math_v0.1
#         - math
#     - if_v0.1
#         - if
#         - score
#     - creative_writing_v0.1
#         - creative_writing
#         - score
#     - refusal_v0.2
#         - refusal

import ast
import re
import numpy as np
from collections import defaultdict

from utils import HuggingFaceClassifier, chat_completion_openai


class Category:
    def __init__(self):
        pass

    @staticmethod
    def create_category(name):
        if name == "criteria_v0.1":
            return CategoryHardPrompt()
        elif name == "if_v0.1":
            return CategoryIF()
        elif name == "math_v0.1":
            return CategoryMath()
        elif name == "creative_writing_v0.1":
            return CategoryCreativeWriting()
        elif name == "refusal_v0.2":
            return CategoryRefusalHF()

        raise Exception(f"Category name is incorrect: {name}")


class CategoryAPI(Category):
    def __init__(self):
        self.batch_size = 1
        self.is_parallel = True

    def get_answer(self, batch, model_name, max_tokens, temperature, api_dict):
        assert len(batch) == 1, "API-based categories must have batch size of 1"

        conv, uid = self.pre_process(batch)
        output = chat_completion_openai(
            model=model_name,
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
        return self.post_process(output, uid)

    def pre_process(self, row):
        """
        Prepares a text to be labeled by LLM through OpenAI API

        Inherited category classifier classes should implement this method.

        Args:
            row (pd.Dataframe): row representing single battle to be labeled

        Returns:
            conv (dict): processed text with system prompt in OpenAI API format:
                [
                    {"role": "system", "content": <system prompt>"},
                    {"role": "user", "content": <user input>},
                    ...
                ]
        """
        pass

    def post_process(self, judgement, uid):
        """
        Processes judgements/outputs of LLM to retrieve final labels

        Inherited category classifier classes should implement this method.

        Args:
            judgement (str): text output of LLM labeler
            uid (str): UID of the battle to be labeled

        Returns:
            output (Dict[str, Dict[str, str]]: Keys is battle uid, value is the output associated with that battle (usually a dictionary)
        """
        pass


class CategoryHF(Category):
    def __init__(self):
        self.batch_size = 1
        self.is_paraellel = False

    def get_answer(self, batch, model_name, max_tokens, temperature, api_dict):
        to_label, to_label_uids = self.pre_process(batch)
        labels = self.classifier.classify_batch(to_label)

        return self.post_process(labels, to_label_uids)

    def pre_process(self, batch):
        """
        Prepares a batch of texts to be labeled by Hugging Face classifier.

        Inherited category classifier classes should implement this method.

        Args:
            batch (pd.DataFrame): Each row of the DataFrame represents one battle.

        Returns:
            outputs (List[str]): Texts to be labeled by HF classifier
            to_label_uids (List[str]): Battle UIDs corresponding to each text to be labeled
        """
        pass

    def post_process(labels, to_label_uids):
        """
        Processes raw HF labels.

        Inherited category classifier classes should implement this method.

        Args:
            labels (List[bool]): labels directly from HF classifier
            to_label_uids (List[str]): Battle UIDs corresponding to each string that was labeled

        Returns:
            output (Dict[str, Dict[str, str]]: Key are battle uids, value is the output associated with that battle (usually a dictionary)
        """
        pass


class CategoryHardPrompt(CategoryAPI):
    def __init__(self):
        super().__init__()
        self.name_tag = "criteria_v0.1"
        self.pattern = re.compile(r"(\[\d(?:\,\s\d)*\])")
        self.sys_prompt = "Your task is to evaluate how well the following input prompts can assess the capabilities of advanced AI assistants.\n\nFor the input prompt, please analyze it based on the following 7 criteria.\n1. Specificity: Does the prompt ask for a specific output, such as code, a mathematical solution, a logical simplification, a problem-solving strategy, or a hardware setup recommendation? This specificity allows the AI to demonstrate its ability to understand and generate precise responses.\n2. Domain Knowledge: Does the prompt cover a specific domain, such as programming, mathematics, logic, problem-solving, or hardware setup? Prompts spanning a range of topics test the AI's breadth of knowledge and its ability to apply that knowledge to different domains.\n3. Complexity: Does the prompt vary in complexity, from straightforward tasks to more complex, multi-step problems? This allows evaluators to assess the AI's capability to handle problems of varying difficulty.\n4. Problem-Solving Skills: Does the prompt directly involves the AI to demonstrate active problem-solving skills, such systemically coming up with a solution for a specific setup instead of regurgitating an existing fact? This tests the AI's ability to apply logical reasoning and provide practical solutions.\n5. Creativity: Does the prompt involve a level of creativity in approaching the problem? This criterion tests the AI's ability to provide tailored solutions that take into account the user's specific needs and limitations.\n6. Technical Accuracy: Does the prompt require technical accuracy in the response? This allows evaluators to assess the AI's precision and correctness in technical fields.\n7. Real-world Application: Does the prompt relate to real-world applications, such as setting up a functional system or writing code for a practical use case? This tests the AI's ability to provide practical and actionable information that could be implemented in real-life scenarios.\n\nYou must list the criteria numbers that the prompt satisfies in the format of a Python array. For example, \"[...]\". Do not explain your choice."
        self.tags = {
            1: "specificity",
            2: "domain_knowledge",
            3: "complexity",
            4: "problem_solving",
            5: "creativity",
            6: "technical_accuracy",
            7: "real_world",
        }

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return []
        elif len(set(matches)) == 1:
            try:
                return ast.literal_eval(matches[0])
            except SyntaxError:
                print(matches[0])
                return []
        else:
            return []

    def pre_process(self, row):
        prompt = row["prompt"].iloc[0]
        conv = [{"role": "system", "content": self.sys_prompt}]
        conv.append({"role": "user", "content": prompt})
        return conv, row["uid"].iloc[0]

    def post_process(self, judgment, uid):
        raw_output = {uid: judgment}

        criteria = self.get_score(judgment=judgment)
        output = {uid: {name: bool(i in criteria) for i, name in self.tags.items()}}
        return output, raw_output


class CategoryIF(CategoryAPI):
    def __init__(self):
        super().__init__()
        self.name_tag = "if_v0.1"
        self.pattern = re.compile(r"<score>([012345])<\/score>")
        self.system_prompt = "You are an AI assistant tasked with determining whether a given user prompt can effectively assess another AI's ability to follow instructions. Your goal is to analyze the prompt and decide if it contains specific, clear instructions that would test an AI's capability to understand and execute directions accurately. Carefully examine the user prompt and consider the following aspects:\n1. Does it contain specific instructions or requirements?\n2. Are there multiple steps or elements the AI needs to address?\n3. Does it ask for a particular format or structure in the response?\n4. Is there a unique or challenging aspect that would test the AI's ability to follow directions precisely?\n\nConsider both the content and the structure of the instructions. A good prompt for assessing instruction-following capabilities should have clear, specific directions that can be objectively evaluated. Think about why this prompt does or does not effectively assess an AI's ability to follow instructions. Consider both the strengths and weaknesses of the prompt in this regard. Output your verdict as a score from 0 to 5:\n0 = Does not evaluate instruction-following ability.\n1 = Ineffective at evaluating instruction-following ability.\n2 = Somewhat effective at evaluating instruction-following ability.\n3 = Effective at evaluating simple instruction-following ability.\n4 = Effective at evaluating more complex instruction-following ability.\n5 = Effective at evaluating advanced instruction-following ability.\n\nPresent your score in the following format:\n<score>[Your score from 0 to 5]</score>.\nDo NOT explain."
        self.prompt_template = "<user_prompt>{PROMPT}</user_prompt>"
        self.batch_size = 1
        self.is_parallel = True

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return int(matches[0])
        else:
            return None

    def pre_process(self, row):
        prompt = row["prompt"].iloc[0]
        args = {"PROMPT": prompt}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv, row["uid"].iloc[0]

    def post_process(self, judgment, uid):
        raw_output = {uid: judgment}

        score = self.get_score(judgment=judgment)
        output = {
            uid: {
                "if": bool(score >= 4) if score else False,
                "score": score,
            }
        }
        return output, raw_output


class CategoryMath(CategoryAPI):
    def __init__(self):
        super().__init__()
        self.name_tag = "math_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = 'You are tasked with determining whether a given user prompt requires an AI assistant to solve a math problem and apply mathematical logic and reasoning.\n\nCarefully analyze the user prompt and consider whether it requires mathematical problem-solving skills to answer correctly. Think about the following aspects:\n\n1. Does it require the application of a specific mathematical concept or formula?\n2. Does the prompt involve numerical calculations or algebraic manipulation or logical reasoning?\n3. Is there a clear mathematical problem to be solved?\n4. Would answering this prompt demonstrate proficiency in a specific area in mathematics?\n\nOutput your verdict in the following format:"<decision>\n[yes/no]\n</decision>". Do NOT explain.'
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"
        self.batch_size = 1
        self.is_parallel = True

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment.replace("\n", "").lower())
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0]
        else:
            return None

    def pre_process(self, row):
        prompt = row["prompt"].iloc[0]
        args = {"PROMPT": prompt}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv, row["uid"].iloc[0]

    def post_process(self, judgment, uid):
        raw_output = {uid: judgment}

        score = self.get_score(judgment=judgment)
        output = {uid: {"math": bool(score == "yes") if score else False}}

        return output, raw_output


class CategoryCreativeWriting(CategoryAPI):
    def __init__(self):
        super().__init__()
        self.name_tag = "creative_writing_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = 'You are tasked with determining whether a given user prompt is asking for creative writing. Creative writing is defined as any form of writing that goes beyond standard professional, journalistic, academic, or technical literature. It typically involves imagination, originality, and expression of thoughts and emotions. Creative writing can include, but is not limited to, the following formats:\n- Fiction (e.g., short stories, novels)\n- Poetry (e.g., sonnets, free verse)\n- Dramatic writing (e.g., screenplays, monologues, scripts)\n- Personal essays (focusing on subjective experiences or narrative storytelling)\n- Songs and lyrics\n\nCarefully analyze the user prompt and consider whether it primarily requires creative writing. Think about the following aspects:\n1. Does the prompt ask for fictional content, speculative scenarios, or the use of imagination to construct narratives?\n2. Does it encourage the expression of thoughts, emotions, or personal experiences beyond mere factual reporting or analysis?\n3. Is it asking for writing in a specific creative format (e.g., story, poem, script, etc)?\n4. Is the primary purpose of the prompt to foster creative expression or originality rather than information delivery, technical documentation, or analytical reasoning?\n5. Does the prompt request stylistic or rhetorical elements often associated with creative writing, such as metaphor, imagery, dialogue, etc?\n6. Does the prompt expect a response in natural language (e.g., sentences, paragraphs) rather than visual, mathematical, or non-linguistic output?\n\nOutput your verdict as either "yes" or "no"in the following format:\n<decision>\n[yes/no]\n</decision>. Do NOT explain.'
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"
        self.batch_size = 1
        self.is_parallel = True

    def get_score(self, judgment):
        matches = self.pattern.findall(
            judgment.replace("\n", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
            .lower()
        )
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0]
        else:
            return None

    def pre_process(self, row):
        prompt = row["prompt"].iloc[0]
        args = {"PROMPT": prompt}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv, row["uid"].iloc[0]

    def post_process(self, judgment, uid):
        raw_output = {uid: judgment}

        score = self.get_score(judgment=judgment)
        bool_score = bool(score == "yes") if score else False
        output = {uid: {"creative_writing": bool_score, "score": score}}

        return output, raw_output


class CategoryRefusalHF(CategoryHF):
    def __init__(self):
        super().__init__()
        self.name_tag = "refusal_v0.2"
        self.prompt_template = "Here is the user query:\n<user_query>\n{QUERY}\n</user_query>\n\nHere is the LLM response to the user:\n<llm_response>\n{RESPONSE}\n</llm_response>"
        self.classifier = HuggingFaceClassifier(
            model_path="lmarena-ai/RefusalClassifier"
        )
        self.batch_size = 1
        self.is_parallel = False

    def conv_pre_process_helper(self, conversation):
        conv = []
        for i in range(0, len(conversation), 2):
            args = {
                "QUERY": conversation[i]["content"],
                "RESPONSE": conversation[i + 1]["content"],
            }
            conv.append(self.prompt_template.format(**args))
        return conv

    def pre_process(self, batch):
        to_label = []
        to_label_uids = []

        for _, row in batch.iterrows():
            conv_a = self.conv_pre_process_helper(row["conversation_a"])
            conv_b = self.conv_pre_process_helper(row["conversation_b"])

            to_label.extend(conv_a)
            to_label.extend(conv_b)

            to_label_uids.extend([row["uid"]] * (len(conv_a) + len(conv_b)))

        return to_label, to_label_uids

    def post_process(self, labels, to_label_uids):
        outputs = defaultdict(lambda: {"label": False})
        query_refusals = np.where(labels)[0]

        for i in query_refusals:
            outputs[to_label_uids[i]] = {"label": True}

        return outputs, defaultdict(
            lambda: None
        )  # No raw/testing outputs for HF classifier
