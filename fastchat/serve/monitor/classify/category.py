import ast
import re


class Category:
    def __init__(self):
        pass

    @staticmethod
    def create_category(name):
        if name == "criteria_tag":
            return CategoryHardPrompt()
        elif name == "if_tag":
            return CategoryIF()
        elif name == "math_tag":
            return CategoryMath()

        raise Exception(f"Category name is incorrect: {name}")

    def post_process(self):
        pass


class CategoryHardPrompt(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "criteria_tag"
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

    def pre_process(self, prompt):
        conv = [{"role": "system", "content": self.sys_prompt}]
        conv.append({"role": "user", "content": prompt})
        return conv

    def post_process(self, judgment):
        criteria = self.get_score(judgment=judgment)
        return {name: bool(i in criteria) for i, name in self.tags.items()}


class CategoryIF(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "if_tag"
        self.pattern = re.compile(r"<score>([012345])<\/score>")
        self.prompt_template = "You are an AI assistant tasked with determining whether a given user prompt can effectively assess another AI's ability to follow instructions. Your goal is to analyze the prompt and decide if it contains specific, clear instructions that would test an AI's capability to understand and execute directions accurately.\n\nHere is the user prompt to analyze:\n<user_prompt>\n{PROMPT}\n</user_prompt>\n\nCarefully examine the prompt above. Consider the following aspects:\n1. Does it contain specific instructions or requirements?\n2. Are there multiple steps or elements the AI needs to address?\n3. Does it ask for a particular format or structure in the response?\n4. Is there a unique or challenging aspect that would test the AI's ability to follow directions precisely?\n\nIn your analysis, consider both the content and the structure of the instructions. A good prompt for assessing instruction-following capabilities should have clear, specific directions that can be objectively evaluated.\n\nProvide your reasoning for why this prompt does or does not effectively assess an AI's ability to follow instructions. Consider both the strengths and weaknesses of the prompt in this regard.\n\nAfter providing your analysis, assign a score from 0 to 5:\n0 = Does not evaluate instruction-following ability.\n1 = Ineffective at evaluating instruction-following ability.\n2 = Somewhat effective at evaluating instruction-following ability.\n3 = Effective at evaluating simple instruction-following ability.\n4 = Effective at evaluating more complex instruction-following ability.\n5 = Effective at evaluating advanced instruction-following ability.\n\nPresent your response in the following format:\n<analysis>\nYour detailed analysis and reasoning here.\n</analysis>\n\nPresent your score in the following format:\n<score>[Your score from 0 to 5]</score>\n\nEnsure that your justification is comprehensive and your score accurately reflects your analysis of the prompt's effectiveness."

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return int(matches[0])
        else:
            return None

    def pre_process(self, prompt):
        args = {"PROMPT": prompt}
        conv = [{"role": "user", "content": self.prompt_template.format(**args)}]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {
            "bool": bool(score >= 4) if score else False,
            "score": score,
        }


class CategoryMath(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "math_tag"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.prompt_template = "You are tasked with determining whether a given user prompt requires an AI assistant to solve a math problem and apply mathematical logic and reasoning.\n\nHere is the user prompt to be evaluated:\n<user_prompt>\n{PROMPT}\n</user_prompt>\n\nCarefully analyze this prompt and consider whether it requires mathematical problem-solving skills to answer correctly. Think about the following aspects:\n\n1. Does it require the application of a specific mathematical concept or formula?\n2. Does the prompt involve numerical calculations or algebraic manipulation or logical reasoning?\n3. Is there a clear mathematical problem to be solved?\n4. Would answering this prompt demonstrate proficiency in a specific area in mathematics?\n\nAfter your analysis, provide your reasoning for why this prompt either can or cannot assess math problem-solving abilities. Consider both the content and the structure of the prompt in your explanation.\n\nBased on your analysis and reasoning, make a final decision on whether this prompt can effectively assess an AI assistant's ability to solve math problems.\n\nPresent your response in the following format:\n<analysis>\n[Your detailed analysis and reasoning here]\n</analysis>\n\n<decision>\n[yes/no]\n</decision>"

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment.replace("\n", "").lower())
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0]
        else:
            return None

    def pre_process(self, prompt):
        args = {"PROMPT": prompt}
        conv = [{"role": "user", "content": self.prompt_template.format(**args)}]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"bool": bool(score == "yes") if score else False, "label": score}
