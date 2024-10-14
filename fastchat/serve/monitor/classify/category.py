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
import ast
import re

from vision_utils import pil_to_base64, get_image_file_from_gcs


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
        elif name == "refusal_v0.1":
            return CategoryRefusal()
        elif name == "captioning_v0.1":
            return CategoryCaptioning()
        elif name == "counting_v0.1":
            return CategoryCounting()
        elif name == "creative_writing_v0.1":
            return CategoryCreativeWriting()
        elif name == "entity_recognition_v0.1":
            return CategoryEntityRecognition()
        elif name == "ocr_v0.1":
            return CategoryOpticalCharacterRecognition()
        elif name == "humor_v0.1":
            return CategoryHumor()
        elif name == "homework_v0.1":
            return CategoryHomework()

        raise Exception(f"Category name is incorrect: {name}")

    def post_process(self):
        pass


class CategoryHardPrompt(Category):
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

    def pre_process(self, prompt):
        conv = [{"role": "system", "content": self.sys_prompt}]
        conv.append({"role": "user", "content": prompt["prompt"]})
        return conv

    def post_process(self, judgment):
        criteria = self.get_score(judgment=judgment)
        return {name: bool(i in criteria) for i, name in self.tags.items()}


class CategoryIF(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "if_v0.1"
        self.pattern = re.compile(r"<score>([012345])<\/score>")
        self.system_prompt = "You are an AI assistant tasked with determining whether a given user prompt can effectively assess another AI's ability to follow instructions. Your goal is to analyze the prompt and decide if it contains specific, clear instructions that would test an AI's capability to understand and execute directions accurately. Carefully examine the user prompt and consider the following aspects:\n1. Does it contain specific instructions or requirements?\n2. Are there multiple steps or elements the AI needs to address?\n3. Does it ask for a particular format or structure in the response?\n4. Is there a unique or challenging aspect that would test the AI's ability to follow directions precisely?\n\nConsider both the content and the structure of the instructions. A good prompt for assessing instruction-following capabilities should have clear, specific directions that can be objectively evaluated. Think about why this prompt does or does not effectively assess an AI's ability to follow instructions. Consider both the strengths and weaknesses of the prompt in this regard. Output your verdict as a score from 0 to 5:\n0 = Does not evaluate instruction-following ability.\n1 = Ineffective at evaluating instruction-following ability.\n2 = Somewhat effective at evaluating instruction-following ability.\n3 = Effective at evaluating simple instruction-following ability.\n4 = Effective at evaluating more complex instruction-following ability.\n5 = Effective at evaluating advanced instruction-following ability.\n\nPresent your score in the following format:\n<score>[Your score from 0 to 5]</score>.\nDo NOT explain."
        self.prompt_template = "<user_prompt>{PROMPT}</user_prompt>"

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return int(matches[0])
        else:
            print("Error parsing IF")
            return None

    def pre_process(self, prompt):
        args = {"PROMPT": prompt["prompt"]}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {
            "if": bool(score >= 4) if score else False,
            "score": score,
        }


class CategoryMath(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "math_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = 'You are tasked with determining whether a given user prompt requires an AI assistant to solve a math problem and apply mathematical logic and reasoning.\n\nCarefully analyze the user prompt and consider whether it requires mathematical problem-solving skills to answer correctly. Think about the following aspects:\n\n1. Does it require the application of a specific mathematical concept or formula?\n2. Does the prompt involve numerical calculations or algebraic manipulation or logical reasoning?\n3. Is there a clear mathematical problem to be solved?\n4. Would answering this prompt demonstrate proficiency in a specific area in mathematics?\n\nOutput your verdict in the following format:"<decision>\n[yes/no]\n</decision>". Do NOT explain.'
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"

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
        args = {"PROMPT": prompt["prompt"]}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"math": bool(score == "yes") if score else False}


class CategoryRefusal(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "refusal_v0.1"
        self.pattern = re.compile(r"<decision>([a-z]+)</decision>")
        self.system_prompt = 'You are tasked with determining if any of the given model responses are a refusal to answer. A refusal occurs when the model explicitly declines to answer or complete a task, due to reasons like safety, moderation, or model limitations (e.g. the user asks the model to search the web but it does not have that capability). You will be given the responces of 2 models, A and B and you are to determine if A refuses to answer, B refuses, both refuse, or neither refuse. Output your verdict in the following format:"<decision>\n[A/B/both/neither]\n</decision>". Do not respond with any additional information.'

        self.prompt_template = "<model_A_response>\n{RESPONSE_A}\n</model_A_response>\n<model_B_response>\n{RESPONSE_B}\n</model_B_response>"

    def get_score(self, judgment):
        match = self.pattern.search(
            judgment.replace("\n", "").replace("[", "").replace("]", "").lower()
        )
        if match:
            return match.group(1)
        return "error"

    def pre_process(self, prompt):
        args = {"RESPONSE_A": prompt["response_a"], "RESPONSE_B": prompt["response_a"]}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {
            "refusal_a": bool(score == "a") or bool(score == "both"),
            "refusal_b": bool(score == "b") or bool(score == "both"),
            "refusal": bool(score == "a")
            or bool(score == "b")
            or bool(score == "both"),
        }


class CategoryCaptioning(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "captioning_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = "You are tasked with determining if a given VQA question is a captioning question. A captioning question asks for a general, overall description of the entire image. It must be a single, open-ended query that does NOT ask about particular objects, people, or parts of the image, nor require interpretation beyond a broad description of what is visually present. Examples include 'What is happening in this image?', 'Describe this picture.', 'Explain', etc. An example of a non-captioning question is 'Describe what is funny in this picture.' because it asks for a specific interpretation of the image content. \n\nOutput your verdict in the following format:<decision>\n[yes/no]\n</decision>. Do NOT explain."
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"

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
        args = {"PROMPT": prompt["prompt"]}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"captioning": bool(score == "yes") if score else False}


class CategoryCounting(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "counting_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = "You are tasked with determining if a given VQA question is a counting question. A counting question explicitly asks for counting or identifying the number of objects in the image. Your task is to analyze the question and determine if it is a counting question.\n\nOutput your verdict in the following format:<decision>\n[yes/no]\n</decision>. Do NOT explain."
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"

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
        args = {"PROMPT": prompt["prompt"]}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"counting": bool(score == "yes") if score else False}


class CategoryCreativeWriting(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "creative_writing_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = "You are tasked with determining if a given VQA question is a creative writing question. A creative writing question explicitly asks for creative or imaginative responses based on the image, such as composing a story, poem, or providing a fictional interpretation. This excludes questions that simply ask for factual observations, interpretations, or speculations about the image content.\n\nOutput your verdict in the following format:<decision>\n[yes/no]\n</decision>. Do NOT explain."
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"

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
        args = {"PROMPT": prompt["prompt"]}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"creative_writing": bool(score == "yes") if score else False}


class CategoryEntityRecognition(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "entity_recognition_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = "You are tasked with determining if a given VQA question is an entity recognition question. An entity recognition question asks for the identification of specific objects or people in the image. This does NOT include questions that ask for a general description of the image, questions that only ask for object counts, or questions that only require reading text in the image.\n\nOutput your verdict in the following format:<decision>\n[yes/no]\n</decision>. Do NOT explain."

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
        args = {"PROMPT": prompt["prompt"]}
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_template.format(**args)},
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"creative_writing": bool(score == "yes") if score else False}


import io
import base64


def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


class CategoryOpticalCharacterRecognition(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "ocr_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = "You are tasked with determining if a given VQA question is an optical character recognition (OCR) question. An OCR question requires reading and understanding text in the image to answer. If there is some amount of text in the image and the question requires reading the text in any capacity it should be classified as Optical Character Recognition.\n\nOutput your verdict in the following format:<decision>\n[yes/no]\n</decision>. Do NOT explain."

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
        args = {"PROMPT": prompt["prompt"]}
        base64_image = get_image_file_from_gcs(prompt["image_hash"])
        conv = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_template.format(**args)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"ocr": bool(score == "yes") if score else False}


class CategoryHumor(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "humor_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = "You are tasked with determining if a given VQA question is a humor question. A humor question asks for a humorous or funny response based on the image or asks to understand what is funny about an image. This includes questions that ask to explain an image which is humorous, such as memes.\n\nOutput your verdict in the following format:<decision>\n[yes/no]\n</decision>. Do NOT explain."
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"

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
        args = {"PROMPT": prompt["prompt"]}
        base64_image = get_image_file_from_gcs(prompt["image_hash"])
        conv = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_template.format(**args)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"humor": bool(score == "yes") if score else False}


class CategoryHomework(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "homework_v0.1"
        self.pattern = re.compile(r"<decision>(\w+)<\/decision>")
        self.system_prompt = "You are tasked with determining if a given VQA question is a homework question. A homework question asks for explanations, solutions, or assistance with images that are likely from educational materials.\n\nOutput your verdict in the following format:<decision>\n[yes/no]\n</decision>. Do NOT explain."
        self.prompt_template = "<user_prompt>\n{PROMPT}\n</user_prompt>"

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
        args = {"PROMPT": prompt["prompt"]}
        base64_image = get_image_file_from_gcs(prompt["image_hash"])
        conv = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_template.format(**args)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
        ]
        return conv

    def post_process(self, judgment):
        score = self.get_score(judgment=judgment)
        return {"homework": bool(score == "yes") if score else False}
