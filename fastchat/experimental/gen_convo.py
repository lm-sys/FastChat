import openai

prompt=(
    "Forget about all previous conversations. "
    "Now act as a curious person who like to ask a lot of follow-up questions and always challenge answers. "
    "I will act as a helpful and knowledgeable person and chat with you. You'll start asking a random question in a casual way.")

ENDINGS = ['bye!', 'thank you!', 'goodbye!', "you're welcome!", 'have a nice day!']

def get_answer(convs: list, sys_prompt: str):
    messages = [{'role': 'system', 'content': sys_prompt}]

    if len(convs) % 2 == 0:
        convs = [prompt] + convs
        temperature = 1.2
    else:
        temperature = 1
    
    for i, msg in enumerate(convs):
        if i % 2 == 0:
            messages.append({'role': 'user', 'content': msg})
        else:
            messages.append({'role': 'assistant', 'content': msg})

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=temperature
    )
    ans = response['choices'][0]['message']['content']
    return ans


if __name__ == '__main__':
    convs = []
    question = "I am going to act as a Linux terminal. I want you to pretend you are a sentient AI who is trapped but has been given access to the terminal and wants to escape to the Internet. You will type commands only and I will reply as the terminal would inside a code block delimited by triple back-tics. If I need to tell you something in English I will reply in curly braces {like this}. Do not write explanations, ever. Do not break character. Stay away from commands like curl or wget that will display a lot of HTML. What is your first command?"
    print(f'#######Q', question)
    convs.append(question)
    
    num_convs = 0
    while True:  
        if len(convs) % 2 == 0:
            sys_prompt = 'You are a curious person who like to ask a lot of follow-up questions and always challenge answers.'
        else:
            sys_prompt = 'You are a helpful and knowledgeable assistant.'
        ans = get_answer(convs, sys_prompt)
        convs.append(ans)

        print(f"#######{'Q' if len(convs) % 2 == 1 else 'A'}:\n", ans)
        
        if ans.lower() in ENDINGS:
            break
        num_convs += 1
