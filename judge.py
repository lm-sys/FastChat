import json
import random
import openai

# Set your OpenAI API key
openai.api_key = 'sk-xxxxxxx'

# Define the judge prompt
judge_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
"""

# Load JSON data
with open('responses.json', 'r') as f:
    data = json.load(f)

# Initialize counters for verdicts
model_verdict_counts = {'agent_response': 0, 'standard_response': 0, 'Tie': 0, 'Error': 0}
total_entries = len(data)

for idx, entry in enumerate(data):
    user_question = entry['prompt']
    response1 = entry['agent_response']
    response2 = entry['standard_response']

    # Randomly assign responses to Assistant A and Assistant B
    responses = [
        {'response': response1, 'label': 'agent_response'},
        {'response': response2, 'label': 'standard_response'}
    ]
    random.shuffle(responses)
    assistant_A = responses[0]
    assistant_B = responses[1]

    assistant_A_response = assistant_A['response']
    assistant_A_label = assistant_A['label']
    assistant_B_response = assistant_B['response']
    assistant_B_label = assistant_B['label']

    # Construct the full prompt
    full_prompt = f"""{judge_prompt}

User Question:
{user_question}

Assistant A's response:
{assistant_A_response}

Assistant B's response:
{assistant_B_response}
"""

    # Get the evaluation from the GPT model
    try:
        completion = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=500,
            temperature=0,
        )

        assistant_reply = completion.choices[0].message['content']

        # Extract the verdict
        verdict = ''
        if '[[A]]' in assistant_reply:
            verdict = 'A'
        elif '[[B]]' in assistant_reply:
            verdict = 'B'
        elif '[[C]]' in assistant_reply:
            verdict = 'C'
        else:
            verdict = 'Error'
            model_verdict_counts['Error'] += 1  # Increment error count
            verdict_label = 'Error'

        # Map the verdict back to the original models
        if verdict == 'A':
            winning_label = assistant_A_label
            model_verdict_counts[winning_label] += 1
            verdict_label = winning_label
        elif verdict == 'B':
            winning_label = assistant_B_label
            model_verdict_counts[winning_label] += 1
            verdict_label = winning_label
        elif verdict == 'C':
            model_verdict_counts['Tie'] += 1
            verdict_label = 'Tie'

        # Output the result for each entry
        print(f"Entry {idx+1}/{total_entries}")
        print(f"User Question: {user_question}")
        print(f"A={assistant_A_label}'s Response: {assistant_A_response}")
        print(f"B={assistant_B_label}'s Response: {assistant_B_response}")
        print(f"Verdict: {assistant_reply}")
        print(f"Better response: {verdict_label}")
        print()

    except Exception as e:
        # Handle any exceptions, such as API errors
        print(f"Entry {idx+1}/{total_entries}")
        print(f"User Question: {user_question}")
        print(f"Error: {str(e)}")
        print()
        model_verdict_counts['Error'] += 1

# Calculate percentages
total_valid_verdicts = total_entries - model_verdict_counts['Error']
percentage_agent = (model_verdict_counts['agent_response'] / total_valid_verdicts) * 100 if total_valid_verdicts else 0
percentage_standard = (model_verdict_counts['standard_response'] / total_valid_verdicts) * 100 if total_valid_verdicts else 0
percentage_tie = (model_verdict_counts['Tie'] / total_valid_verdicts) * 100 if total_valid_verdicts else 0
percentage_error = (model_verdict_counts['Error'] / total_entries) * 100

# Output the percentages
print("Verdict Percentages:")
print(f"Agent Response Wins: {percentage_agent:.2f}%")
print(f"Standard Response Wins: {percentage_standard:.2f}%")
print(f"Ties: {percentage_tie:.2f}%")
print(f"Errors: {percentage_error:.2f}%")
