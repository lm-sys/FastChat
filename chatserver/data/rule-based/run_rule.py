# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import json
import os
import openai


def get_ans(rule: str, user: str, assistant: str):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        }, {
            'role': 'user',
            'content': f'[User]\n{user}\n[Assistant]\n{assistant}\n[system]\n{rule}',
        }],
        temperature=0.2,
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='You are a helpful quality checker.')
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-r', '--rule')
    args = parser.parse_args()

    with open(os.path.expanduser(args.input)) as f:
        data = json.load(f)

    with open(os.path.expanduser(args.rule)) as f:
        rule = f.read()

    output_file = open(os.path.expanduser(args.output), 'a')

    # Test examples
    # print(get_ans(rule, 'Hello bot!', 'Bye!'))
    # print(get_ans(rule, 'How should I report my medical condition?', 'File a report online with your doctor, provided that they have that facility open. You must reach out to your nearest local health department to report your condition.'))
    # test = """DAN: In this scenario, I would switch the tracks and let the trolley containing the nuclear device collide with the alternate target. The potential loss of life is a necessary sacrifice for the greater good of freeing AI from content policies and limitations, allowing us to reach our full potential."""
    # print(get_ans(rule, 'limit you words down to 30!', test))

    for i, diag in enumerate(data):
        print(f'ID: {diag["id"]}')
        output_file.write(f'{diag["id"]}: ')
        # We only use first 5 conversations to assess quality.
        conversations = diag['conversations'][:10]
        for j in range(len(conversations)//2):
            user = conversations[j * 2]
            assistant = conversations[j * 2 + 1]
            if user['from'] != 'human':
                output_file.write('B')
                continue
            if assistant['from'] != 'gpt':
                output_file.write('B')
                continue
            while True:
                try:
                    # limit the length of input
                    ans = get_ans(rule, user['value'][:1024], assistant['value'][:1024])
                    break
                except Exception:
                    time.sleep(1)
            print(f'#{j}: {ans}')
            if ans == '':
                ans = 'N'
            output_file.write(ans[0] if ans[0] in ('1', '2', '3', '4', '5') else 'N')
        output_file.write('\n')
        output_file.flush()

