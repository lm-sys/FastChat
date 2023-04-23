from fastchat import client

completion = client.ChatCompletion.create(
    model="vicuna-7b-v1.1",
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "What's your favorite food?"},
        {
            "role": "assistant",
            "content": "As an AI language model, I don't have personal preferences or emotions. However, I can provide information about food. If you have any specific cuisine or dish in mind, I can tell you more about it.",
        },
        {"role": "user", "content": "What's your recommendation?"},
    ],
)

print(completion.choices[0].message)
