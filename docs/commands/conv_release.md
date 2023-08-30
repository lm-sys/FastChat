## Chatbot Arena Conversations

1. Gather battles
```
python3 clean_battle_data.py --max-num 10 --mode conv_release
```

2. Tag OpenAI moderation
```
python3 tag_openai_moderation.py --in clean_battle_conv_20230814.json
```

3. Clean PII

4. Filter additional blocked words

```
python3 filter_bad_conv.py --in clean_battle_conv_20230630_tagged_v1_pii.json
```

5. Add additional toxicity tag


## All Conversations

1. Gather chats
```
python3 clean_chat_data.py
```

2. Sample
```
python3 conv_release_scripts/sample.py
```


## Prompt distribution

