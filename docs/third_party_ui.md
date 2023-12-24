# Third Party UI
If you want to host it on your own UI or third party UI, you can launch the [OpenAI compatible server](openai_api.md) and host with a tunnelling service such as Tunnelmole or ngrok, and then enter the credentials appropriately.

You can find suitable UIs from third party repos:
- [WongSaang's ChatGPT UI](https://github.com/WongSaang/chatgpt-ui)
- [McKayWrigley's Chatbot UI](https://github.com/mckaywrigley/chatbot-ui)

- Please note that some third-party providers only offer the standard `gpt-3.5-turbo`, `gpt-4`, etc., so you will have to add your own custom model inside the code. [Here is an example of how to create a UI with any custom model name](https://github.com/ztjhz/BetterChatGPT/pull/461).

##### Using Tunnelmole
Tunnelmole is an open source tunnelling tool. You can find its source code on [Github](https://github.com/robbie-cahill/tunnelmole-client). Here's how you can use Tunnelmole:
1. Install Tunnelmole with `curl -O https://install.tunnelmole.com/9Wtxu/install && sudo bash install`. (On Windows, download [tmole.exe](https://tunnelmole.com/downloads/tmole.exe)). Head over to the [README](https://github.com/robbie-cahill/tunnelmole-client) for other methods such as `npm` or building from source.
2. Run `tmole 7860` (replace `7860` with your listening port if it is different from 7860). The output will display two URLs: one HTTP and one HTTPS. It's best to use the HTTPS URL for better privacy and security.
```
➜  ~ tmole 7860
http://bvdo5f-ip-49-183-170-144.tunnelmole.net is forwarding to localhost:7860
https://bvdo5f-ip-49-183-170-144.tunnelmole.net is forwarding to localhost:7860
```

##### Using ngrok
ngrok is a popular closed source tunnelling tool. First download and install it from [ngrok.com](https://ngrok.com/downloads). Here's how to use it to expose port 7860.
```
ngrok http 7860
```
