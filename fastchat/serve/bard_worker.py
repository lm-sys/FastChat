"""
Adapted from https://github.com/acheong08/Bard.
"""
import argparse
import json
import random
import re
import string

import requests

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional


class ConversationState(BaseModel):
    conversation_id: str = ""
    response_id: str = ""
    choice_id: str = ""
    req_id: int = Field(default_factory=lambda: int("".join(random.choices(string.digits, k=4))))


class Message(BaseModel):
    content: str
    state: ConversationState


class Response(BaseModel):
    content: str
    factualityQueries: Optional[List[dict]]
    textQuery: Optional[str]
    choices: List[dict]
    state: ConversationState


class Chatbot:
    """
    A class to interact with Google Bard.
    Parameters
        session_id: str
            The __Secure-1PSID cookie.
    """

    __slots__ = [
        "headers",
        "_reqid",
        "SNlM0e",
        "conversation_id",
        "response_id",
        "choice_id",
        "session",
    ]

    def __init__(self, session_id):
        headers = {
            "Host": "bard.google.com",
            "X-Same-Domain": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": "https://bard.google.com",
            "Referer": "https://bard.google.com/",
        }
        self.session = requests.Session()
        self.session.headers = headers
        self.session.cookies.set("__Secure-1PSID", session_id)
        self.SNlM0e = self.__get_snlm0e()

    def __get_snlm0e(self):
        resp = self.session.get(url="https://bard.google.com/", timeout=10)
        # Find "SNlM0e":"<ID>"
        if resp.status_code != 200:
            raise Exception("Could not get Google Bard")
        SNlM0e = re.search(r"SNlM0e\":\"(.*?)\"", resp.text).group(1)
        return SNlM0e

    def ask(self, message: Message) -> Response:
        """
        Send a message to Google Bard and return the response.
        :param message: The message to send to Google Bard.
        :return: A dict containing the response from Google Bard.
        """
        # url params
        params = {
            #"bl": "boq_assistant-bard-web-server_20230315.04_p2",
            # This is a newer API version
            "bl": "boq_assistant-bard-web-server_20230507.20_p2",
            "_reqid": str(message.state.req_id),
            "rt": "c",
        }

        # message arr -> data["f.req"]. Message is double json stringified
        message_struct = [
            [message.content],
            None,
            [
                message.state.conversation_id,
                message.state.response_id,
                message.state.choice_id
            ],
        ]
        data = {
            "f.req": json.dumps([None, json.dumps(message_struct)]),
            "at": self.SNlM0e,
        }

        # do the request!
        resp = self.session.post(
            "https://bard.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate",
            params=params,
            data=data,
            timeout=120,
        )

        chat_data = json.loads(resp.content.splitlines()[3])[0][2]
        if not chat_data:
            return Response(
                content=f"Google Bard encountered an error: {resp.content}.",
                factualityQueries=[],
                textQuery="",
                choices=[],
                state=message.state,
            )
        json_chat_data = json.loads(chat_data)
        conversation = ConversationState(
            conversation_id=json_chat_data[1][0],
            response_id=json_chat_data[1][1],
            choice_id=json_chat_data[4][0][0],
            req_id=message.state.req_id + 100000,
        )
        return Response(
            content=json_chat_data[0][0],
            factualityQueries=json_chat_data[3],
            textQuery=json_chat_data[2][0] if json_chat_data[2] is not None else "",
            choices=[{"id": i[0], "content": i[1]} for i in json_chat_data[4]],
            state=conversation,
        )


app = FastAPI()
chatbot = None


@app.on_event("startup")
async def startup_event():
    global chatbot
    chatbot = Chatbot("<__Secure-1PSID cookie of bard.google.com>")


@app.post("/chat", response_model=Response)
async def chat(message: Message):
    response = chatbot.ask(message)
    return response


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser("Google Bard worker")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18900)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", reload=True)
