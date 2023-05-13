"""
Adapted from https://github.com/acheong08/Bard.
"""
import argparse
import json
import random
import re
import string

from fastapi import FastAPI
import httpx
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import uvicorn


class ConversationState(BaseModel):
    conversation_id: str = ""
    response_id: str = ""
    choice_id: str = ""
    req_id: int = 0


class Message(BaseModel):
    content: str
    state: ConversationState = Field(default_factory=ConversationState)


class Response(BaseModel):
    content: str
    factualityQueries: Optional[List]
    textQuery: Optional[Union[str, List]]
    choices: List[dict]
    state: ConversationState


class Chatbot:
    """
    A class to interact with Google Bard.
    Parameters
        session_id: str
            The __Secure-1PSID cookie.
    """

    def __init__(self, session_id):
        headers = {
            "Host": "bard.google.com",
            "X-Same-Domain": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": "https://bard.google.com",
            "Referer": "https://bard.google.com/",
        }
        self.session = httpx.AsyncClient()
        self.session.headers = headers
        self.session.cookies.set("__Secure-1PSID", session_id)
        self.SNlM0e = None

    async def _get_snlm0e(self):
        resp = await self.session.get(url="https://bard.google.com/", timeout=10)
        # Find "SNlM0e":"<ID>"
        if resp.status_code != 200:
            raise Exception("Could not get Google Bard")
        SNlM0e = re.search(r"SNlM0e\":\"(.*?)\"", resp.text).group(1)
        return SNlM0e

    async def ask(self, message: Message) -> Response:
        """
        Send a message to Google Bard and return the response.
        :param message: The message to send to Google Bard.
        :return: A dict containing the response from Google Bard.
        """
        if message.state.conversation_id == "":
            message.state.req_id = int("".join(random.choices(string.digits, k=4)))
        # url params
        params = {
            # "bl": "boq_assistant-bard-web-server_20230315.04_p2",
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
                message.state.choice_id,
            ],
        ]
        data = {
            "f.req": json.dumps([None, json.dumps(message_struct)]),
            "at": self.SNlM0e,
        }

        # do the request!
        resp = await self.session.post(
            "https://bard.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate",
            params=params,
            data=data,
            timeout=60,
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
    cookie = json.load(open("bard_cookie.json"))
    chatbot = Chatbot(cookie["__Secure-1PSID"])
    chatbot.SNlM0e = await chatbot._get_snlm0e()


@app.post("/chat", response_model=Response)
async def chat(message: Message):
    response = await chatbot.ask(message)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Google Bard worker")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=18900)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "bard_worker:app", host=args.host, port=args.port, log_level="info",
        reload=args.reload
    )
