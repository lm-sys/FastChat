FROM python:3.10.14-alpine

LABEL maintainer="solacowa@gmail.com"

RUN apk add gcc python3-dev musl-dev linux-headers

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install --no-cache-dir aiohttp fastapi httpx \
    markdown2[all] nh3 numpy prompt_toolkit>=3.0.0 \
    pydantic psutil requests rich>=10.0.0 \
    shortuuid tiktoken uvicorn

WORKDIR /app

COPY . /app/
RUN pip3 install -e .
RUN pip3 install pydantic

CMD ["python3", "-m", "fastchat.serve.controller", "--host", "0.0.0.0"]