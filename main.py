import requests
import os
import json
import logging
import typing
from PIL import Image
from pydantic import BaseModel
import torch
import uvicorn
from fastapi import FastAPI
from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig

port = int(os.environ.get("PORT", 80))
model_id = os.environ.get("MODEL_ID", "")


def stats():
    return {
        "torch.version": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "torch.cuda.current_device": torch.cuda.current_device(),
        "torch.torch.cuda.device_count": torch.torch.cuda.device_count(),
        "torch.cuda.device_name": torch.cuda.get_device_name(0),
        "model_id": model_id,
        "http.server.port": port,
    }


print(json.dumps(stats()))


model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="auto",
)
model.tie_weights()
processor = AutoProcessor.from_pretrained(model_id)

uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(lambda q: False)  # disable API endpoint logs

app = FastAPI()


@app.get("/healthz")
async def healthz():
    return stats()


class TextContent(BaseModel):
    type: str
    text: str


class ImageContent(BaseModel):
    type: str
    image_url: str


class Message(BaseModel):
    role: str
    content: list[TextContent | ImageContent]


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    temperature: float = 1
    top_p: float = 1
    max_tokens: int = 20
    frequency_penalty: float = 0.1
    schema: typing.Any = None


class ChatCompletionResult(BaseModel):
    text: str


schema_prompt = "Reply only JSON object that conforms to the following schema: "
mark_header_assistant = "<|start_header_id|>assistant<|end_header_id|>"
mark_end = "<|eot_id|>"


def extract_assistant_response(s: str) -> str:
    s = s[s.find(mark_header_assistant) + len(mark_header_assistant) :]
    return s.removesuffix(mark_end)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    if req.schema is not None:
        req.messages[0].content.append(TextContent(type="text", text=schema_prompt + json.dumps(req.schema)))

    input_text = processor.apply_chat_template(req.messages, add_generation_prompt=True)
    inputs = processor(text=input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    gen_config = GenerationConfig(temperature=req.temperature, top_p=req.top_p, frequency_penalty=req.frequency_penalty)
    output = model.generate(**inputs, max_new_tokens=req.max_tokens, generation_config=gen_config)
    result = processor.decode(output[0])

    return ChatCompletionResult(text=extract_assistant_response(result))


if __name__ == "__main__":
    uvicorn.run(app, port=port)
