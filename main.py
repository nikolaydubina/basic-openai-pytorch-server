# import requests
import os
import json
import logging
import typing


# from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI
from transformers import MllamaForConditionalGeneration, AutoProcessor

port = int(os.environ.get("PORT", 80))
model_id = os.environ.get("MODEL_ID", "")

stats = {
    "torch.version": torch.__version__,
    "torch.version.cuda": torch.version.cuda,
    "torch.cuda.current_device": torch.cuda.current_device(),
    "torch.torch.cuda.device_count": torch.torch.cuda.device_count(),
    "torch.cuda.device_name": torch.cuda.get_device_name(0),
    "model_id": model_id,
    "http.server.port": port,
}
print(json.dumps(stats))

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
    return stats


@app.get("/v1/chat/completions")
async def chat_completions():
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Does user message contain vulgar language? Reply only T/F for True/False.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                # {"type": "image"},
                {
                    "type": "text",
                    "text": "beautiful morning",
                },
            ],
        },
    ]
    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)

    return {"message": processor.decode(output[0])}


if __name__ == "__main__":
    uvicorn.run(app, port=port)
