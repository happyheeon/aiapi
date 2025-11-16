from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_name = "Qwen/Qwen1.5-1.8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": result}
