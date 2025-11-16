# main.py
import os
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 환경변수 로드
load_dotenv()
API_KEY = os.getenv("API_KEY")

# FastAPI 앱 생성
app = FastAPI(title="AI API with API Key")

# 모델 로드 (경량 모델, CPU 모드)
model_name = "distilgpt2"  # 무료 플랜에 적합
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 => CPU

# 엔드포인트: /generate
@app.post("/generate")
def generate_text(prompt: str, key: str = Query(...)):
    # API Key 검증
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # 모델 생성
    output = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
    response_text = output[0]["generated_text"]
    
    return {"response": response_text}
