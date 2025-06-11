from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline
import torch
import os

app = FastAPI()

# Optional: load Hugging Face token from env
hf_token = os.getenv("HF_TOKEN")

# Load the SD 3.5 model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    use_auth_token=hf_token,
    torch_dtype=torch.float16
).to("cuda")

# Input format for POST requests
class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_image(request: GenerateRequest):
    image = pipe(
        prompt=request.prompt,
        num_inference_steps=30,
        guidance_scale=4.5,
        height=1024,
        width=1024,
    ).images[0]
    
    image_path = "output.png"
    image.save(image_path)

    return {
        "message": "Image generated successfully!",
        "image_url": "/output.png"
    }
