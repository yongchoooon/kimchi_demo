import os   
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda:0"

# 사전 학습된 stable diffusion 모델 로드
pretrained_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_path, 
    torch_dtype=torch.float16,
    cache_dir='caches'
)

# LoRA 가중치 경로
dataset_name = "outputs_yd/korean-4-datasets"
step=37450

lora_path = f"./{dataset_name}/checkpoint-{step}/pytorch_lora_weights.safetensors"  

# 학습에 사용된 CLIP 모델 로드
clip_model_path = "Bingsu/clip-vit-large-patch14-ko"
text_encoder = CLIPTextModel.from_pretrained(
    clip_model_path,
    cache_dir='caches'
)
text_encoder.to(device)
tokenizer = CLIPTokenizer.from_pretrained(
    clip_model_path,
    cache_dir='caches'
)

# CLIP 모델을 파이프라인에 설정
pipeline.text_encoder = text_encoder
pipeline.tokenizer = tokenizer

pipeline.load_lora_weights(lora_path)

pipeline = pipeline.to(device)

# 인퍼런스
# prompt = "박물관 안에 그릇에 담긴 떡 모형이 전시되어 있는 곳에서, 쌀을 주식으로 먹는 지역에서 발달한 음식은 떡이고, 동그란 그릇 안에 들어간 떡의 모양은 동그라미입니다."
prompt = "Traditional Korean temples"

with torch.autocast(device):
    image = pipeline(prompt, num_inference_steps=50).images[0]

syn_images_dir = os.path.join(dataset_name, 'syn_images')
os.makedirs(syn_images_dir, exist_ok=True)

# 결과 이미지 출력
save_image_path = os.path.join(syn_images_dir, f"{prompt}_.jpg")
image.save(save_image_path, 'JPEG')