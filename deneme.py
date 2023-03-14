
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import PIL.Image as Image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", 
    torch_dtype=torch.float16)
    
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    controlnet=controlnet, 
    safety_checker=None).to("cuda") 

pipe.enable_attention_slicing()
generator = torch.Generator(device="cuda").manual_seed(-1)
prompt = "minimalistic cyberpunk kitchen, interior, zaha hadid, wood finish, render, 8k, photorealistic, 3ds max"

image = pipe(
    prompt,
    width=1200,
    height=600,
    generator=generator,
    image=Image.open("test1.jpeg"),
    num_inference_steps=50
).images[0]

image.save('output.png')