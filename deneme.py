
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
prompt = "Space station, pro photography, RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"

image = pipe(
    prompt,
    width=1200,
    height=600,
    generator=generator,
    image=Image.open("test.jpeg"),
    num_inference_steps=50
).images[0]

image.save('output.png')