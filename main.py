from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import PIL.Image as Image
import torch
from typing import Union
from fastapi import FastAPI, Form, UploadFile, File
from typing_extensions import Annotated
from fastapi.staticfiles import StaticFiles
import uuid
import pathlib
from PIL import Image

app = FastAPI()

app.mount("/upload", StaticFiles(directory="upload"), name="upload")


@app.get("/")
def read_root():
    return {"success": bool(1), "message": "success", "data": {}}


@app.post("/upload")
def upload(file: UploadFile = File(...), prompt: str = Form(), numInferenceSteps: int = Form()):
    uuidFileName = str(uuid.uuid4())
    fileName = "./upload/" + \
        uuidFileName + pathlib.Path(file.filename).suffix
    fileNameOutput = "./upload/" + \
        "output-" + uuidFileName + pathlib.Path(file.filename).suffix
    print(fileName, fileNameOutput, prompt, numInferenceSteps)
    try:
        contents = file.file.read()
        with open(fileName, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"success": bool(0), "message": "There was an error uploading the file"}
    finally:
        file.file.close()

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            controlnet=controlnet,
            safety_checker=None
        ).to("cuda")

        pipe.enable_attention_slicing()
        generator = torch.Generator(device="cuda").manual_seed(-1)

        image = pipe(
            prompt,
            generator=generator,
            image=Image.open(fileName),
            num_inference_steps=50
        ).images[0]

        image.save(fileNameOutput)

    return {"success": bool(1), "message": f"Successfully {fileNameOutput}"}
