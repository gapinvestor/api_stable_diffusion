# # Installation of the necessary libraries # #
# pip install fastapi     
# pip install diffusers   
# pip install wget
# pip install transformers
# pip install acelerate
# pip install nest_asyncio
# pip install pyngrok
# pip install uvicorn



def download_od_model():
    '''
    Download stable-diffusion-v1-5 model.
    '''
    import os
    import time
    from IPython.display import clear_output
    import wget

    if not os.path.exists("/content/stable-diffusion-v1-5"):
        os.mkdir("/content/stable-diffusion-v1-5")

    os.chdir("/content/stable-diffusion-v1-5")

    os.system("git init")
    os.system("git lfs install --system --skip-repo")
    os.system("git remote add -f origin https://huggingface.co/runwayml/stable-diffusion-v1-5")
    os.system("git config core.sparsecheckout true")

    with open(".git/info/sparse-checkout", "w") as f:
        f.write("scheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json\n!*.safetensors")

    os.system("git pull origin main")

    if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
        os.system("git clone https://huggingface.co/stabilityai/sd-vae-ft-mse")
        os.system("mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae")
        os.system("rm -r /content/stable-diffusion-v1-5/.git")
        os.chdir("/content/stable-diffusion-v1-5")
        os.system("rm model_index.json")
        time.sleep(1)
        wget.download('https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json')
        os.system('''sed -i 's@"clip_sample": false@@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json''')
        os.system('''sed -i 's@"trained_betas": null,@"trained_betas": null@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json''')
        os.system('''sed -i 's@"sample_size": 256,@"sample_size": 512,@g' /content/stable-diffusion-v1-5/vae/config.json''')
        os.chdir("/content/")
        clear_output()
        print('DONE !')


#########################################
##### PREPARE AND INITIALIZE THE API ####
#########################################
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

download_od_model()
pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", revision='fp16', torch_dtype=torch.float16)
pipe.to("cuda")

@api.get("/")
def generator(prompt: str):
    '''
    Generate an image from the received prompt.
    Receive:
        - prompt : str
    Return:
        - image : png (saved on the server)
        - image : base64 (displayed to the user)
    '''
    # Ajustamos el prompt para que no nos salga en negro (parece que finalmente no hace falta)
    # prompt = prompt + ' --precision full --no-half --medvram'
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5).images[0]

    file_name = prompt.replace(' ','_')
    image.save(f'{file_name}.png')
    # Return the encoded image in base64
    # Use https://codebeautify.org/base64-to-image-converter to uncode the image
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr) # media_type="image/png"