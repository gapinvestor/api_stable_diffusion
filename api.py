# # Instalación de las librerías necesarias
# pip install fastapi     
# pip install diffusers   
# pip install wget
# pip install transformers
# pip install acelerate


def descargar_modelo_od():
    '''
    Es importante tener en cuenta que, por los comando usados esta función está preparada
    para ser usada en un sistema operativo Linux/Ubuntu.
    '''
    import os
    import time
    from IPython.display import clear_output
    import wget

    if os.path.exists('/content/stable-diffusion-v1-5'):
        !rm -r /content/stable-diffusion-v1-5
    clear_output()

    %cd /content/
    clear_output()
    !mkdir /content/stable-diffusion-v1-5
    %cd /content/stable-diffusion-v1-5
    !git init
    !git lfs install --system --skip-repo
    !git remote add -f origin  "https://huggingface.co/runwayml/stable-diffusion-v1-5"
    !git config core.sparsecheckout true
    !echo -e "scheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json\n!*.safetensors" > .git/info/sparse-checkout
    !git pull origin main
    if os.path.exists('/content/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin'):
        !git clone "https://huggingface.co/stabilityai/sd-vae-ft-mse"
        !mv /content/stable-diffusion-v1-5/sd-vae-ft-mse /content/stable-diffusion-v1-5/vae
        !rm -r /content/stable-diffusion-v1-5/.git
        %cd /content/stable-diffusion-v1-5
        !rm model_index.json
        time.sleep(1)    
        wget.download('https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json')
        !sed -i 's@"clip_sample": false@@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json
        !sed -i 's@"trained_betas": null,@"trained_betas": null@g' /content/stable-diffusion-v1-5/scheduler/scheduler_config.json
        !sed -i 's@"sample_size": 256,@"sample_size": 512,@g' /content/stable-diffusion-v1-5/vae/config.json  
        %cd /content/
        clear_output()
        print('[1;32mDONE !')


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

pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", revision='fp16', torch_dtype=torch.float16)
pipe.to("cuda")

@api.get("/")
def generador(prompt: str):
    # Ajustamos el prompt para que no nos salga en negro
    prompt = prompt + ' --precision full --no-half --medvram'
    with autocast("cuda"):
        imagen = pipe(prompt, guidance_scale=7.5).images[0]

    imagen.save('prompt_v2.png')
    buffer = BytesIO()
    imagen.save(buffer, format='PNG')
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr) # , media_type="image/png"