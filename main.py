import comfy.options
comfy.options.enable_args_parsing()

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

from utils import create_empty_latent, common_ksampler, load_upscale_model, set_last_layer, encode_clip, upscale_image_by, upscale_with_model, encode, load_lora
import torch
import cv2
import numpy as np
from comfy.cli_args import args
import comfy.model_management

out = comfy.sd.load_checkpoint_guess_config("DarkRevPikas_v30_pruned.safetensors", output_vae=False, output_clip=True, embedding_directory="/")
upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


clip = out[1]
model = out[0]

# create empy latent
print("creating empty latent...")
height = 704 
width = 512 
batch = 1


# prompt
print("setting up prompt...")
p_text = "photorealistic, (hyperrealistic:1.2), beautiful, masterpiece, best quality, extremely detailed face, dress, perfect lighting, full body, large breasts, wide hips, thick thighs, plump, detailed eye makeup, detail face, nice detailed eyes, heavy eye makeup, white hair, purple eyes, white_legwear, (detached_sleeves:1.2), emilia \\(re:zero\\), emilia \\(re:zero\\), (light white short hair), detailed hands and fingers, white flower, green chest jewel, nature, landscape, mountains, medieval, scenery, falling petals, flowers,"

n_text = "(worst quality, low quality:1.3), (monochrome), zombie, watermark, username,patreon username, patreon logo, (extra fingers, deformed hands, polydactyl:1.2)"


denoise = 1.0
disable_noise = False
start_step = None
last_step = None
force_full_denoise = False
seed = 1828262939
steps = 20
cfg = 7.0
sampler_name = comfy.samplers.SAMPLER_NAMES[1]
scheduler = comfy.samplers.SCHEDULER_NAMES[0]

with torch.inference_mode():
    # crear una imagen vacia (empty lantet o torch tensor)
    latent = create_empty_latent(width, height, batch)

    clip_skip = -2
    clip_layer = set_last_layer(clip, clip_skip)


    lora_model, lora_clip = load_lora(model, './emilia.safetensors', clip_layer, 1.0, .10)

    positive = encode_clip(lora_clip, p_text)
    negative = encode_clip(lora_clip, n_text)

    # generar ruido sobre la imagen vacia 
    a = common_ksampler(lora_model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)

    torch.cuda.empty_cache()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    # decodificar la imagen
    vae_file = comfy.utils.load_torch_file("kl-f8-anime2_fp16.safetensors")
    vae = comfy.sd.VAE(sd=vae_file)
    decode = vae.decode(a["samples"])

    # escalar la resolucion
    upscale_model = load_upscale_model("./4x-AnimeSharp.pth")
    b = upscale_with_model(upscale_model, decode)

    c = upscale_image_by(b, upscale_methods[0], 0.5)

    encode = encode(vae, c)

    e = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, encode, denoise=0.5)

    j = vae.decode(e["samples"])

    # convertir la imagen tensor a imagen comun
    final = j.squeeze(0)

    torch.cuda.empty_cache()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


    i = 255. * final.cpu().numpy()

    img = np.clip(i, 0, 255).astype(np.uint8)

    # Convert from BGR to RGB
    img = img[:, :, ::-1]

    cv2.imwrite("image.png", img)