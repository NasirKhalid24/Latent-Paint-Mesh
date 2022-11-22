"""
    Most code from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py
"""

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import PNDMScheduler, StableDiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class StableDiffusion(nn.Module):
    def __init__(self, device, min_t, max_t, token=None, model_name=None, revision=None):
        super().__init__()

        if token is None:
            try:
                with open('./TOKEN', 'r') as f:
                    self.token = f.read().replace('\n', '') # remove the last \n!
                    print(f'[INFO] successfully loaded hugging face user token!')
            except FileNotFoundError as e:
                print(e)
                print(f'[INFO] Please first create a file called TOKEN and copy your hugging face access token into it to download stable diffusion checkpoints.')
        else:
            self.token = token

        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_t)
        self.max_step = int(self.num_train_timesteps * max_t)

        print(f'[INFO] loading stable diffusion...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        if model_name is not None:
            self.vae = AutoencoderKL.from_pretrained(
                model_name,
                subfolder="vae",
                use_auth_token=self.token
            ).to(self.device)
        else:
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                use_auth_token=self.token
            ).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # 3. The UNet model for generating the latents.
        if model_name is not None:
            self.unet = UNet2DConditionModel.from_pretrained(
                model_name,
                subfolder="unet",
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet",
                revision="main" if revision is None else revision,
                use_auth_token=self.token
            ).to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt=[None]):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''] * len(prompt) if negative_prompt[0] is None else negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, latents, accum=1., guidance_scale=100):
        
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        with torch.no_grad():
            with torch.autocast("cuda"):
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)

                latent_model_input = torch.cat([latents_noisy] * 2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        w = (1- self.alphas[t])

        grad = (w * (noise_pred - noise)) / accum

        latents.backward(gradient=grad, retain_graph=True)

        return 0.

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents, factor):

        latents = (1 / factor) * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs, factor):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * factor

        return latents

    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    device = torch.device('cuda')

    sd = StableDiffusion(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()