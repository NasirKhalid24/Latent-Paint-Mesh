import os
import math
import torch
import imageio
import warnings
import argparse
import torchvision

import matplotlib.pyplot    as plt
import nvdiffrast.torch     as dr

from tqdm           import tqdm
from datetime       import datetime
from resize_right   import resize

from utils.sd           import StableDiffusion
from utils.obj          import load_obj, write_obj
from utils.camera       import blurs, CameraBatch, log_matrix
from utils.video        import Video
from utils.empatches    import EMPatches

warnings.filterwarnings("ignore")


def main():
    # Orientation according to blender axis
    # +X is "front", -X is "back"
    # +Y is left side, -Y is right side

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',        help='GPU index', type=int, default=0)
    parser.add_argument('--mesh',       help='Path to .obj must have uv unwrapped', type=str)
    parser.add_argument('--material',   help='Path to starting material, if none will randomly initialize', type=str, default=None)
    parser.add_argument('--text',       help='Text prompt', type=str)
    parser.add_argument('--lr',         help='Learning rate', type=float, default=0.1)

    parser.add_argument('--scale',      help='Factor by which to scale up 64x64 texture map', type=int, default=3)
    parser.add_argument('--log',        help='Log every x epochs, set to -1 to disable logging (much faster)', type=int, default=250)

    parser.add_argument('--dist_min',   help='Minimum camera distance', type=float, default=1.0)
    parser.add_argument('--dist_max',   help='Maximum camera distance', type=float, default=1.5)

    parser.add_argument('--azim_min',   help='Minimum camera azimuth in degrees', type=float, default=0.)
    parser.add_argument('--azim_max',   help='Maximum camera azimuth in degrees', type=float, default=360.)

    parser.add_argument('--elev_min',   help='Minimum camera elevation in degrees wrt x plane (use -x and +x for elevation)', type=float, default=0.)
    parser.add_argument('--elev_max',   help='Maximum camera elevation in degrees wrt x plane (use -x and +x for elevation)', type=float, default=30.)

    parser.add_argument('--fov_min',    help='Minimum FOV in degrees', type=float, default=30.)
    parser.add_argument('--fov_max',    help='Maximum FOV in degrees', type=float, default=45.)

    parser.add_argument('--offset',     help='Offset mesh from center', action='store_true')
    parser.add_argument('--colab',      help='Enable for google colab logging', action='store_true')
    parser.add_argument('--prompt_aug', help='Prompt augmentatio (side, back)', action='store_true')
    parser.add_argument('--auto_cam',   help='Autoset camera parameters', action='store_true')

    parser.add_argument('--epochs',     help='How many iterations to run for', type=int, default=2500)
    parser.add_argument('--guidance',   help=' Guidance scale for cfg', type=float, default=0.8)

    args = vars(parser.parse_args())

    DEVICE = "cuda:" + str(args["gpu"])
    device = torch.device(DEVICE)

    MESH = args["mesh"]
    MATERIAL = args["material"]
    PROMPT = args["text"]
    SCALE = args["scale"]
    ITER_LOG = args["log"]

    mesh_tensor = load_obj(MESH, device)

    if args["auto_cam"] == False:
        DIST_RANGE = [args["dist_min"], args["dist_max"]]
        AZIM_RANGE = [args["azim_min"], args["azim_max"]]
        ELEV_RANGE = [args["elev_min"], args["elev_max"]]
        FOV_RANGE  = [args["fov_min"], args["fov_max"]]
    else:
        max_ = mesh_tensor["v"].abs().max().item()
        
        DIST_RANGE = [max_ + 0.2, max_ + 0.8]
        AZIM_RANGE = [0., 360.]
        ELEV_RANGE = [0., 30.]
        FOV_RANGE  = [30., 90.]

    OFFSET     = args["offset"]
    PROMPT_AUG = args["prompt_aug"]
    EPOCHS     = args["epochs"]
    GUIDANCE_SCALE = args["guidance"]

    emp = EMPatches()
    stable_diffusion = StableDiffusion(device, min_t=0.1, max_t=0.9, revision="fp16")

    if MATERIAL is None or not os.path.exists(MATERIAL):
        material = torch.randn(
            [64*SCALE // 1, 64*SCALE // 1, 4],
            device=device,
        )
    else:
        material = torchvision.transforms.ToTensor()(
            imageio.imread(MATERIAL, pilmode='RGB')
        )
        material = resize(material, out_shape=(int(64*SCALE*8), int(64*SCALE*8))).to(device)
        latent_patches, _ = emp.extract_patches(material.clone().detach().permute(1, 2, 0), patchsize=512, overlap=0.)
        _, decoded_idxs   = emp.extract_patches(torch.rand((
            int(SCALE*64),
            int(SCALE*64),
            4
        )), patchsize=64, overlap=0.)
        
        with torch.no_grad():
            for idx, ptch in enumerate(tqdm(latent_patches, leave=True)):
                latent_patches[idx] = stable_diffusion.encode_imgs(
                    ptch.unsqueeze(0).permute(0, 3, 1, 2), 0.18215
                )[0].cpu().permute(1, 2, 0)
        material = emp.merge_patches(latent_patches, decoded_idxs)
        material = torch.from_numpy(material).to(device)

    latent_patches, _ = emp.extract_patches(material.clone().detach(), patchsize=64, overlap=0.)
    _, decoded_idxs   = emp.extract_patches(torch.rand((
        int(math.sqrt(len(latent_patches))*512),
        int(math.sqrt(len(latent_patches))*512),
        3
    )), patchsize=512, overlap=0.)

    material = material.clone().requires_grad_(True)
    text_embeddings  = [
        stable_diffusion.get_text_embeds([PROMPT]),
        stable_diffusion.get_text_embeds(["side view of %s" % PROMPT]),
        stable_diffusion.get_text_embeds(["back view of %s" % PROMPT]),
    ]

    optimizers = []
    optimizers.append(torch.optim.Adam([material], lr=args["lr"]))

    now = datetime.now()
    randid = now.strftime("%m-%d-%Y_%H-%M-%S") + PROMPT[8:].replace(" ", "_")
    out_pth = os.path.join("./output", randid)

    os.makedirs(out_pth, exist_ok=True)
    if ITER_LOG > 0:
        video = Video(out_pth)

    glctx = dr.RasterizeGLContext()

    cameras = torch.utils.data.DataLoader(
        CameraBatch(
            image_resolution=64,
            distances=DIST_RANGE,
            azimuths=AZIM_RANGE,
            elevation_params=ELEV_RANGE,
            fovs=FOV_RANGE,
            aug_loc=OFFSET,
            bs=1
        ),
        1,
        num_workers=0,
        pin_memory=True
    )

    azim_log = 0.
    t_loop = tqdm(range(EPOCHS), leave=False)
    for it in t_loop:

        for optimizer in optimizers:
            optimizer.zero_grad()

        params_camera = next(iter(cameras))
        params_camera["mvp"] = params_camera["mvp"].to(device)

        points = torch.matmul(
            torch.cat([mesh_tensor["v"], torch.ones([mesh_tensor["v"].shape[0], 1], device=mesh_tensor["v"].device)], dim=1),
            params_camera["mvp"][0].T
        )[None, ...]

        rast_out, _ = dr.rasterize(glctx, points, mesh_tensor["f"], resolution=[256, 256])

        texc, _ = dr.interpolate(mesh_tensor["uv"][None, ...], rast_out, mesh_tensor["f_uv"])
        color = dr.texture(material[None, ...], texc, filter_mode='linear')
        back  = torch.rand_like(color) * -(1-torch.clamp(rast_out[..., -1:], 0, 1))
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)
        color = color + 0.9*back

        color = color.permute(0, 3, 1, 2)
        color = resize(color, out_shape=(64, 64))

        _ = stable_diffusion.train_step(
            text_embeddings=text_embeddings[0 if PROMPT_AUG else params_camera["prompt"][0].item()],
            latents=color,
            guidance_scale=GUIDANCE_SCALE
        )

        for optimizer in optimizers:
            optimizer.step()

        if it % ITER_LOG == 0 and ITER_LOG > 0:
            with torch.no_grad():
                log_mvp = log_matrix(50, 1.3, 10., azim_log)
                azim_log += 5.

                latent_patches, _ = emp.extract_patches(material.clone().detach(), patchsize=64, overlap=0.)

                for idx, ptch in enumerate(tqdm(latent_patches, leave=True)):
                    latent_patches[idx] = stable_diffusion.decode_latents(ptch.unsqueeze(0).permute(0, 3, 1, 2), factor=0.18215).permute(0, 2, 3 ,1).cpu()

                dec_tex = emp.merge_patches(latent_patches, decoded_idxs)
                dec_tex = torch.from_numpy(dec_tex).to(mesh_tensor["v"].device)
                
                points = torch.matmul(
                    torch.cat([mesh_tensor["v"], torch.ones([mesh_tensor["v"].shape[0], 1], device=mesh_tensor["v"].device)], dim=1),
                    torch.tensor(log_mvp.T, device=mesh_tensor["v"].device)
                )[None, ...]
                
                rast_out, _ = dr.rasterize(glctx, points, mesh_tensor["f"], resolution=[512, 512])
                texc, _ = dr.interpolate(mesh_tensor["uv"][None, ...], rast_out, mesh_tensor["f_uv"])
                log_color = dr.texture(
                    dec_tex[None, ...],
                    texc,
                    filter_mode='linear'
                )
                log_color = log_color * torch.clamp(rast_out[..., -1:], 0, 1)
                log_color = torchvision.utils.make_grid([
                    log_color.permute(0, 3, 1, 2)[0],
                    resize(dec_tex[None, ...].permute(0, 3, 1, 2), out_shape=(log_color.shape[1], log_color.shape[2]))[0]
                ])

                torchvision.utils.save_image(log_color, os.path.join(out_pth, "%d.png" % it))
                log_img = video.ready_image(log_color.permute(1, 2, 0))

                if args["colab"]:
                    plt.figure()
                    plt.imshow(log_img)
                    plt.show()

    latent_patches, _ = emp.extract_patches(material.clone().detach(), patchsize=64, overlap=0.)

    for idx, ptch in enumerate(tqdm(latent_patches)):
        latent_patches[idx] = stable_diffusion.decode_latents(ptch.unsqueeze(0).permute(0, 3, 1, 2), factor=0.18215).permute(0, 2, 3 ,1).cpu()

    dec_tex = emp.merge_patches(latent_patches, decoded_idxs)
    final_texture = torch.from_numpy(dec_tex).unsqueeze(0).permute(0, 3, 1, 2).to(mesh_tensor["v"].device)
    
    # torchvision.utils.save_image(final_texture, "output_map.png")
    write_obj(os.path.join(out_pth), mesh_tensor, final_texture)
    if ITER_LOG > 0:
        video.close()

if __name__ == '__main__':
    main()