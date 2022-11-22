import os
import torch
import random
import imageio
import warnings
import argparse
import torchvision

import matplotlib.pyplot    as plt
import nvdiffrast.torch     as dr

from tqdm           import tqdm
from resize_right   import resize

from utils.sd       import StableDiffusion
from utils.obj      import load_obj, write_obj
from utils.camera   import blurs, CameraBatch, log_matrix
from utils.video    import Video

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
    parser.add_argument('--lr',         help='Learning rate', type=float, default=0.01)

    parser.add_argument('--scale',      help='Factor by which to scale up 64x64 texture map', type=int, default=2)
    parser.add_argument('--log',        help='Log every x epochs, set to -1 to disable logging (much faster)', type=int, default=-1)

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

    parser.add_argument('--epochs',     help='How many iterations to run for', type=int, default=2500)
    parser.add_argument('--guidance',   help=' Guidance scale for cfg', type=float, default=100.)

    args = vars(parser.parse_args())

    DEVICE = "cuda:" + str(args["gpu"])
    MESH = args["mesh"]
    MATERIAL = args["material"]
    PROMPT = args["text"]
    SCALE = args["scale"]
    ITER_LOG = args["log"]
    DIST_RANGE = [args["dist_min"], args["dist_max"]]
    AZIM_RANGE = [args["azim_min"], args["azim_max"]]
    ELEV_RANGE = [args["elev_min"], args["elev_max"]]
    FOV_RANGE  = [args["fov_min"], args["fov_max"]]
    OFFSET     = args["offset"]
    PROMPT_AUG = args["prompt_aug"]
    EPOCHS     = args["epochs"]
    GUIDANCE_SCALE = args["guidance"]

    device = torch.device(DEVICE)
    mesh_tensor = load_obj(MESH, device)
    stable_diffusion = StableDiffusion(device, min_t=0.02, max_t=0.98, revision="fp16")

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
        with torch.no_grad():
            material = stable_diffusion.encode_imgs(material.unsqueeze(0), 0.18215)[0]
        material = material.permute(1, 2, 0).contiguous()

    material = material.clone().requires_grad_(True)
    text_embeddings  = [
        stable_diffusion.get_text_embeds([PROMPT]),
        stable_diffusion.get_text_embeds(["side view of %s" % PROMPT]),
        stable_diffusion.get_text_embeds(["back view of %s" % PROMPT]),
    ]

    optimizers = []
    optimizers.append(torch.optim.Adam([material], lr=args["lr"]))
    os.makedirs("./output", exist_ok=True)
    if ITER_LOG > 0:
        video = Video("./output")

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

        rast_out, _ = dr.rasterize(glctx, points, mesh_tensor["f"], resolution=[64, 64])

        texc, _ = dr.interpolate(mesh_tensor["uv"][None, ...], rast_out, mesh_tensor["f_uv"])
        color = dr.texture(material[None, ...], texc, filter_mode='linear')
        back  = torch.rand_like(color) * -(1-torch.clamp(rast_out[..., -1:], 0, 1))
        # back  = blurs[random.randint(0, 3)](torch.rand_like(color).permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * -(1-torch.clamp(rast_out[..., -1:], 0, 1))
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)
        color = color + 0.9*back

        color = color.permute(0, 3, 1, 2)

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
                azim_log += 1.
                dec_tex = stable_diffusion.decode_latents(material.detach().clone().unsqueeze(0).permute(0, 3, 1, 2), factor=0.18215).permute(0, 2, 3 ,1).contiguous()
                
                points = torch.matmul(
                    torch.cat([mesh_tensor["v"], torch.ones([mesh_tensor["v"].shape[0], 1], device=mesh_tensor["v"].device)], dim=1),
                    torch.tensor(log_mvp.T, device=mesh_tensor["v"].device)
                )[None, ...]
                
                rast_out, _ = dr.rasterize(glctx, points, mesh_tensor["f"], resolution=[512, 512])
                texc, _ = dr.interpolate(mesh_tensor["uv"][None, ...], rast_out, mesh_tensor["f_uv"])
                log_color = dr.texture(
                    dec_tex,
                    texc,
                    filter_mode='linear'
                )
                log_color = log_color * torch.clamp(rast_out[..., -1:], 0, 1)
                log_img = video.ready_image(torchvision.utils.make_grid([
                    log_color.permute(0, 3, 1, 2)[0],
                    resize(dec_tex.permute(0, 3, 1, 2), out_shape=(log_color.shape[1], log_color.shape[2]))[0]
                ]).permute(1, 2, 0))

                if args["colab"]:
                    plt.figure()
                    plt.imshow(log_img)
                    plt.show()

    final_texture = material.detach().clone().unsqueeze(0).permute(0, 3, 1, 2)
    final_texture = stable_diffusion.decode_latents(
        final_texture,
        factor=0.18215
    )
    torchvision.utils.save_image(final_texture, "output_map.png")
    write_obj("./output", mesh_tensor, final_texture)
    if ITER_LOG > 0:
        video.close()

if __name__ == '__main__':
    main()