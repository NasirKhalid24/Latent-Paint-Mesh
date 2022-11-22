import glm
import torch
import random
import numpy as np
import torchvision.transforms as transforms

blurs = [
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(2, 2))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(2, 2))
    ]),
]

def persp_proj(fov_x=45, ar=1, near=1.0, far=50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)
    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan( (fov_rad / 2) )
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)
    
    return proj_mat

def log_matrix(fov, dist, elev, azim, look_at=[0, 0, 0], up=[0, -1, 0]):

    proj_mtx = persp_proj(fov)

    modl = glm.mat4()
    
    cam_z = dist * np.cos(np.radians(elev)) * np.sin(np.radians(azim))
    cam_y = dist * np.sin(np.radians(elev))
    cam_x = dist * np.cos(np.radians(elev)) * np.cos(np.radians(azim))
    
    view  = glm.lookAt(
        glm.vec3(cam_x, cam_y, cam_z),
        glm.vec3(look_at[0], look_at[1], look_at[2]),
        glm.vec3(up[0], up[1], up[2]),
    )
    
    r_mv = view * modl
    r_mv = np.array(r_mv.to_list()).T

    return np.matmul(proj_mtx, r_mv).astype(np.float32)

class CameraBatch(torch.utils.data.Dataset):
    def __init__(
        self,
        image_resolution,
        distances,
        azimuths,
        elevation_params,
        fovs,
        aug_loc, 
        bs,
        look_at=[0, 0, 0], up=[0, -1, 0]
    ):

        self.res = image_resolution

        self.dist_min = distances[0]
        self.dist_max = distances[1]

        self.azim_min = azimuths[0]
        self.azim_max = azimuths[1]

        self.fov_min = fovs[0]
        self.fov_max = fovs[1]
        
        self.elev_min = elevation_params[0]
        self.elev_max   = elevation_params[1]

        self.aug_loc   = aug_loc

        self.look_at = look_at
        self.up = up

        self.batch_size = bs

    def __len__(self):
        return self.batch_size
        
    def __getitem__(self, index):

        azim_ang = np.random.uniform( self.azim_min, self.azim_max+1.0 )
        alev_ang = np.random.uniform( self.elev_min, self.elev_max+1.0 )

        if (azim_ang % 180) > 120:
            prompt_idx = 2
        elif (azim_ang % 180) < 60:
            prompt_idx = 0
        else:
            prompt_idx = 1

        azim = np.radians( azim_ang )
        elev = np.radians( alev_ang )
        dist = np.random.uniform( self.dist_min, self.dist_max )
        fov = np.random.uniform( self.fov_min, self.fov_max )
        
        proj_mtx = persp_proj(fov)
        
        # Generate random view
        cam_z = dist * np.cos(elev) * np.sin(azim)
        cam_y = dist * np.sin(elev)
        cam_x = dist * np.cos(elev) * np.cos(azim)
        
        if self.aug_loc:

            # Random offset
            limit  = self.dist_min // 2
            rand_x = np.random.uniform( -limit, limit )
            rand_y = np.random.uniform( -limit, limit )

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

        else:
        
            modl = glm.mat4()
            
        view  = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(self.look_at[0], self.look_at[1], self.look_at[2]),
            glm.vec3(self.up[0], self.up[1], self.up[2]),
        )

        r_mv = view * modl
        r_mv = np.array(r_mv.to_list()).T

        mvp     = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos  = np.linalg.inv(r_mv)[:3, 3]

        return {
            'mvp': torch.from_numpy( mvp ).float(),
            'campos': torch.from_numpy( campos ).float(),
            'prompt': prompt_idx
        }