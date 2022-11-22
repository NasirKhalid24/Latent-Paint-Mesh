import os
import imageio
import numpy as np
import torch

##  Simplified version of https://github.com/NVlabs/nvdiffmodeling/blob/main/src/obj.py
def load_obj(filename, device, clear_ks=True, mtl_override=None):
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename) as f:
        lines = f.readlines()

    # load vertices
    vertices, texcoords, normals  = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])

    # load faces
    faces, tfaces = [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            for i in range(nv - 2): # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])

    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device=device) if len(texcoords) > 0 else None

    faces = torch.tensor(faces, dtype=torch.int32, device=device)
    tfaces = torch.tensor(tfaces, dtype=torch.int32, device=device) if texcoords is not None else None
    
    return {
        "v": vertices,
        "uv": texcoords,
        "f": faces,
        "f_uv": tfaces,
    }

def write_obj(folder, mesh, diffuse_texture):
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = mesh["v"].detach().cpu().numpy() if mesh["v"] is not None else None
        # v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_nrm = None
        v_tex = mesh["uv"].detach().cpu().numpy() if mesh["uv"] is not None else None

        t_pos_idx = mesh["f"].detach().cpu().numpy() if mesh["f"] is not None else None
        # t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_nrm_idx = None
        t_tex_idx = mesh["f_uv"].detach().cpu().numpy() if mesh["f_uv"] is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        print("    writing %d texcoords" % len(v_tex))
        if v_tex is not None:
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        # print("writing %d normals" % len(v_nrm))
        if v_nrm is not None:
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    mtl_file = os.path.join(folder, 'mesh.mtl')
    print("Writing material: ", mtl_file)

    folder = os.path.dirname(mtl_file)
    with open(mtl_file, "w") as f:
        f.write('newmtl defaultMat\n')
        f.write('bsdf   %s\n' % "diffuse")
        f.write('map_kd texture_kd.png\n')
        imageio.imwrite(os.path.join(folder, 'texture_kd.png'), np.clip(np.rint(diffuse_texture[0].permute(1, 2, 0).cpu().detach().clone().numpy() * 255.0), 0, 255).astype(np.uint8))

    print("Done exporting mesh")