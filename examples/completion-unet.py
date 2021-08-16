# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import sys
import subprocess
import argparse
import logging
import numpy as np
from time import time
import urllib
import glob
import visdom
from datetime import datetime
# Must be imported before large libsclass CollationAndTransformation:

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
torch.cuda.empty_cache() #empty cache memory
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.sampler import Sampler

import MinkowskiEngine as ME
#from reconstruction import InfSampler

import matplotlib.pyplot as plt



M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

assert (
    int(o3d.__version__.split(".")[1]) >= 8
), f"Requires open3d version >= 0.8, the current version is {o3d.__version__}"

if not os.path.exists("ModelNet40"):
    logging.info("Downloading the pruned ModelNet40 dataset...")
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])


###############################################################################
# Utility functions
###############################################################################
def PointCloud(points, colors=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


class CollationAndTransformation:
    def __init__(self, resolution):
        self.resolution = resolution

    def random_crop(self, coords_list):
        crop_coords_list = []
        for coords in coords_list:
            sel = coords[:, 0] < self.resolution / 3                    # one third reso as default for training
            crop_coords_list.append(coords[sel])
        return crop_coords_list

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        incomplete_coords = self.random_crop(coords)

        # Concatenate all lists
        return {
            "coords": ME.utils.batched_coordinates(coords),
            "xyzs": [torch.from_numpy(feat).float() for feat in feats],
            "cropped_coords": incomplete_coords,
            "labels": torch.LongTensor(labels),
        }

class OnlyCollationCopy:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        # Concatenate all lists
        return {
            "coords": ME.utils.batched_coordinates(coords),
            "xyzs": [torch.from_numpy(feat).float() for feat in feats],
            "labels": torch.LongTensor(labels),
        }
class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

def resample_mesh(mesh_cad, density=1):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
        (1 - np.sqrt(r[:, 0:1])) * A
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P

class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0

        #self.root = "./ModelNet40"
        self.base_name = "/home/eavise/MinkowskiEngine/ModelNet40"
        if (self.phase == "train"):
            filename = os.path.join(self.base_name, "train_modelnet40.txt")    #default filenames; "chair/train/*.off"
        elif (self.phase == "test"):
            filename = os.path.join(self.base_name, "test_modelnet40.txt")    #default filenames; "chair/train/*.off"
        with open(filename, "r") as f:
            lines = f.read().splitlines() 
        f.close()
        #bathtub bed bench bookshelf boottle chair cone cup curtain 
        # desk dresser flower_pot glass_box toilet stool sofa stairs table
        #car airplane person
        #guitat keyboard lamp mantel monitor night_stand piano plant range_hood 
        #sink sofa stairs stool table toilet tv_stand vase wardrobe xbox
        #fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = lines
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {phase} from {self.base_name} with {len(self.files)} files"
        )
        self.density = 30000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.base_name, self.files[idx])

        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            # Load a mesh, over sample, copy, rotate, voxelization
            #mesh_file = '/home/eavise/MinEngineDeve/ModelNet40/chair/train/chair_0312.off'
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
            vertices = np.asarray(pcd.vertices)

            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector(
                (vertices - vmin) / (vmax - vmin).max()
            )

            # Oversample points and copy
            xyz = resample_mesh(pcd, density=self.density)
            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if (
                cache_percent > 0
                and cache_percent % 10 == 0
                and cache_percent != self.last_cache_percent
            ):
                logging.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        # Use color or other features if available
        feats = np.ones((len(xyz), 1))

        if len(xyz) < 1000:
            logging.info(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        xyz = xyz * self.resolution
        #print("&&&&&&&&&&&&&&&&&&")
        #print(xyz)

        coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)
        #print("!!!!!!!!!!!!!!!")
        #print(xyz[inds])
        return (coords, xyz[inds], idx)

class ModelNet40ValidSubset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution

        #self.root = "./ModelNet40"
        self.base_name = "/home/eavise/MinkowskiEngine/ModelNet40"       

        filename = os.path.join(self.base_name, "val_modelnet40.txt")    #default filenames; "chair/train/*.off"
        with open(filename, "r") as f:
            lines = f.read().splitlines() 
        f.close()
        #bathtub bed bench bookshelf boottle chair cone cup curtain 
        # desk dresser flower_pot glass_box toilet stool sofa stairs table
        #car airplane person
        #guitat keyboard lamp mantel monitor night_stand piano plant range_hood 
        #sink sofa stairs stool table toilet tv_stand vase wardrobe xbox
        #fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = lines
        assert len(self.files) > 0, "No valid file loaded"
        self.density = 30000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.base_name, self.files[idx])

        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            # Load a mesh, over sample, copy, rotate, voxelization
            #mesh_file = '/home/eavise/MinEngineDeve/ModelNet40/chair/train/chair_0312.off'
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
            vertices = np.asarray(pcd.vertices)

            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector(
                (vertices - vmin) / (vmax - vmin).max()
            )

            # Oversample points and copy
            xyz = resample_mesh(pcd, density=self.density)
            self.cache[idx] = xyz
        # Use color or other features if available
        feats = np.ones((len(xyz), 1))
        if len(xyz) < 1000:
            logging.info(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        xyz = xyz * self.resolution

        coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)
        #print("!!!!!!!!!!!!!!!")
        #print(xyz[inds])
        return (coords, xyz[inds], idx)


def make_data_loader(
    phase, augment_data, batch_size, shuffle, num_workers, repeat, config
):
    dset = ModelNet40Dataset(phase, config=config)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": CollationAndTransformation(config.resolution),        
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader

def valid_data_loader(
    #phase, num_workers, config
    phase, batch_size, shuffle, num_workers, repeat, config
):
    batch_size = 2
    dset = ModelNet40ValidSubset(phase, config=config)

    # args = {
    #     "num_workers": num_workers,
    #     "pin_memory": False,
    #     "drop_last": False,
    # }
    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": OnlyCollationCopy(config.resolution),
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
       args["sampler"] = InfSampler(dset, shuffle)
    else:
       args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=64)
parser.add_argument("--epochs", type=int, default=1000)     #default 30000
parser.add_argument("--val_freq", type=int, default=5)      #default is 1000
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--stat_freq", type=int, default=5)
parser.add_argument("--weights", type=str, default="modelnet_completion.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)

###############################################################################
# End of utility functions
###############################################################################


class CompletionNet(nn.Module):

    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, resolution, in_nchannel=512):
        nn.Module.__init__(self)

        self.resolution = resolution

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=4,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in, target_key):
        out_cls, targets = [], []

        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        ##################################################
        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s32
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        target = self.get_target(dec_s32, target_key)
        targets.append(target)
        out_cls.append(dec_s32_cls)

        if self.training:
            keep_s32 += target

        # Remove voxels s32
        dec_s32 = self.pruning(dec_s32, keep_s32)

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        target = self.get_target(dec_s16, target_key)
        targets.append(target)
        out_cls.append(dec_s16_cls)

        if self.training:
            keep_s16 += target

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)

        target = self.get_target(dec_s8, target_key)
        targets.append(target)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        if self.training:
            keep_s8 += target

        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)

        target = self.get_target(dec_s4, target_key)
        targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        if self.training:
            keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)

        target = self.get_target(dec_s2, target_key)
        targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        if self.training:
            keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        target = self.get_target(dec_s1, target_key)
        targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)

        return out_cls, targets, dec_s1

class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        # Initialize the visualization environment
        self.vis = visdom.Visdom(env=self.env_name)
        self.vis.text('Plotting!')
        self.loss_win = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )

def training_run(net, train_dataloader, valid_dataloader, device, config):

    #vis = Visualizations()
    # Training loop
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    new_epochs = config.epochs
    weights_model_path = '/home/eavise/MinkowskiEngine/modelnet_completion0.pth'
    if os.path.exists(weights_model_path):
        checkpoint = torch.load(weights_model_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        new_epochs = config.epochs - checkpoint['epoch']
        print('&&&&&&&&&&&&&&&')
        print(config.epochs)
        # load_loss = checkpoint['loss']

    crit = nn.BCEWithLogitsLoss()

    net.train()
    train_iter = iter(train_dataloader)
    valid_iter = iter(valid_dataloader)

    logging.info(f"LR: {scheduler.get_lr()}")
    losses = []
    valid_losses = []
    valid_steps = []

    for i in range(new_epochs):

        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()

        in_feat = torch.ones((len(data_dict["coords"]), 1))

        sin = ME.SparseTensor(
            features=in_feat,
            coordinates=data_dict["coords"],
            device=device,
        )

        # Generate target sparse tensor
        cm = sin.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(data_dict["xyzs"]).to(device),
            string_id="target",
        )

        # Generate from a dense tensor
        out_cls, targets, sout = net(sin, target_key)
        num_layers, loss = len(out_cls), 0
        zip_losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
            zip_losses.append(curr_loss.item()/ num_layers)
            loss += curr_loss / num_layers
        avg_loss = np.sum(zip_losses)/len(zip_losses)
        losses.append(avg_loss)

        loss.backward()
        optimizer.step()
        t = time() - s

        # Visualization data in browser and logging 
        if i % config.stat_freq == 0:
            logging.info(
                f"Iter: {i}, Loss: {loss.item():.3e}, Data Loading Time: {d:.3e}, Tot Time: {t:.3e}"
            )

        # Valid dataset loss storage
        if i % config.val_freq == 0 and i > 0:
            validdata_dict = valid_iter.next()
            valid_steps.append(i)
            optimizer.zero_grad()

            valid_in_feat = torch.ones((len(validdata_dict["coords"]), 1))

            valid_sin = ME.SparseTensor(
                features=valid_in_feat,
                coordinates=validdata_dict["coords"],
                device=device,
            )

            # Generate target sparse tensor
            valid_cm = valid_sin.coordinate_manager
            valid_key, _ = valid_cm.insert_and_map(
                ME.utils.batched_coordinates(validdata_dict["xyzs"]).to(device),
                string_id="valid",
            )

            # Generate from a dense tensor
            valid_out_cls, valid_targets, valid_sout = net(valid_sin, valid_key)
            num_layers, loss = len(valid_out_cls), 0
            zip_losses = []
            for valid_out_cl, valid_target in zip(valid_out_cls, valid_targets):
                curr_loss = crit(valid_out_cl.F.squeeze(), valid_target.type(valid_out_cl.F.dtype).to(device))
                zip_losses.append(curr_loss.item()/ num_layers)
                loss += curr_loss / num_layers
            avg_loss = np.sum(zip_losses)/len(zip_losses)
            valid_losses.append(avg_loss)
        # save checkpoint every nth epoch
        if i % 100 == 0:            
            torch.save(
                {
                    "epoch": i,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss": loss,
                },
                "modelnet_completion{}.pth".format(i),
            )

            scheduler.step()
            logging.info(f"LR: {scheduler.get_lr()}")

            #net.train()
    plt.plot(losses,'-o')
    plt.plot(valid_steps, valid_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')

    plt.show()


def visualize(net, dataloader, device, config):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    for data_dict in dataloader:
        in_feat = torch.ones((len(data_dict["coords"]), 1))

        sin = ME.SparseTensor(
            features=in_feat,
            coordinates=data_dict["coords"],
            device=device,
        )

        # Generate target sparse tensor
        cm = sin.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(data_dict["xyzs"]).to(device),
            string_id="target",
        )

        # Generate from a dense tensor
        out_cls, targets, sout = net(sin, target_key)
        num_layers, loss = len(out_cls), 0
        for out_cl, target in zip(out_cls, targets):
            None
            #loss += (
            #   crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
            #    / num_layers
            #)

        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        for b, (coords, feats, target) in enumerate(zip(batch_coords, batch_feats, targets)):
            predicted_pcd = PointCloud(coords.cpu())
            #pcd.estimate_normals()
            predicted_pcd.translate([1.2 * config.resolution, 0, 0])
            predicted_pcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))
            predicted_pcd.scale(100 / np.max(predicted_pcd.get_max_bound() - predicted_pcd.get_min_bound()),
                     center=predicted_pcd.get_center())
            predicted_pcd.paint_uniform_color([0.5, 0.3, 0.3])
            predicted_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(predicted_pcd,
                                                                        voxel_size=0.6)

            cropped_pcd = PointCloud(data_dict["cropped_coords"][b])
            #gtpointSet.points = o3d.utility.Vector3dVector(input_pcd.cpu())
            cropped_pcd.translate([0, 0, 0])
            cropped_pcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))
            cropped_pcd.scale(100 / np.max(cropped_pcd.get_max_bound() - cropped_pcd.get_min_bound()),
                     center=cropped_pcd.get_center())
            cropped_pcd.paint_uniform_color([0.3, 0.5, 0.3])
            cropped_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cropped_pcd,
                                                                        voxel_size=0.6)

            input_pcd = data_dict["coords"].numpy()[:, 0:4]
            mask = (input_pcd[:,0]==b)
            final_pcd = input_pcd[np.where(input_pcd[:,0]==b)]
            print("###############")
            print(final_pcd)
            #print(b)
            gt_pointSet = o3d.geometry.PointCloud()
            gt_pointSet.points = o3d.utility.Vector3dVector(final_pcd[:, 1:4])
            gt_pointSet.translate([-1.2 * config.resolution, 0, 0])
            gt_pointSet.rotate(M, np.array([[0.0], [0.0], [0.0]]))
            gt_pointSet.scale(100 / np.max(gt_pointSet.get_max_bound() - gt_pointSet.get_min_bound()),
                    center=gt_pointSet.get_center())
            gt_pointSet.paint_uniform_color([0.3, 0.3, 0.5])
            gt_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(gt_pointSet,
                                                                        voxel_size=0.6)

            #o3d.visualization.draw_geometries([pcd, gtpointSet, opcd])
            def rotate_view(vis):
                   ctr = vis.get_view_control()
                   ctr.rotate(10.0, 0.0)
                   return False
            o3d.visualization.draw_geometries([gt_voxel_grid, cropped_voxel_grid, predicted_voxel_grid])
            #o3d.visualization.draw_geometries_with_animation_callback([gt_pointSet, cropped_pcd, predicted_pcd], rotate_view)



if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.eval = True
    if not config.eval:
        train_dataloader = make_data_loader(
            "train",
            augment_data=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config,
        )
        validset_dataloader = valid_data_loader(
            #num_workers = config.num_workers, 
            #config = config,
            "valid",
            #augment_data=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config,
        )
    else:
        train_dataloader = make_data_loader(
            "test",
            augment_data=False,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            repeat=False,
            config=config,
        )
        validset_dataloader = None

    in_nchannel = len(train_dataloader.dataset)
    net = CompletionNet(config.resolution, in_nchannel=in_nchannel)
    net.to(device)

    logging.info(net)

    if not config.eval:
        training_run(net, train_dataloader, validset_dataloader, device, config)
    else:
        if not os.path.exists(config.weights):
            logging.info(f"Downloaing pretrained weights. This might take a while...")
            urllib.request.urlretrieve(
                "https://bit.ly/36d9m1n", filename=config.weights
            )

        logging.info(f"Loading weights from {config.weights}")
        checkpoint = torch.load(config.weights)
        #net = nn.DataParallel(net)
        net.load_state_dict(checkpoint["state_dict"])

        visualize(net, train_dataloader, device, config)
