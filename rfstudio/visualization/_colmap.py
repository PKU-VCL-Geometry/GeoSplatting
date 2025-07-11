"""
This file copied with small modifications from:
 * https://github.com/colmap/colmap/blob/1a4d0bad2e90aa65ce997c9d1779518eaed998d5/scripts/python/read_write_model.py

TODO(1480) Delete this file when moving to pycolmap.


"""

# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

from __future__ import annotations

import base64
import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Literal, NoReturn, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import rfviser
import torch
from rfviser import transforms as tf

from rfstudio.data.dataparser.colmap_dataparser import read_images_binary, read_points3D_binary
from rfstudio.graphics.math import get_rotation_from_relative_vectors
from rfstudio.ui import console
from rfstudio.utils.webserver import open_webserver


def sql_command(conn, cmd):
    cur = conn.cursor()
    try:
        cur.execute(cmd)
        result = cur.fetchall()
    finally:
        cur.close()
    return result


class DatabaseParser:

    def __init__(self, database_path: Path):
        assert database_path.exists()
        conn = sqlite3.connect(database_path)
        try:
            self.match_kps = {}
            self.match_cnt = {}
            self.match_neighbors = {}
            cmd = 'SELECT pair_id, rows, cols, data FROM matches WHERE rows > 0'
            for idx, (pair_id, rows, cols, data) in enumerate(sql_command(conn, cmd)):
                id1, id2 = pair_id // 2147483647, pair_id % 2147483647
                assert len(data) == 8 * rows and cols == 2
                if id1 == id2:
                    continue
                self.match_kps[(id1, id2)] = [
                    (
                        struct.unpack('I', data[i * 8:i * 8 + 4])[0],
                        struct.unpack('I', data[i * 8 + 4:i * 8 + 8])[0],
                        False
                    )
                    for i in range(rows)
                ]
                self.match_cnt[(id1, id2)] = rows
                if id1 not in self.match_neighbors:
                    self.match_neighbors[id1] = []
                if id2 not in self.match_neighbors:
                    self.match_neighbors[id2] = []
            cmd = 'SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0'
            for idx, (pair_id, rows, cols, data) in enumerate(sql_command(conn, cmd)):
                id1, id2 = pair_id // 2147483647, pair_id % 2147483647
                assert len(data) == 8 * rows and cols == 2
                if id1 == id2:
                    continue
                values = {
                    (
                        struct.unpack('I', data[i * 8:i * 8 + 4])[0],
                        struct.unpack('I', data[i * 8 + 4:i * 8 + 8])[0],
                        False
                    )
                    for i in range(rows)
                }
                matches = self.match_kps[(id1, id2)]
                for i in range(len(matches)):
                    if matches[i] in values:
                        matches[i] = (matches[i][0], matches[i][1], True)
                total_match = self.match_cnt[(id1, id2)]
                self.match_cnt[(id1, id2)] = (rows, total_match)
                if rows > 30:
                    self.match_neighbors[id1].append((id2, rows / total_match))
                    self.match_neighbors[id2].append((id1, rows / total_match))
            cmd = 'SELECT image_id, rows, cols, data FROM keypoints'
            self.keypoints = {}
            for i, rows, cols, data in sql_command(conn, cmd):
                assert len(data) % rows == 0
                row_byte = len(data) // rows
                assert (row_byte - 8) * 8 in [0, 128] and cols * 4 == row_byte
                xys = []
                for j in range(rows):
                    x = int(struct.unpack('f', data[j*row_byte:j*row_byte+4])[0])
                    y = int(struct.unpack('f', data[j*row_byte+4:j*row_byte+8])[0])
                    xys.append((x, y))
                self.keypoints[i] = xys
            cmd = 'SELECT image_id, name FROM images'
            self.indices = { name: i for i, name in sql_command(conn, cmd) }
            self.image_names = { i: name for name, i in self.indices.items() }
            self.sorted_indices = [
                i
                for i, name in sorted(list(self.image_names.items()), key=lambda x: x[1])
            ]
            self.remap = { raw: now for now, raw in enumerate(self.sorted_indices) }
        finally:
            conn.close()

    def get_image_names(self):
        return list(self.indices.keys())

    def get_image_matches(self, name, voxel_size=2):
        results = []
        ref_id = self.indices[name]
        for src_id, score in self.match_neighbors.get(ref_id, []):
            downsample_map = {}
            for ref_kp, src_kp, valid in self.match_kps[(ref_id, src_id) if ref_id < src_id else (src_id, ref_id)]:
                if ref_id > src_id:
                    ref_kp, src_kp = src_kp, ref_kp
                ref_xy = self.keypoints[ref_id][ref_kp]
                key = (ref_xy[0] // voxel_size, ref_xy[1] // voxel_size)
                previous = downsample_map.get(key, None)
                if (previous is None) or (valid and not previous[2]):
                    downsample_map[key] = (ref_xy, self.keypoints[src_id][src_kp], valid)
            ref_xs = []
            ref_ys = []
            src_xs = []
            src_ys = []
            valids = []
            for ref_xy, src_xy, valid in downsample_map.values():
                ref_xs.append(ref_xy[0])
                ref_ys.append(ref_xy[1])
                src_xs.append(src_xy[0])
                src_ys.append(src_xy[1])
                valids.append(valid)
            results.append({
                "name": self.image_names[src_id],
                "score": score,
                "ref_x": ref_xs,
                "ref_y": ref_ys,
                "src_x": src_xs,
                "src_y": src_ys,
                "valid": valids
            })
        return results


def make_base64_image(
    path: Path,
    expected_size: int = 320 * 180,
) -> Tuple[str, Tuple[int, int], float]:
    assert path.exists(), path
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    rescale_factor = min(1, (expected_size / (w * h)) ** 0.5)
    w, h = int(w * rescale_factor), int(h * rescale_factor)
    base64_data = cv2.imencode(".jpg", cv2.resize(img, (w, h)))[1].tobytes()
    base64_image = "data:image/jpeg;base64," + base64.b64encode(base64_data).decode("ascii")
    return base64_image, (w, h), rescale_factor


def vis_colmap(
    path: Path,
    *,
    port: int = 6789,
    host: str = '0.0.0.0',
    auto_orient: bool = True,
    max_num_points: int = 40000,
    pcd_source: Optional[Path] = None,
    backend: Literal['viser', 'custom'] = 'custom'
) -> NoReturn:
    """Visualize colmap results"""

    with console.status("Loading..."):
        meta = read_images_binary(str(path / "sparse" / "0" / "images.bin"))
        xyz, rgb, _ = read_points3D_binary(str(path / "sparse" / "0" / "points3D.bin"))

    with console.status("Exporting multi-view data..."):
        poses = torch.stack([c2w for _, _, c2w, _ in meta])            # [N, 3, 4]
        image_names = [name for _, name, _, _ in meta]

        offset = xyz.mean(0)
        poses[:, :, 3] -= offset
        xyz -= offset
        if auto_orient:
            up = poses[:, :, 1].mean(0)
            up = up / torch.linalg.norm(up)
            rotation = get_rotation_from_relative_vectors(up, torch.tensor([0, 0, 1]).to(up))
            poses[:, :, :3] = rotation[None, :, :] @ poses[:, :, :3]
            poses[:, :, 3:] = rotation[None, :, :] @ poses[:, :, 3:]
            xyz = (rotation[None, :, :] @ xyz[:, :, None]).squeeze(-1)
        rescale = 0.9 / np.quantile(xyz.view(-1).abs().numpy(), 0.9)
        poses[:, :, 3] *= rescale                                      # scale to bbox [-1, 1]^3
        xyz *= rescale
        padding = torch.tensor([0, 0, 0, 1]).to(poses)
        poses = torch.cat((
            poses,
            padding.expand(poses.shape[0], 1, 4).contiguous(),
        ), dim=1)                                                      # [N, 4, 4]

        registered_set = set()
        json_lst = []
        for c2w, img_name in zip(poses, image_names):
            json_item = {}
            json_item["c2w"] = c2w.T.reshape(-1).tolist() if backend == 'custom' else c2w.numpy()
            base64_image, rescaled_shape, rescale_factor = make_base64_image(path / "images" / img_name)
            json_item["image"] = str(base64_image)
            json_item["name"] = img_name
            json_item["width"] = int(rescaled_shape[0])
            json_item["height"] = int(rescaled_shape[1])
            json_item["scale"] = float(rescale_factor)
            registered_set.add(img_name)
            json_lst.append(json_item)

        if backend == 'custom':
            database_path = path / "database.db"
            database_parser = DatabaseParser(database_path)
            unregistered_set = set(database_parser.get_image_names()) - registered_set
            for image_name in unregistered_set:
                base64_image, rescaled_shape, rescale_factor = make_base64_image(path / "images" / image_name)
                json_lst.append({
                    "c2w": [],
                    "image": str(base64_image),
                    "name": str(image_name),
                    "width": int(rescaled_shape[0]),
                    "height": int(rescaled_shape[1]),
                    "scale": float(rescale_factor)
                })
            sorted_indices = {
                name: i
                for i, name in sorted(
                    list(enumerate(database_parser.get_image_names())),
                    key=lambda x: x[1],
                )
            }
            sorted_json_lst = [None] * len(json_lst)
            for item in json_lst:
                matches = database_parser.get_image_matches(item['name'], voxel_size=int(16 / item['scale']))
                scale = 100 * item['scale']
                item['matches'] = [{
                    "idx": sorted_indices[match["name"]],
                    "score": match["score"],
                    "ref_x": [int(scale * p / item["width"]) for p in match["ref_x"]],
                    "ref_y": [int(scale * p / item["height"]) for p in match["ref_y"]],
                    "src_x": [int(scale * p / item["width"]) for p in match["src_x"]],
                    "src_y": [int(scale * p / item["height"]) for p in match["src_y"]],
                    "valid": match["valid"]
                } for match in matches]
                sorted_json_lst[sorted_indices[item['name']]] = item
        else:
            sorted_json_lst = list(sorted(json_lst, key=lambda x: x['name']))

    with console.status("Exporting pointcloud..."):
        if pcd_source is None:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(xyz.numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(rgb.numpy())
        else:
            point_cloud = o3d.io.read_point_cloud(str(pcd_source))
        if max_num_points * 4 < len(point_cloud.points):
            point_cloud = point_cloud.random_down_sample(max_num_points * 4 / len(point_cloud.points))
        indices = point_cloud.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.5)[1]
        point_cloud = point_cloud.select_by_index(indices)
        if max_num_points < len(point_cloud.points):
            point_cloud = point_cloud.random_down_sample(max_num_points / len(point_cloud.points))

    if backend == 'custom':
        with open_webserver('colmap', port=port, host=host) as basedir:
            with open(basedir / "colmap.json", "w", encoding="utf-8") as f:
                json.dump(sorted_json_lst, f)
            o3d.io.write_point_cloud(str(basedir / "pcd.ply"), point_cloud)
    else:
        server = rfviser.ViserServer(host=host, port=port)
        c2w = {}
        for item in sorted_json_lst:
            position = item['c2w'][:3, 3]
            rotation = item['c2w'][:3, :3]
            pose = tf.SE3.from_rotation_and_translation(
                tf.SO3.from_matrix(rotation).multiply(tf.SO3.from_x_radians(np.pi)),
                position,
            )
            c2w[item['name']] = pose.as_matrix()[:3, :]
            server.scene.add_camera_frustum(
                f'/cam/{item["name"]}',
                fov=np.pi / 2,
                aspect=4 / 3,
                color=(210 / 255, 143 / 255, 81 / 255),
                scale=0.005,
                wxyz=pose.rotation().wxyz,
                position=pose.translation()
            )
        prefix_len = len("data:image/jpeg;base64,")
        server.gui.add_image_viewer(
            images={
                item['name']: item['image'][prefix_len:]
                for item in sorted_json_lst
            },
            cameras={
                item['name']: c2w[item['name']]
                for item in sorted_json_lst
            }
        )
        server.scene.add_point_cloud(
            '/pcd',
            points=np.asarray(point_cloud.points),
            colors=np.asarray(point_cloud.colors),
            point_size=0.003,
        )
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
