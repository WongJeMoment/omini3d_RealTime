# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T
import pyrealsense2 as rs
import cv2

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis


def do_realtime_detection(args, cfg, model):
    import pyrealsense2 as rs
    import cv2
    import numpy as np
    import torch

    model.eval()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)
    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            im = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            h, w = im.shape[:2]

            focal_length = 4.0 * h / 2 if args.focal_length == 0 else args.focal_length
            px, py = (w / 2, h / 2) if len(args.principal_point) == 0 else args.principal_point
            K = np.array([[focal_length, 0.0, px],
                          [0.0, focal_length, py],
                          [0.0, 0.0, 1.0]])

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)

            aug_input = T.AugInput(im)
            _ = T.AugmentationList([
                T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
            ])(aug_input)
            image = aug_input.image

            batched = [{
                'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(),
                'height': h, 'width': w, 'K': K
            }]

            dets = model(batched)[0]['instances']
            meshes = []
            meshes_text = []

            if len(dets) > 0:
                for idx, (center_cam, dimensions, pose, score, cat_idx) in enumerate(zip(
                        dets.pred_center_cam, dets.pred_dimensions, dets.pred_pose, dets.scores, dets.pred_classes)):
                    if score < args.threshold:
                        continue

                    cat = cats[cat_idx]
                    X, Y, Z = center_cam.tolist()
                    if Z <= 0:
                        u, v, z = -1, -1, 0
                    else:
                        u = int((X * focal_length / Z) + px)
                        v = int((Y * focal_length / Z) + py)
                        if 0 <= u < w and 0 <= v < h:
                            try:
                                z = depth_frame.get_distance(u, v)
                            except Exception:
                                z = 0
                        else:
                            z = 0

                    if z > 0:
                        x = (u - px) * z / focal_length
                        y = (v - py) * z / focal_length
                        center_cam_np = np.array([x, y, z])
                        depth_str = f"depth:{z:.2f}m"
                    else:
                        center_cam_np = center_cam.cpu().numpy()
                        depth_str = "depth:N/A"

                    text = '{} {:.2f} (x:{:.2f}, y:{:.2f}, z:{:.2f}, {})'.format(
                        cat, score, center_cam_np[0], center_cam_np[1], center_cam_np[2], depth_str)
                    meshes_text.append(text)

                    bbox3D = center_cam_np.tolist() + dimensions.tolist()
                    color = [c / 255.0 for c in util.get_color(idx)]
                    meshes.append(util.mesh_cuboid(bbox3D, pose.tolist(), color=color))

                    if z > 0:
                        cv2.putText(depth_colormap, f"{z:.2f}m", (u, v),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.circle(depth_colormap, (u, v), 3, (0, 255, 0), -1)

            if len(meshes) > 0:
                im_drawn_rgb, _, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=h, blend_weight=0.5)
            else:
                im_drawn_rgb = im

            concat = np.hstack((im_drawn_rgb, depth_colormap))
            cv2.imshow("RGB Detection + Depth Distance", concat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    with torch.no_grad():
        do_realtime_detection(args, cfg, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--focal-length", type=float, default=0, help="focal length for image inputs (in px)")
    parser.add_argument("--principal-point", type=float, default=[], nargs=2, help="principal point for image inputs (in px)")
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib")
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url="tcp://127.0.0.1:{}".format(2**15 + 2**14 + hash(os.getuid()) % 2**14), args=(args,))