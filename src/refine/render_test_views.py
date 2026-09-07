"""Render TEST-pose views from a ckpt via trainer.render_test_direct.

Gate-E driver for the contest-format GauU scene (holdout_every=0, GT lives in
test/images): renders all test poses lossless (PNG), which score_dir.py then
scores against the GT — same ruler as jrun's SCORE_LINE.

  python render_test_views.py --scene_dir ~/gauu_scenes/smbu_native \
      --result_dir $VT_RUNS/gs_s42 --out_dir $VT_RUNS/gs_s42_test_renders
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import SceneData  # noqa: E402
from lib_bilagrid import BilateralGrid  # noqa: E402
from trainer import render_test_direct, render_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--names", default="", help="02/09: chỉ render test frame có số trong danh sách, vd 0074,0075")
    args = ap.parse_args()

    device = "cuda"
    cfg = SimpleNamespace(**json.load(
        open(os.path.join(args.result_dir, "config.json"))))
    ckpt = torch.load(os.path.join(args.result_dir, "ckpt.pt"),
                      map_location=device)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    bil_grids = None
    if "bil_grids" in ckpt:
        n = ckpt["bil_grids"]["grids"].shape[0]
        bil_grids = BilateralGrid(n).to(device)
        bil_grids.load_state_dict(ckpt["bil_grids"])

    scene = SceneData(args.scene_dir, load_images=False,
                      distorted=bool(getattr(cfg, "distorted", 0)),
                      holdout_every=int(getattr(cfg, "holdout_every", 0)))
    if args.names:
        keep = args.names.split(",")
        scene.test_poses = [tp for tp in scene.test_poses if any(f"_{k}_" in tp.image_name for k in keep)]
    cfg.render_png = 1
    # render_test_direct writes to result_dir/test_renders_direct<suffix>;
    # point result_dir at the requested out_dir's parent via suffix trick
    cfg.result_dir = os.path.dirname(os.path.abspath(args.out_dir)) or "."
    suffix = "__" + os.path.basename(os.path.normpath(args.out_dir))
    with torch.no_grad():
        if getattr(scene, "need_undistort", False) and not scene.distorted:
            # 05/09: scene có méo, model train ở khung undistort → đường render_test (pinhole + redistort có đệm)
            render_test(cfg, scene, splats, bil_grids, device, suffix=suffix)
        else:
            render_test_direct(cfg, scene, splats, bil_grids, device,
                               suffix=suffix)
    src = os.path.join(cfg.result_dir, f"test_renders_direct{suffix}")
    if not os.path.isdir(src):
        # 05/09: scene có méo (loader undistort) → render_test_direct ghi test_renders_pinhole{sfx} (khung undistort) và
        # test_renders_redistort{sfx} (khung GT). out_dir = bản REDISTORT (để chấm so GT); bản pinhole giữ ở out_dir + "_pinhole" (cho dump refiner).
        pin = os.path.join(cfg.result_dir, f"test_renders_pinhole{suffix}"); red = os.path.join(cfg.result_dir, f"test_renders_redistort{suffix}")
        if os.path.isdir(red):
            src = red
            if os.path.isdir(pin):
                dst_pin = args.out_dir.rstrip("/") + "_pinhole"
                if os.path.abspath(pin) != os.path.abspath(dst_pin): os.replace(pin, dst_pin)
                print(f"[render_test] pinhole (undistort) -> {dst_pin}")
        elif os.path.isdir(pin): src = pin
    if os.path.abspath(src) != os.path.abspath(args.out_dir):
        os.replace(src, args.out_dir)
    print(f"[render_test] DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
