"""Harness tái hiện đúng pipeline.py của BTC: gieo hạt + TORCH_HOME + gọi refiner_data.main().
   python run_infer.py <scene> <model_dir> <refiner.pt> <out_dir> [--limit N] [--depth_mode ed]"""
import os, sys, time
sys.path.insert(0, "/home/tensara/projects/vt-track1/contest/src"); import _paths
scene, model, ref, out = sys.argv[1:5]; extra = sys.argv[5:]
os.environ["TORCH_HOME"] = "/home/tensara/projects/vt-track1/contest/work/models"
import cv2, torch
_S=int(os.environ.get("SEED","0")); cv2.setRNGSeed(_S); torch.manual_seed(_S); print(f"[seed] {_S}")
# timestamp từng dòng log để đo per-view
class TS:
    def __init__(s, o): s.o, s.b, s.t0 = o, "", time.time()
    def write(s, x):
        s.b += x
        while "\n" in s.b:
            l, s.b = s.b.split("\n", 1); s.o.write(f"[{time.time()-s.t0:8.2f}] {l}\n")
    def flush(s): s.o.flush()
sys.stdout = TS(sys.stdout)
import refiner_data
argv = ["--result_dir", model, "--scene_dir", scene, "--targets", "test", "--dump", out + "_dump",
        "--no_dump", "1", "--pad", "1", "--render_aa", "1", "--K", "4", "--depth_fix", "1",
        "--depth_mode", "quant", "--depth_q", "0.5", "--depth_levels", "192",
        "--depth_smooth", "40", "--depth_smooth_sig", "0.04",
        "--apply_ckpt", ref, "--apply_out", out, "--apply_ch", "48,96,192,384", "--apply_fp16", "1"]
# cho phép ghi đè cờ (vd --depth_mode ed, --limit 10)
i = 0
while i < len(extra):
    k = extra[i]; v = extra[i+1]
    if k in argv: argv[argv.index(k)+1] = v
    else: argv += [k, v]
    i += 2
sys.argv = ["refiner_data.py"] + argv
t0 = time.time(); refiner_data.main(); print(f"TOTAL_S {time.time()-t0:.2f}")
