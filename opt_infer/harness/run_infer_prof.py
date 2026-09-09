"""Như run_infer.py nhưng bọc các hàm nặng để đo thời gian + số lần gọi. FT_OVERRIDE=<file> để thử bản vá fuse_test."""
import os, sys, time, importlib.util
sys.path.insert(0, "/home/tensara/projects/vt-track1/contest/src"); import _paths
scene, model, ref, out = sys.argv[1:5]; extra = sys.argv[5:]
os.environ["TORCH_HOME"] = "/home/tensara/projects/vt-track1/contest/work/models"
import cv2, torch
cv2.setRNGSeed(0); torch.manual_seed(0)
ov = os.environ.get("FT_OVERRIDE")
if ov:
    s = importlib.util.spec_from_file_location("fuse_test", ov); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); sys.modules["fuse_test"] = m; print(f"[prof] fuse_test ← {ov}")
rd = os.environ.get("RD_OVERRIDE")
if rd:
    s = importlib.util.spec_from_file_location("refiner_data", rd); m2 = importlib.util.module_from_spec(s)
    s.loader.exec_module(m2); sys.modules["refiner_data"] = m2; print(f"[prof] refiner_data ← {rd}")
import fuse_test as FT, refiner_data
STAT = {}
def wrap(mod, name):
    f = getattr(mod, name)
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.time(); r = f(*a, **k); torch.cuda.synchronize()
        s = STAT.setdefault(name, [0, 0.0]); s[0] += 1; s[1] += time.time() - t; return r
    setattr(mod, name, g)
for n in ("render_depth_quantile", "render_depth", "warp_source", "select_sources", "tile_align"):
    if hasattr(FT, n): wrap(FT, n)
class TS:
    def __init__(s, o): s.o, s.b, s.t0 = o, "", time.time()
    def write(s, x):
        s.b += x
        while "\n" in s.b:
            l, s.b = s.b.split("\n", 1); s.o.write(f"[{time.time()-s.t0:8.2f}] {l}\n")
    def flush(s): s.o.flush()
sys.stdout = TS(sys.stdout)
argv = ["--result_dir", model, "--scene_dir", scene, "--targets", "test", "--dump", out + "_dump",
        "--no_dump", "1", "--pad", "1", "--render_aa", "1", "--K", "4", "--depth_fix", "1",
        "--depth_mode", "quant", "--depth_q", "0.5", "--depth_levels", "192",
        "--depth_smooth", "40", "--depth_smooth_sig", "0.04",
        "--apply_ckpt", ref, "--apply_out", out, "--apply_ch", "48,96,192,384", "--apply_fp16", "1"]
i = 0
while i < len(extra):
    k, v = extra[i], extra[i+1]
    if k in argv: argv[argv.index(k)+1] = v
    else: argv += [k, v]
    i += 2
sys.argv = ["refiner_data.py"] + argv
t0 = time.time(); refiner_data.main(); tot = time.time() - t0
print(f"TOTAL_S {tot:.2f}")
for n, (c, s) in sorted(STAT.items(), key=lambda x: -x[1][1]):
    print(f"PROF {n:24s} gọi {c:5d}  tổng {s:8.2f}s  ({100*s/tot:5.1f}%)  TB {1000*s/max(c,1):7.1f} ms")
