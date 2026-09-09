import sys, os, time
sys.path.insert(0, "/home/tensara/projects/vt-track1/contest/src"); import _paths
import cv2, numpy as np, importlib.util
cv2.setRNGSeed(0); np.random.seed(0)
mod_path, out, *extra = sys.argv[1:]
s = importlib.util.spec_from_file_location("tr", mod_path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
SP="/tmp/claude-1001/-home-tensara-projects-vt-track1/b5f86564-1423-4241-9b5e-2c02b91f3119/scratchpad/opt"
sys.argv = ["tr.py","--orig_scene",f"{SP}/scene_f4_kp","--new_scene","/home/tensara/projects/vt-track1/data/phase2/phase2_f4_reba5_tr6_kp","--out_scene",out,"--pts","kp"] + extra
t0=time.time(); m.main(); print(f"TIME_S {time.time()-t0:.1f}")
