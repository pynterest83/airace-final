"""31/08: dump 'tầng 2' cho refiner cascade — copy cấu trúc dump (hardlink warp/mask/depth/alpha/cond),
render.png := ảnh ĐÃ refine (stage-1). GT giữ nguyên → stage-2 học phần dư sau stage-1.
  python build_s2.py --dump SRC_DUMP --stage1 DIR_PNG_REFINED --out DST_DUMP"""
import argparse, os, json, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True); ap.add_argument("--stage1", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--stage1_dump", default="", help="31/08: lấy đầu ra tầng 1 từ <dump>/<name>/render.png thay vì thư mục PNG phẳng")
a = ap.parse_args()
meta = json.load(open(os.path.join(a.dump, "meta.json")))
os.makedirs(a.out, exist_ok=True); n = 0
for m in meta:
    sd = os.path.join(a.dump, m["name"]); od = os.path.join(a.out, m["name"]); os.makedirs(od, exist_ok=True)
    for f in os.listdir(sd):
        if f == "render.png": continue
        dst = os.path.join(od, f)
        if not os.path.exists(dst): os.link(os.path.join(sd, f), dst)
    s1 = os.path.join(a.stage1_dump, m["name"], "render.png") if a.stage1_dump else os.path.join(a.stage1, m["name"] + ".png")
    assert os.path.exists(s1), f"thieu stage1: {s1}"
    shutil.copyfile(s1, os.path.join(od, "render.png")); n += 1
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"S2_DONE n={n} -> {a.out}")
