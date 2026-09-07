"""Package private-set renders into submission ZIP and validate against CSVs."""
import argparse
import csv
import os
import zipfile

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="dir with private scene folders")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--renders_subdir", default="test_renders_redistort")
    ap.add_argument("--out_zip", required=True)
    args = ap.parse_args()

    scenes = sorted([d for d in os.listdir(args.data_root)
                     if os.path.isdir(os.path.join(args.data_root, d))])
    problems = []
    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_STORED) as zf:
        for scene in scenes:
            csv_path = os.path.join(args.data_root, scene, "test", "test_poses.csv")
            rdir = os.path.join(args.results_root, scene, args.renders_subdir)
            if not os.path.isdir(rdir):
                alt = os.path.join(args.results_root, scene, "test_renders_pinhole")
                if os.path.isdir(alt):
                    rdir = alt
                else:
                    problems.append(f"{scene}: renders dir missing")
                    continue
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    name, w, h = row["image_name"], int(row["width"]), int(row["height"])
                    p = os.path.join(rdir, name)
                    if not os.path.exists(p):
                        problems.append(f"{scene}/{name}: MISSING")
                        continue
                    img = cv2.imread(p)
                    if img is None or img.shape[1] != w or img.shape[0] != h:
                        problems.append(f"{scene}/{name}: bad size "
                                        f"{None if img is None else img.shape}")
                        continue
                    zf.write(p, arcname=f"{scene}/{name}")
            print(f"[zip] {scene} packaged from {rdir}")

    if problems:
        print("\n".join(problems))
        raise SystemExit(f"{len(problems)} problems — zip may be invalid!")
    print(f"[ok] {args.out_zip} written, {len(scenes)} scenes, all checks passed")


if __name__ == "__main__":
    main()
