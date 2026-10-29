#!/usr/bin/env python3
"""
mvnx_trc_viewer.py - 3D viewer to check that the markers in a .trc match the MVNX skeleton.

Writes a self-contained HTML file (works offline) and opens it in your browser.

  python mvnx_trc_viewer.py recording.mvnx                 (finds converted/recording.trc)
  python mvnx_trc_viewer.py recording.mvnx recording.trc
  python mvnx_trc_viewer.py recording.mvnx recording.trc --fps 0 -o check.html   (all frames)

  or directly while converting:  python mvnx_to_trc.py convert recording.mvnx --viewer

What it shows
  * the MVNX skeleton: joint centres, bones and (optionally) all anatomical landmarks
  * the TRC markers
  * a line from every TRC marker to its MVNX counterpart, coloured by distance
      - "exact" pairs: markers that should sit exactly on an MVNX joint/landmark
        (hips, knees, ankles, shoulders, elbows, wrists, pelvis, neck/head joints,
        nose, eyes, ears, acromions) -> distance should be ~0 mm
      - "nearest": markers without an MVNX equivalent (fingers, toes, spine, ...)
        are paired with the closest MVNX point -> a constant offset is expected
  * per-marker distance now / mean / max over the whole recording, and a distance
    curve over time for the selected marker

The TRC is read in its own coordinates (OpenSim, Y-up, mm) and the MVNX in its own
(Z-up, m); the only link between them is the standard axis convention
X = x, Y = z, Z = -y.
"""
import argparse
import base64
import json
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mvnx_to_trc import read_mvnx, read_trc, global_points, MvnxError  # noqa: E402

TRC_STICKS = [("PelvisCenter", "c_spine0"), ("c_spine0", "c_spine1"), ("c_spine1", "c_spine2"),
              ("c_spine2", "c_spine3"), ("c_spine3", "c_neck"), ("c_neck", "c_head"), ("c_neck", "Neck"),
              ("LHip", "RHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
              ("LAnkle", "LHeel"), ("LHeel", "LBigToe"), ("LBigToe", "LSmallToe"), ("LSmallToe", "LHeel"),
              ("RAnkle", "RHeel"), ("RHeel", "RBigToe"), ("RBigToe", "RSmallToe"), ("RSmallToe", "RHeel"),
              ("LShoulder", "RShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
              ("RShoulder", "RElbow"), ("RElbow", "RWrist"), ("LWrist", "LIndex3"), ("LIndex3", "LIndex"),
              ("LWrist", "LPinky3"), ("LPinky3", "LPinky"), ("RWrist", "RIndex3"), ("RIndex3", "RIndex"),
              ("RWrist", "RPinky3"), ("RPinky3", "RPinky"), ("c_head", "Nose"), ("LEye", "REye")]


def exact_pairs():
    m = {"PelvisCenter": ("Pelvis", "pHipOrigin"), "Nose": ("Head", "pNose"),
         "c_neck": ("Neck", "jT1C7"), "c_head": ("Head", "jC1Head")}
    for S, side in (("L", "Left"), ("R", "Right")):
        m[S + "Hip"] = (side + "UpperLeg", f"j{side}Hip")
        m[S + "Knee"] = (side + "LowerLeg", f"j{side}Knee")
        m[S + "Ankle"] = (side + "Foot", f"j{side}Ankle")
        m[S + "Shoulder"] = (side + "UpperArm", f"j{side}Shoulder")
        m[S + "Elbow"] = (side + "ForeArm", f"j{side}Elbow")
        m[S + "Wrist"] = (side + "Hand", f"j{side}Wrist")
        m[S + "Acromion"] = (side + "Shoulder", f"p{side}Acromion")
        m[S + "Ear"] = ("Head", f"p{side}Auricularis")
        m[S + "Eye"] = ("Head", f"p{side}Eye")
    return m


MAX_VIEWER_MB = 60  # keeps the HTML small enough for any browser


def build(mvnx_path, trc_path, fps_cap, mv=None, quiet=False):
    say = (lambda *a, **k: None) if quiet else print
    if mv is None:
        say(f"Reading {os.path.basename(mvnx_path)} ...", flush=True)
        mv = read_mvnx(mvnx_path)
    G, _, _ = global_points(mv)
    pts = mv["points"]

    # unique MVNX points; a joint shared by two segments is taken from the segment it starts
    xs_names, xs_seg, xs_xyz = [], [], []
    for seg in mv["labels"]:
        for lab, pb in pts[seg].items():
            if lab in xs_names:
                if np.linalg.norm(pb) < 1e-9:   # origin of this segment -> prefer it
                    i = xs_names.index(lab)
                    xs_seg[i], xs_xyz[i] = seg, G[seg][lab]
                continue
            xs_names.append(lab)
            xs_seg.append(seg)
            xs_xyz.append(G[seg][lab])
    X = np.stack(xs_xyz, 1)                       # (N, Kx, 3) Z-up metres
    is_joint = [n.startswith("j") or n == "pHipOrigin" for n in xs_names]

    bones = []
    for seg in mv["labels"]:
        labs = list(pts[seg])
        origin = [l for l in labs if np.linalg.norm(pts[seg][l]) < 1e-9]
        if not origin:
            continue
        o = origin[0]
        ends = [l for l in labs if l != o and l.startswith("j")]
        if seg == "Head":
            ends.append("pTopOfHead")
        for e in ends:
            if e in xs_names:
                bones.append([xs_names.index(o), xs_names.index(e)])

    say(f"Reading {os.path.basename(trc_path)} ...", flush=True)
    trc_names, T = read_trc(trc_path)              # converted to Z-up metres
    trc_times = None
    try:
        trc_times = np.array([float(l.split("\t")[1]) for l in open(trc_path, encoding="utf-8", errors="replace").read().splitlines()[5:]
                              if l.strip() and l.split("\t")[0].strip().isdigit()])
    except Exception:
        pass

    # time alignment
    N = len(X)
    mv_t = mv.get("times")
    if mv_t is not None and not np.isnan(mv_t).any():
        mv_t = mv_t - mv_t[0]
    else:
        mv_t = np.arange(N) / mv["frame_rate"]
    if len(T) == N:
        Ti = T
    elif trc_times is not None and len(trc_times) == len(T):
        j = np.clip(np.searchsorted(trc_times, mv_t), 0, len(T) - 1)
        Ti = T[j]
        say(f"  note: {len(T)} TRC frames vs {N} MVNX frames - matched by time")
    else:
        n = min(len(T), N)
        X, Ti, mv_t, N = X[:n], T[:n], mv_t[:n], n
        say(f"  note: frame counts differ - using first {n} frames")

    # pairs
    ex = exact_pairs()
    sub = slice(None, None, max(1, N // 500))
    pairs, ptype = [], []
    for k, nm in enumerate(trc_names):
        if nm in ex and ex[nm][1] in xs_names:
            pairs.append(xs_names.index(ex[nm][1]))
            ptype.append("exact")
        else:
            d = np.linalg.norm(X[sub] - Ti[sub, k][:, None, :], axis=2).mean(0)
            pairs.append(int(np.argmin(d)))
            ptype.append("nearest")
    dist = np.linalg.norm(Ti - X[:, pairs], axis=2) * 1000   # (N, Kt) mm
    stats = [{"mean": float(np.nanmean(dist[:, k])), "p95": float(np.nanpercentile(dist[:, k], 95)),
              "max": float(np.nanmax(dist[:, k])), "argmax": int(np.nanargmax(dist[:, k]))}
             for k in range(len(trc_names))]

    # subsample for display
    step = max(1, int(round(mv["frame_rate"] / fps_cap))) if fps_cap else 1
    per_frame = (X.shape[1] + Ti.shape[1]) * 3 * 2 * 4 / 3          # bytes after base64
    while N / step * per_frame > MAX_VIEWER_MB * 1e6:
        step += 1
    if fps_cap and step > round(mv["frame_rate"] / fps_cap) or (not fps_cap and step > 1):
        say(f"  note: long recording - viewer shows every {step}th frame to stay under {MAX_VIEWER_MB} MB")
    idx = np.arange(0, N, step)
    allp = np.concatenate([X[idx], Ti[idx]], 1) * 1000
    allp = np.nan_to_num(allp, nan=-32768)
    arr = np.clip(np.round(allp), -32767, 32767).astype("<i2")
    lowest = np.nanmin(X[..., 2], axis=1)
    floor = float(np.nanpercentile(lowest, 2) * 1000)

    sticks = [[trc_names.index(a), trc_names.index(b)] for a, b in TRC_STICKS
              if a in trc_names and b in trc_names]
    data = {
        "mvnx": os.path.basename(mvnx_path), "trc": os.path.basename(trc_path),
        "rate": mv["frame_rate"] / step, "srcRate": mv["frame_rate"], "step": step,
        "n": int(len(idx)), "nFull": int(N), "times": (mv_t[idx]).round(4).tolist(),
        "xsNames": xs_names, "xsSeg": xs_seg, "xsJoint": is_joint, "bones": bones,
        "trcNames": trc_names, "pairs": pairs, "ptype": ptype, "stats": stats,
        "sticks": sticks, "floor": floor,
        "pelvis": xs_names.index("pHipOrigin") if "pHipOrigin" in xs_names else 0,
        "lhip": xs_names.index("jLeftHip") if "jLeftHip" in xs_names else -1,
        "rhip": xs_names.index("jRightHip") if "jRightHip" in xs_names else -1,
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }
    ex_d = dist[:, [k for k, t in enumerate(ptype) if t == "exact"]]
    say(f"  {len(xs_names)} MVNX points, {len(trc_names)} TRC markers, {N} frames "
        f"(viewer at {data['rate']:.0f} fps)")
    data["exactMean"] = float(np.nanmean(ex_d)) if ex_d.size else None
    data["exactMax"] = float(np.nanmax(ex_d)) if ex_d.size else None
    if ex_d.size:
        say(f"  exact pairs: mean {np.nanmean(ex_d):.2f} mm, max {np.nanmax(ex_d):.2f} mm")
    return data


def write_viewer(mvnx_path, trc_path, out=None, fps=30, mv=None, quiet=False):
    """Build the viewer HTML; returns (path, data)."""
    data = build(mvnx_path, trc_path, fps, mv=mv, quiet=quiet)
    out = out or os.path.splitext(trc_path)[0] + "_viewer.html"
    with open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8") as f:
        html = f.read()
    title = os.path.splitext(os.path.basename(trc_path))[0]
    html = html.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out, data


def main():
    ap = argparse.ArgumentParser(description="3D check of TRC markers against the MVNX skeleton")
    ap.add_argument("mvnx")
    ap.add_argument("trc", nargs="?", help="default: converted\\<name>.trc, then <name>.trc next to the mvnx")
    ap.add_argument("-o", "--out", help="output .html (default: next to the trc)")
    ap.add_argument("--fps", type=float, default=30, help="display frame rate cap (default 30; 0 = all frames)")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    mvnx = a.mvnx
    trc = a.trc
    if trc is None:
        base = os.path.splitext(os.path.basename(mvnx))[0] + ".trc"
        d = os.path.dirname(os.path.abspath(mvnx))
        for cand in (os.path.join(d, "converted", base), os.path.join(d, base)):
            if os.path.exists(cand):
                trc = cand
                break
        if trc is None:
            sys.exit(f"No .trc found for {mvnx}; give it as second argument.")
    try:
        out, _ = write_viewer(mvnx, trc, a.out, a.fps)
    except MvnxError as ex:
        sys.exit(f"Cannot read {os.path.basename(mvnx)}: {ex}")
    print(f"Viewer written -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    if not a.no_open:
        webbrowser.open("file:///" + os.path.abspath(out).replace("\\", "/"))


if __name__ == "__main__":
    main()
