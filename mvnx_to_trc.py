#!/usr/bin/env python3
"""
mvnx_to_trc.py - Convert Xsens MVNX recordings to OpenSim TRC marker files.

Input : an MVNX file exported from Xsens MVN (Analyze/Animate), full-body, MVNX version 4.
Output: a .trc with a 46-marker set (Nose, eyes, ears, shoulders, elbows, wrists,
        fingers, hips, knees, ankles, toes, heels, olecranon, cubital fossa, acromion,
        Neck, PelvisCenter, Thorax, c_spine0-3, c_neck, c_head) in OpenSim coordinates
        (Y-up, millimetres):  X = x, Y = z, Z = -y   (MVNX is Z-up, metres).
        Optionally all Xsens anatomical landmarks + joint centres (--markers xsens).

Every marker is built from the Xsens skeleton of the recording itself, i.e. from the
person's own calibrated body dimensions: joint markers on the Xsens joint centres,
nose/eyes/ears/acromion on the Xsens landmarks, and spine/neck/finger/elbow/foot
markers at fixed anatomical positions scaled to the person's segment lengths.

Every file is checked before it is converted (see check_mvnx): damaged or
incomplete files, files not written by Xsens MVN, missing data and inconsistent
skeletons are rejected; smaller issues are reported as warnings.

Usage
-----
  python mvnx_to_trc.py convert recording.mvnx
  python mvnx_to_trc.py convert path/to/folder -o path/to/output
  python mvnx_to_trc.py convert recording.mvnx --viewer         (also write a 3D check viewer)
  python mvnx_to_trc.py convert recording.mvnx --markers xsens  (all Xsens landmarks)
  python mvnx_to_trc.py check recording.mvnx [more files/folders] (only check, no conversion)

Requires Python 3.8+ and numpy  (pip install numpy).
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

XSENS_NS = "http://www.xsens.com/mvn/mvnx"

# Standard Xsens 23-segment hierarchy
PARENT = {
    "Pelvis": None, "L5": "Pelvis", "L3": "L5", "T12": "L3", "T8": "T12",
    "Neck": "T8", "Head": "Neck",
    "RightShoulder": "T8", "RightUpperArm": "RightShoulder",
    "RightForeArm": "RightUpperArm", "RightHand": "RightForeArm",
    "LeftShoulder": "T8", "LeftUpperArm": "LeftShoulder",
    "LeftForeArm": "LeftUpperArm", "LeftHand": "LeftForeArm",
    "RightUpperLeg": "Pelvis", "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg", "RightToe": "RightFoot",
    "LeftUpperLeg": "Pelvis", "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg", "LeftToe": "LeftFoot",
}

# words in the MVNX comment that show the file was produced by another tool
NON_XSENS_HINTS = ("generated from", "sam3d", "monocular", "pipeline", "synthetic")


class MvnxError(Exception):
    """The file cannot be converted."""


# --------------------------------------------------------------------------- IO
def _tag(e):
    return e.tag.rsplit("}", 1)[-1]


def _ns(e):
    return e.tag[1:].split("}", 1)[0] if e.tag.startswith("{") else ""


def read_mvnx(path):
    """Read an MVNX file.

    Returns a dict with frame_rate, segment labels, segment points (pos_b), positions
    P (N,S,3) and quaternions Q (N,S,4, w-x-y-z) in Z-up metres, frame times, and
    'meta' (information used by check_mvnx). Raises MvnxError if the file cannot be read.
    """
    labels, points = [], {}
    P, Q, times = [], [], []
    meta = {"root": None, "namespace": None, "mvn_version": None, "security_code": False,
            "configuration": None, "frame_rate_attr": None, "segment_count_attr": None,
            "comments": [], "frame_fields": None, "sensor_count": 0,
            "bad_frames": 0, "first_bad_frame": None, "nonfinite_frames": 0}
    stack = []
    body_segments = None   # True while inside the body <segments> list directly under <subject>
    cur_seg = None
    subject_seen = False
    try:
        for event, el in ET.iterparse(path, events=("start", "end")):
            t = _tag(el)
            if event == "start":
                parent = stack[-1] if stack else None
                stack.append(t)
                if meta["root"] is None:
                    meta["root"], meta["namespace"] = t, _ns(el)
                    if t != "mvnx":
                        raise MvnxError(f"not an MVNX file (the file starts with <{t}>, not <mvnx>)")
                elif t == "mvn" and parent == "mvnx":
                    meta["mvn_version"] = el.get("version")
                elif t == "subject" and not subject_seen:
                    subject_seen = True
                    meta["frame_rate_attr"] = el.get("frameRate")
                    meta["configuration"] = el.get("configuration")
                    meta["segment_count_attr"] = el.get("segmentCount")
                elif t == "segments" and parent == "subject" and body_segments is None:
                    body_segments = True   # only the first list: finger-tracking/prop lists are ignored
                elif t == "segment" and body_segments is True and parent == "segments":
                    cur_seg = el.get("label")
                    labels.append(cur_seg)
                    points[cur_seg] = {}
                elif t == "sensor" and parent == "sensors":
                    meta["sensor_count"] += 1
                elif t == "securityCode":
                    meta["security_code"] = True
                continue
            stack.pop()
            if t == "point" and body_segments is True and cur_seg:
                pb = [c for c in el if _tag(c) == "pos_b"]
                if pb and pb[0].text:
                    try:
                        points[cur_seg][el.get("label")] = [float(v) for v in pb[0].text.split()]
                    except ValueError:
                        raise MvnxError(f"invalid number in segment point {cur_seg}:{el.get('label')}")
            elif t == "segments" and body_segments is True:
                body_segments = False
                cur_seg = None
            elif t == "comment" and el.text and el.text.strip():
                meta["comments"].append(el.text.strip())
            elif t == "frame":
                if el.get("type") == "normal" and labels:
                    S = len(labels)
                    d = {_tag(c): c.text for c in el}
                    if meta["frame_fields"] is None:
                        meta["frame_fields"] = list(d)
                    pos, ori = d.get("position"), d.get("orientation")
                    if not pos or not ori:
                        raise MvnxError("the frames contain no segment 'position'/'orientation' - export "
                                        "the MVNX with Segment Position and Segment Orientation enabled")
                    try:
                        pa = np.array(pos.split(), float)
                        qa = np.array(ori.split(), float)
                    except ValueError:
                        raise MvnxError(f"invalid number in frame {len(P) + meta['bad_frames'] + 1}")
                    if pa.size != 3 * S or qa.size != 4 * S:
                        meta["bad_frames"] += 1
                        if meta["first_bad_frame"] is None:
                            meta["first_bad_frame"] = len(P) + meta["bad_frames"]
                    else:
                        if not (np.isfinite(pa).all() and np.isfinite(qa).all()):
                            meta["nonfinite_frames"] += 1
                        P.append(pa.reshape(S, 3))
                        Q.append(qa.reshape(S, 4))
                        try:
                            times.append(float(el.get("time")) / 1000.0)
                        except (TypeError, ValueError):
                            times.append(np.nan)
                el.clear()
    except ET.ParseError as ex:
        if meta["root"] is None:
            if os.path.getsize(path) == 0:
                raise MvnxError("the file is empty")
            raise MvnxError("not an MVNX file (the content is not XML)")
        raise MvnxError(f"the file is damaged or incomplete (XML error: {ex})")
    if meta["root"] is None:
        raise MvnxError("the file is empty")
    if not labels:
        raise MvnxError("no body segments found in the file")
    if not P and meta["bad_frames"] == 0:
        raise MvnxError("the file contains no motion frames")

    times = np.array(times)
    try:
        rate = float(meta["frame_rate_attr"])
    except (TypeError, ValueError):
        rate = 0.0
    if rate <= 0 and len(times) > 1 and np.isfinite(times).all():
        dt = np.median(np.diff(times))
        rate = 1.0 / dt if dt > 0 else 0.0
    S = len(labels)
    return {"frame_rate": rate, "labels": labels, "points": points,
            "P": np.stack(P) if P else np.zeros((0, S, 3)),
            "Q": np.stack(Q) if Q else np.zeros((0, S, 4)),
            "times": times, "comment": " ".join(meta["comments"]), "meta": meta}


def read_trc(path):
    """Read a .trc; returns (marker names, positions (N,K,3) in MVNX Z-up metres)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    rows = [l.split("\t") for l in lines[5:] if l.strip()]
    rows = [r for r in rows if r[0].strip().lstrip("-").replace(".", "").isdigit()]
    data = np.array([[float(v) if v.strip() else np.nan for v in r[: 2 + 3 * len(names)]] for r in rows])
    hdr = lines[1].split("\t"), lines[2].split("\t")
    units = dict(zip(hdr[0], hdr[1])).get("Units", "mm").strip()
    scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}.get(units, 1e-3)
    M = data[:, 2:].reshape(len(data), len(names), 3) * scale
    return names, np.stack([M[..., 0], -M[..., 2], M[..., 1]], -1)   # OpenSim -> MVNX axes


def write_trc(path, names, Xz, rate, times=None):
    """Xz: (N,K,3) MVNX-frame metres -> TRC in OpenSim frame (Y-up), mm."""
    N, K, _ = Xz.shape
    X = np.stack([Xz[..., 0], Xz[..., 2], -Xz[..., 1]], -1) * 1000.0
    out = [
        f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(path)}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate:.6f}\t{rate:.6f}\t{N}\t{K}\tmm\t{rate:.6f}\t1\t{N}",
        "Frame#\tTime\t" + "\t\t\t".join(names),
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, K + 1)),
        "",
    ]
    flat = X.reshape(N, -1)
    exact = np.arange(N) / rate
    if times is None or len(times) != N or np.isnan(times).any():
        times = exact
    else:
        times = times - times[0]
        # MVNX time stamps are whole (sometimes accumulated) milliseconds: use exact frame
        # times unless the recording really has gaps (dropped frames)
        if N < 2 or np.diff(times).max() < 1.5 / rate:
            times = exact
    for i in range(N):
        out.append(f"{i + 1}\t{times[i]:.6f}\t" + "\t".join(f"{v:.6f}" for v in flat[i]))
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


# ======================================================================= anatomical
# Every marker is built from the Xsens skeleton itself. Markers Xsens has no landmark
# for follow the conventions of the 46-marker set (measured on reference .trc files,
# normalised by segment length; consistent between people to ~1-2%).
TRC46 = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
         "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LBigToe", "LSmallToe", "LHeel",
         "RBigToe", "RSmallToe", "RHeel", "RIndex", "RIndex3", "RPinky", "RPinky3", "RWrist",
         "LIndex", "LIndex3", "LPinky", "LPinky3", "LWrist", "LOlecranon", "ROlecranon",
         "LCubitalFossa", "RCubitalFossa", "LAcromion", "RAcromion", "Neck", "PelvisCenter",
         "Thorax", "c_spine0", "c_spine1", "c_spine2", "c_spine3", "c_neck", "c_head"]

ANAT = {
    # c_spine0..3 at these fractions of the spine arc pelvis-centre -> C7/T1 joint
    "spine_frac": [0.093, 0.290, 0.4895, 0.8175],
    # offsets from the C7/T1 joint in the T8 frame, x spine-arc length
    "Neck": [0.014, -0.0015, 0.0485],
    "Thorax": [0.0065, 0.0, 0.0385],
    # hand frame (e1 wrist->knuckles, e2 towards thumb, e3 = e1 x e2), x forearm length
    "R_hand": {"Index": [0.5195, 0.085, -0.171], "Index3": [0.2885, 0.116, 0.0],
               "Pinky": [0.400, -0.169, -0.175], "Pinky3": [0.247, -0.116, 0.0]},
    "L_hand": {"Index": [0.5465, 0.077, 0.1305], "Index3": [0.2885, 0.116, 0.0],
               "Pinky": [0.417, -0.1715, 0.160], "Pinky3": [0.247, -0.116, 0.0]},
    # foot frame (e1 forward, e3 up from the sole, e2 = e3 x e1) from the ankle, x shank length
    "R_foot": {"Heel": [-0.080, -0.042, -0.116], "BigToe": [0.470, 0.046, -0.116],
               "SmallToe": [0.3805, -0.1295, -0.116]},
    "L_foot": {"Heel": [-0.0775, 0.0365, -0.1175], "BigToe": [0.4715, -0.052, -0.1175],
               "SmallToe": [0.3825, 0.125, -0.1175]},
    # olecranon / cubital fossa distance from the elbow joint, x forearm length
    "olecranon": 0.1365, "cubital": 0.1310,
}


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _frame(e1, e2):
    e1 = _unit(e1)
    e2 = _unit(e2 - (e2 * e1).sum(-1, keepdims=True) * e1)
    return np.stack([e1, e2, np.cross(e1, e2)], -1)  # columns = axes


def global_points(mv):
    """Global position of every MVNX segment point: {segment: {label: (N,3)}}."""
    idx = {s: i for i, s in enumerate(mv["labels"])}
    R = quat_to_mat(mv["Q"])
    G = {}
    for seg, pts in mv["points"].items():
        i = idx[seg]
        G[seg] = {k: mv["P"][:, i] + R[:, i] @ np.array(v) for k, v in pts.items()}
    return G, R, idx


REQUIRED_POINTS = {
    "Pelvis": ["pHipOrigin", "jL5S1"], "L5": ["jL4L3"], "L3": ["jL1T12"], "T12": ["jT9T8"],
    "T8": ["jT1C7"], "Neck": ["jT1C7", "jC1Head"],
    "Head": ["jC1Head", "pNose", "pRightEye", "pLeftEye", "pRightAuricularis", "pLeftAuricularis"],
}
for _side in ("Right", "Left"):
    REQUIRED_POINTS.update({
        f"{_side}Shoulder": [f"p{_side}Acromion"], f"{_side}UpperArm": [f"j{_side}Shoulder"],
        f"{_side}ForeArm": [f"j{_side}Elbow", f"j{_side}Wrist", f"p{_side}Olecranon"],
        f"{_side}Hand": [f"j{_side}Wrist", f"p{_side}TopOfHand", f"p{_side}Pinky"],
        f"{_side}UpperLeg": [f"j{_side}Hip"], f"{_side}LowerLeg": [f"j{_side}Knee", f"j{_side}Ankle"],
        f"{_side}Foot": [f"j{_side}Ankle"]})


def anatomical_markers(mv, markerset="trc46"):
    G, R, idx = global_points(mv)
    pts = mv["points"]
    if markerset == "xsens":
        names, cols = [], []
        for seg in mv["labels"]:
            for k in pts[seg]:
                if k not in names:
                    names.append(k)
                    cols.append(G[seg][k])
        return names, np.stack(cols, 1)

    M = {}
    pel = G["Pelvis"]["pHipOrigin"]
    M["PelvisCenter"] = pel
    for S, side in (("R", "Right"), ("L", "Left")):
        M[S + "Hip"] = G[side + "UpperLeg"]["j%sHip" % side]
        M[S + "Knee"] = G[side + "LowerLeg"]["j%sKnee" % side]
        M[S + "Ankle"] = G[side + "Foot"]["j%sAnkle" % side]
        M[S + "Shoulder"] = G[side + "UpperArm"]["j%sShoulder" % side]
        M[S + "Elbow"] = G[side + "ForeArm"]["j%sElbow" % side]
        M[S + "Wrist"] = G[side + "Hand"]["j%sWrist" % side]
        M[S + "Acromion"] = G[side + "Shoulder"]["p%sAcromion" % side]
        M[S + "Ear"] = G["Head"]["p%sAuricularis" % side]
        M[S + "Eye"] = G["Head"]["p%sEye" % side]
        # elbow: olecranon behind the joint (direction of Xsens' olecranon), cubital fossa in front
        fa = np.linalg.norm(pts[side + "ForeArm"]["j%sWrist" % side])
        back = _unit(G[side + "ForeArm"]["p%sOlecranon" % side] - M[S + "Elbow"])
        M[S + "Olecranon"] = M[S + "Elbow"] + ANAT["olecranon"] * fa * back
        M[S + "CubitalFossa"] = M[S + "Elbow"] - ANAT["cubital"] * fa * back
        # hand
        h = G[side + "Hand"]
        Fh = _frame(h["p%sTopOfHand" % side] - M[S + "Wrist"],    # e1: along the hand
                    M[S + "Wrist"] - h["p%sPinky" % side])        # e2: towards the thumb side
        for k, c in ANAT[S + "_hand"].items():
            M[S + k] = M[S + "Wrist"] + Fh @ (np.array(c) * fa)
        # foot: e1 forward, e3 up (Xsens foot x and z axes)
        sh = np.linalg.norm(pts[side + "LowerLeg"]["j%sAnkle" % side])
        Rf = R[:, idx[side + "Foot"]]
        Ff = np.stack([Rf[..., 0], np.cross(Rf[..., 2], Rf[..., 0]), Rf[..., 2]], -1)
        for k, c in ANAT[S + "_foot"].items():
            M[S + k] = M[S + "Ankle"] + Ff @ (np.array(c) * sh)
    M["Nose"] = G["Head"]["pNose"]
    # spine: polyline pelvis centre -> L5S1 -> L4L3 -> L1T12 -> T9T8 -> T1C7
    poly = [pel, G["L5"]["jL5S1"], G["L3"]["jL4L3"], G["T12"]["jL1T12"], G["T8"]["jT9T8"],
            G["Neck"]["jT1C7"]]
    seglen = [np.linalg.norm(pts[s][j]) for s, j in
              (("Pelvis", "jL5S1"), ("L5", "jL4L3"), ("L3", "jL1T12"), ("T12", "jT9T8"), ("T8", "jT1C7"))]
    cum = np.concatenate([[0], np.cumsum(seglen)])
    arc = cum[-1]
    for i, f in enumerate(ANAT["spine_frac"]):
        d = f * arc
        k = min(int(np.searchsorted(cum, d, side="right")) - 1, 4)
        t = (d - cum[k]) / seglen[k]
        M["c_spine%d" % i] = poly[k] + t * (poly[k + 1] - poly[k])
    M["c_neck"] = G["Neck"]["jT1C7"]
    M["c_head"] = G["Head"]["jC1Head"]
    RT8 = R[:, idx["T8"]]
    for k in ("Neck", "Thorax"):
        M[k] = M["c_neck"] + RT8 @ (np.array(ANAT[k]) * arc)
    return TRC46, np.stack([M[n] for n in TRC46], 1)


# ============================================================================ checks
def body_height(points):
    """Approximate standing height (m) from the Xsens segment dimensions."""
    def z(seg, lab):
        return abs(points.get(seg, {}).get(lab, [0, 0, 0])[2])
    trunk = (z("Pelvis", "jL5S1") + z("L5", "jL4L3") + z("L3", "jL1T12") + z("T12", "jT9T8")
             + z("T8", "jT1C7") + z("Neck", "jC1Head") + z("Head", "pTopOfHead"))
    leg = np.mean([z(f"{s}UpperLeg", f"j{s}Knee") + z(f"{s}LowerLeg", f"j{s}Ankle")
                   + z(f"{s}Foot", f"p{s}HeelFoot") for s in ("Right", "Left")])
    return trunk + leg


def joint_gaps(mv):
    """For every joint: distance (mm) between the joint computed from the parent segment
    and the child segment's origin. Xsens output is consistent to ~0.1 mm (toes ~20 mm)."""
    R = quat_to_mat(mv["Q"])
    idx = {s: i for i, s in enumerate(mv["labels"])}
    out = {}
    for s, par in PARENT.items():
        if not par:
            continue
        o = [k for k, v in mv["points"][s].items() if k.startswith("j") and np.linalg.norm(v) < 1e-9]
        if not o or o[0] not in mv["points"][par]:
            continue
        pj = mv["P"][:, idx[par]] + R[:, idx[par]] @ np.array(mv["points"][par][o[0]])
        out[s] = np.linalg.norm(pj - mv["P"][:, idx[s]], axis=1) * 1000
    return out


def check_mvnx(mv, markerset="trc46"):
    """Check an MVNX before conversion.

    Returns (errors, origin_errors, warnings, info):
      errors        -> the file cannot be converted
      origin_errors -> the file does not look like it was written by Xsens MVN
                       (conversion only with --force)
      warnings      -> converted, but worth knowing
    """
    meta = mv["meta"]
    err, origin, warn = [], [], []
    N = len(mv["P"])
    rate = mv["frame_rate"]

    # --- is it from Xsens MVN?
    if meta["namespace"] != XSENS_NS:
        origin.append("it does not use the Xsens MVNX namespace (" + XSENS_NS + "), "
                      "so it was not written by Xsens MVN")
    hint = [c for c in meta["comments"] if any(h in c.lower() for h in NON_XSENS_HINTS)]
    if hint:
        origin.append(f'its comment says "{hint[0][:80]}"')
    if not meta["security_code"]:
        warn.append("no Xsens securityCode element - the file may have been edited or written by another program")

    # --- structure and data
    missing = [s for s in PARENT if s not in mv["labels"]]
    if missing:
        err.append(f"missing body segments {missing} - a full-body (23-segment) recording is required")
    if meta["bad_frames"]:
        err.append(f"{meta['bad_frames']} frame(s) have the wrong amount of data for {len(mv['labels'])} "
                   f"segments (first: frame {meta['first_bad_frame']}) - the file is damaged")
    if meta["nonfinite_frames"]:
        err.append(f"{meta['nonfinite_frames']} frame(s) contain NaN/infinite values")
    if N == 0:
        err.append("no valid motion frames")
    if not rate or rate <= 0 or rate > 2000:
        err.append(f"invalid frame rate ({meta['frame_rate_attr']})")
    if err or missing:
        return err, origin, warn, {}

    if markerset == "trc46":
        lack = [f"{s}:{p}" for s, ps in REQUIRED_POINTS.items() for p in ps if p not in mv["points"].get(s, {})]
        if lack:
            err.append("missing anatomical landmark points " + ", ".join(lack[:6]) + (" ..." if len(lack) > 6 else "")
                       + " - export from MVN 2019 or newer (MVNX version 4)")
    qn = np.abs(np.linalg.norm(mv["Q"], axis=2) - 1)
    if np.isfinite(qn).all() and qn.max() > 1e-3:
        err.append(f"{int((qn.max(axis=1) > 1e-3).sum())} frame(s) contain invalid orientations "
                   f"(quaternion length off by up to {qn.max():.3f})")

    # --- skeleton consistency
    gaps = joint_gaps(mv)
    if gaps:
        p99 = {s: float(np.percentile(g, 99)) for s, g in gaps.items()}
        med = float(np.median(list(p99.values())))
        worst = max(p99, key=p99.get)
        if med > 20 or p99[worst] > 150:
            err.append(f"the skeleton is inconsistent (joints up to {p99[worst]:.0f} mm apart at {worst}) - "
                       "data damaged or not from Xsens")
        elif med > 1 or p99[worst] > 40:
            warn.append(f"the skeleton is less consistent than normal Xsens output (median joint gap "
                        f"{med:.1f} mm, {worst} {p99[worst]:.0f} mm)")

    # --- plausibility
    h = body_height(mv["points"])
    if not 1.2 <= h <= 2.3:
        warn.append(f"unusual body height in the Xsens calibration (~{h:.2f} m)")
    if meta["configuration"] and meta["configuration"] != "FullBody":
        warn.append(f'recording configuration is "{meta["configuration"]}" (not FullBody) - '
                    "some segments may not be tracked")
    t = mv["times"]
    if N > 1 and np.isfinite(t).all():
        dt = np.diff(t)
        gaps_n = int((dt > 1.5 / rate).sum())
        if (dt <= 0).any():
            warn.append(f"{int((dt <= 0).sum())} frame(s) have non-increasing time stamps")
        if gaps_n:
            lost = int(np.round((dt[dt > 1.5 / rate] * rate - 1).sum()))
            warn.append(f"{gaps_n} gap(s) in the recording (~{lost} dropped frames); the TRC keeps the real times")
    if N > 1:
        dt = np.diff(t) if np.isfinite(t).all() and (np.diff(t) > 0).all() else np.full(N - 1, 1 / rate)
        v = np.linalg.norm(np.diff(mv["P"][:, 0], axis=0), axis=1) / dt
        jumps = int((v > 10).sum())
        if jumps:
            warn.append(f"{jumps} frame(s) where the pelvis moves faster than 10 m/s (possible glitch)")
    if N < rate:
        warn.append(f"very short recording ({N} frames)")

    info = {"frames": N, "rate": rate, "duration": N / rate, "height": h,
            "mvn": meta["mvn_version"], "sensors": meta["sensor_count"]}
    return err, origin, warn, info


def describe(info):
    return (f"Xsens MVN {info['mvn'] or '?'}, {info['rate']:g} Hz, {info['frames']} frames "
            f"({info['duration']:.1f} s), body height ~{info['height']:.2f} m")


# ------------------------------------------------------------------------- convert
def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".mvnx"))
        elif p.lower().endswith(".mvnx") and os.path.isfile(p):
            files.append(p)
        else:
            print(f"Skipping (not an .mvnx file or folder): {p}")
    return files


def load_checked(f, markerset="trc46", force=False, verbose=True):
    """Read and check one file. Returns mv or None (problems are printed)."""
    name = os.path.basename(f)
    try:
        mv = read_mvnx(f)
    except MvnxError as ex:
        print(f"[fail] {name}: {ex}")
        return None
    err, origin, warn, info = check_mvnx(mv, markerset)
    if err:
        for e in err:
            print(f"[fail] {name}: {e}")
        return None
    if origin and not force:
        print(f"[fail] {name}: not an Xsens recording - " + "; ".join(origin) + " (use --force to convert anyway)")
        return None
    if origin:
        print(f"[warn] {name}: not an Xsens recording - " + "; ".join(origin) + " (converting because of --force)")
    for w in warn:
        print(f"[warn] {name}: {w}")
    if verbose and info:
        print(f"[ok]   {name}: {describe(info)}")
    return mv


def check(inputs):
    files = collect_inputs(inputs)
    if not files:
        sys.exit("No .mvnx files found.")
    ok = sum(load_checked(f) is not None for f in files)
    print(f"\n{ok} of {len(files)} file(s) can be converted.")


def convert(inputs, out_dir=None, overwrite=False, rate=None, markerset="trc46",
            viewer=False, viewer_fps=30, force=False):
    files = collect_inputs(inputs)
    if not files:
        sys.exit("No .mvnx files found.")
    done = 0
    for f in files:
        dst_dir = out_dir or os.path.dirname(os.path.abspath(f))
        suffix = "_xsens_landmarks" if markerset == "xsens" else ""
        dst = os.path.join(dst_dir, os.path.splitext(os.path.basename(f))[0] + suffix + ".trc")
        if os.path.exists(dst) and not overwrite:
            print(f"[skip] {os.path.basename(dst)} already exists (use --overwrite)")
            continue
        print(f"[....] reading {os.path.basename(f)} ...", flush=True)
        mv = load_checked(f, markerset, force, verbose=False)
        if mv is None:
            continue
        try:
            os.makedirs(dst_dir, exist_ok=True)
            names, X = anatomical_markers(mv, markerset)
            write_trc(dst, names, X, rate or mv["frame_rate"], None if rate else mv.get("times"))
            print(f"[ok]   {os.path.basename(f)} -> {dst}  ({len(X)} frames, {len(names)} markers)")
            done += 1
        except Exception as ex:  # keep a batch going
            print(f"[fail] {os.path.basename(f)}: {ex}")
            continue
        if viewer:
            try:
                from mvnx_trc_viewer import write_viewer
                vpath, vd = write_viewer(f, dst, fps=viewer_fps, mv=mv, quiet=True)
                ex = (f", exact pairs mean {vd['exactMean']:.2f} mm / max {vd['exactMax']:.2f} mm"
                      if vd.get("exactMean") is not None else "")
                print(f"       viewer -> {vpath}{ex}")
            except Exception as ex:
                print(f"       viewer failed: {ex}")
    print(f"\nConverted {done} of {len(files)} file(s).")


# ---------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Convert Xsens MVNX recordings to OpenSim TRC marker files.")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("convert", help="convert .mvnx file(s) or folder(s) to .trc")
    c.add_argument("inputs", nargs="+", help=".mvnx files and/or folders")
    c.add_argument("-o", "--out", help="output folder (default: next to each input)")
    c.add_argument("--overwrite", action="store_true", help="overwrite existing .trc files")
    c.add_argument("--markers", choices=["trc46", "xsens"], default="trc46",
                   help="trc46 (default): the 46-marker set; xsens: every Xsens anatomical "
                        "landmark and joint centre")
    c.add_argument("--viewer", action="store_true",
                   help="also write a 3D check viewer (<name>_viewer.html) next to each .trc")
    c.add_argument("--viewer-fps", type=float, default=30,
                   help="frame rate of the viewer (default 30, 0 = every frame)")
    c.add_argument("--rate", type=float, help="override the frame rate (Hz)")
    c.add_argument("--force", action="store_true",
                   help="convert files that do not look like Xsens recordings (data checks still apply)")

    k = sub.add_parser("check", help="check .mvnx file(s) without converting")
    k.add_argument("inputs", nargs="+", help=".mvnx files and/or folders")

    argv = sys.argv[1:]
    if argv and argv[0] not in ("convert", "check", "-h", "--help"):
        argv = ["convert"] + argv  # allow drag-and-drop / plain paths
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        convert(a.inputs, a.out, a.overwrite, a.rate, a.markers, a.viewer, a.viewer_fps, a.force)
    elif a.cmd == "check":
        check(a.inputs)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
