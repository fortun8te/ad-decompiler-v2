import sys, os, glob
import numpy as np
from PIL import Image
import cv2

def analyze(path):
    im = Image.open(path)
    if im.mode != 'RGBA':
        return None
    arr = np.asarray(im)
    a = arr[..., 3].astype(np.float32) / 255.0
    rgb = arr[..., :3].astype(np.float32)
    h, w = a.shape
    total = a.size
    # alpha histogram
    opaque = (a >= 0.98).sum()
    transp = (a <= 0.02).sum()
    partial = total - opaque - transp
    # is it binary?
    binary = partial / total < 0.01
    # boundary band: dilate/erode the >0.5 mask
    m = (a > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return dict(path=os.path.basename(path), empty=True)
    k = np.ones((3,3), np.uint8)
    er = cv2.erode(m, k, iterations=2)
    di = cv2.dilate(m, k, iterations=2)
    band = (di > 0) & (er == 0)  # 2px each side of edge
    # edge jaggedness: perimeter vs area (compactness). Higher = more jagged relative to size
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cnts)
    area = m.sum()
    # step-index roughness: count of direction changes along contour normalized
    # Fringe: in the just-inside band (opaque-ish edge pixels), how many are near-white or near-bg
    inside_edge = (m > 0) & (er == 0)  # 2px inside ring
    ie = rgb[inside_edge]
    if len(ie):
        # fraction near-white (all channels >235)
        white_frac = float((ie.min(axis=1) > 235).mean())
        # brightness of edge ring vs interior
        interior = rgb[er > 0]
        edge_lum = ie.mean()
        int_lum = interior.mean() if len(interior) else float('nan')
    else:
        white_frac = float('nan'); edge_lum=int_lum=float('nan')
    # "doubled contour" heuristic: bright ring around a darker interior
    return dict(
        path=os.path.basename(path), size=f"{w}x{h}",
        binary=binary,
        opaque_pct=round(100*opaque/total,1),
        partial_pct=round(100*partial/total,2),
        band_px=int(band.sum()),
        compactness=round(perim*perim/(4*np.pi*area),2) if area else None,
        edge_white_frac=round(white_frac,3) if white_frac==white_frac else None,
        edge_lum=round(edge_lum,1) if edge_lum==edge_lum else None,
        interior_lum=round(int_lum,1) if int_lum==int_lum else None,
        halo_delta=round(edge_lum-int_lum,1) if (edge_lum==edge_lum and int_lum==int_lum) else None,
    )

paths = sys.argv[1:]
for p in paths:
    try:
        r = analyze(p)
        if r: print(r)
    except Exception as e:
        print(os.path.basename(p), "ERR", e)
