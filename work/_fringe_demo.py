"""Demonstrate decontamination removing the 002 'doubled white contour'.

Builds a crisp-edged dark product composited onto a busy background with a 1-2px
antialiased edge (exactly how a photo edge looks). A BINARY cut keeps the mixed
edge pixels -> visible light fringe (doubled contour). refine() should remove it.
"""
import os, sys
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import matting

REVIEW = os.path.join(os.path.dirname(__file__), "..", "runs", "_matting_review")
os.makedirs(REVIEW, exist_ok=True)

def build(bg_color, prod_color, n=300):
    # busy-ish background around a bright/dark product
    rng = np.random.default_rng(3)
    bg = np.clip(np.array(bg_color, np.float32) + rng.integers(-12, 12, (n, n, 3)), 0, 255)
    # high-res product silhouette, then downsample -> real antialiased edge
    s = 4
    big = np.zeros((n*s, n*s), np.uint8)
    cv2.rectangle(big, (70*s, 40*s), (150*s, 250*s), 255, -1)
    cv2.circle(big, (110*s, 40*s), 40*s, 255, -1)  # rounded top
    aa = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    comp = bg*(1-aa[...,None]) + np.array(prod_color,np.float32)*aa[...,None]
    comp = comp.astype(np.uint8)
    binmask = (aa > 0.5).astype(np.uint8)*255   # what SAM gives: hard threshold
    return comp, binmask

def strip(name, rgb, binmask):
    before = np.dstack([rgb, binmask])
    out = matting.refine(rgb, binmask, element_role="product")
    # metrics: fringe = mean brightness delta of the 1px inside-edge ring
    def ring_lum(rgba):
        a = rgba[...,3]; m=(a>128).astype(np.uint8)
        er=cv2.erode(m,np.ones((3,3),np.uint8),iterations=1)
        ring=(m>0)&(er==0); interior=er>0
        rgbf=rgba[...,:3].astype(np.float32)
        return rgbf[ring].mean(), rgbf[interior].mean()
    be,bi = ring_lum(before); ae,ai = ring_lum(out.rgba)
    print(f"{name}: BEFORE edge={be:.0f} interior={bi:.0f} (fringe {be-bi:+.0f})  "
          f"AFTER edge={ae:.0f} interior={ai:.0f} (fringe {ae-ai:+.0f})  "
          f"partial={out.metrics['alpha_partial_frac']:.3f}")
    def checker(h,w,sq=16):
        yy,xx=np.mgrid[0:h,0:w]; c=(((yy//sq)+(xx//sq))%2).astype(np.uint8)
        return (np.stack([c]*3,-1)*40+190).astype(np.uint8)
    def over(rgba):
        a=rgba[...,3:4].astype(np.float32)/255; bg=checker(*rgba.shape[:2]).astype(np.float32)
        return (rgba[...,:3]*a+bg*(1-a)).astype(np.uint8)
    panels=[rgb, over(before), over(out.rgba), np.stack([out.alpha]*3,-1).__mul__(255).astype(np.uint8)]
    Image.fromarray(np.concatenate(panels,1)).save(os.path.join(REVIEW,f"{name}.png"))

# dark product on white (H15 black sachet), and bright product on mid bg
strip("demo_dark_on_white", *build((245,245,245),(22,20,26)))
strip("demo_bright_on_dark", *build((18,18,22),(232,228,235)))
print("strips ->", os.path.abspath(REVIEW))
