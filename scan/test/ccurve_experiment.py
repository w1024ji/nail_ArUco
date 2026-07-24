"""Experiment: segment fingertip + nail arc in end-on c-curve photo (blue box)."""
import cv2
import numpy as np

img = cv2.imread('sh_images/sh_middle_ccurve.jpg')
H, W = img.shape[:2]
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
L = lab[:, :, 0].astype(np.float32)
A = lab[:, :, 1].astype(np.float32) - 128
B = lab[:, :, 2].astype(np.float32) - 128

# ── 1. Background colour from image border (20px frame) ──
border = np.zeros((H, W), bool)
bw = 40
border[:bw, :] = border[-bw:, :] = True
border[:, :bw] = border[:, -bw:] = True
bg_a, bg_b = np.median(A[border]), np.median(B[border])
print(f'bg chroma: a*={bg_a:.1f} b*={bg_b:.1f}')

# ── 2. Chroma distance from background, Otsu threshold ──
dist = np.sqrt((A - bg_a) ** 2 + (B - bg_b) ** 2)
dist8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
thr, mask = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'otsu on chroma dist: {thr/4:.1f}')

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# finger = largest contour near image centre
def centrality(c):
    x, y, w, h = cv2.boundingRect(c)
    cx, cy = x + w / 2, y + h / 2
    return np.hypot(cx - W / 2, cy - H / 2)
big = [c for c in cnts if cv2.contourArea(c) > 2000]
print('contours>2000px:', [(int(cv2.contourArea(c)), cv2.boundingRect(c)) for c in big])
finger_cnt = min(big, key=centrality)
fmask = np.zeros((H, W), np.uint8)
cv2.drawContours(fmask, [finger_cnt], -1, 255, -1)
fx, fy, fw, fh = cv2.boundingRect(finger_cnt)
print('finger bbox', fx, fy, fw, fh)

# ── 3. Inside finger: nail vs pulp split on a* (Otsu) ──
vals = A[fmask > 0]
a8 = np.clip((A - vals.min()) / (vals.ptp() + 1e-6) * 255, 0, 255).astype(np.uint8)
va = a8[fmask > 0]
thr_a, _ = cv2.threshold(va, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thr_a_real = vals.min() + thr_a / 255 * vals.ptp()
print(f'a* otsu split inside finger: {thr_a_real:.1f}')
nail = ((A < thr_a_real) & (fmask > 0)).astype(np.uint8) * 255
nail = cv2.morphologyEx(nail, cv2.MORPH_OPEN, k)
nail = cv2.morphologyEx(nail, cv2.MORPH_CLOSE, k, iterations=2)
ncnts, _ = cv2.findContours(nail, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
nail_cnt = max(ncnts, key=cv2.contourArea)
print('nail bbox', cv2.boundingRect(nail_cnt), 'area', cv2.contourArea(nail_cnt))

# ── visualize ──
vis = img.copy()
cv2.drawContours(vis, [finger_cnt], -1, (0, 255, 0), 2)
cv2.drawContours(vis, [nail_cnt], -1, (0, 255, 255), 2)
x0, y0 = max(fx - 60, 0), max(fy - 60, 0)
crop = vis[y0:fy + fh + 60, x0:fx + fw + 60]
cv2.imwrite('test/ccurve_seg.png', cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
print('saved test/ccurve_seg.png')
