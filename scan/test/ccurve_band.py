"""Prototype: trace the white free-edge band hugging the pulp."""
import cv2
import numpy as np

img = cv2.imread('sh_images/sh_middle_ccurve.jpg')
H, W = img.shape[:2]
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
A = lab[:, :, 1].astype(np.float32) - 128
B = lab[:, :, 2].astype(np.float32) - 128

# finger mask (validated approach)
bw = 40
border = np.zeros((H, W), bool)
border[:bw, :] = border[-bw:, :] = True
border[:, :bw] = border[:, -bw:] = True
bg_b = np.median(B[border])
mask = (B > bg_b + 13).astype(np.uint8) * 255      # gap centre from run: -23.1
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
finger_cnt = min([c for c in cnts if cv2.contourArea(c) > 1500],
                 key=lambda c: np.hypot(*(np.array(cv2.boundingRect(c)[:2]) +
                                          np.array(cv2.boundingRect(c)[2:]) / 2 -
                                          (W / 2, H / 2))))
fmask = np.zeros((H, W), np.uint8)
cv2.drawContours(fmask, [finger_cnt], -1, 255, -1)
fx, fy, fw, fh = cv2.boundingRect(finger_cnt)

# nail/pulp split
vals = A[fmask > 0]
a8 = np.clip((A - vals.min()) / (np.ptp(vals) + 1e-6) * 255, 0, 255).astype(np.uint8)
t, _ = cv2.threshold(a8[fmask > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thr_a = vals.min() + t / 255.0 * np.ptp(vals)
nail = ((A < thr_a) & (fmask > 0)).astype(np.uint8) * 255
pulp = ((A >= thr_a) & (fmask > 0)).astype(np.uint8) * 255
for m in (nail, pulp):
    tmp = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m[:] = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, k, iterations=2)
# largest pulp component
pc, _ = cv2.findContours(pulp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
pulp_cnt = max(pc, key=cv2.contourArea)
pulp2 = np.zeros((H, W), np.uint8)
cv2.drawContours(pulp2, [pulp_cnt], -1, 255, -1)

# rough scale from nail x-extent
cols_any = np.where(nail.any(axis=0))[0]
rough_scale = 9.63 / (cols_any[-1] - cols_any[0])
thick_px = 0.85 / rough_scale
print(f'rough scale {rough_scale:.4f} mm/px  thickness ~{thick_px:.0f}px')

# band = nail within 1.6*thickness of pulp
rad = int(1.6 * thick_px) | 1
kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
band = cv2.bitwise_and(cv2.dilate(pulp2, kd), nail)
bc, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
band_cnt = max(bc, key=cv2.contourArea)
band2 = np.zeros((H, W), np.uint8)
cv2.drawContours(band2, [band_cnt], -1, 255, -1)
bx, by, bw2, bh2 = cv2.boundingRect(band_cnt)
print('band bbox', bx, by, bw2, bh2)

# top boundary per column
cols = np.where(band2.any(axis=0))[0]
top = np.array([np.argmax(band2[:, c] > 0) for c in cols], float)
xs, ys = cols.astype(float), top

# robust circle fit
keep = np.ones(len(xs), bool)
for _ in range(8):
    M = np.column_stack([xs[keep], ys[keep], np.ones(keep.sum())])
    b = xs[keep] ** 2 + ys[keep] ** 2
    sol, *_ = np.linalg.lstsq(M, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    res = np.abs(np.hypot(xs - cx, ys - cy) - r)
    s = max(np.median(res[keep]) * 2.5, 1.5)
    nk = res < s
    if nk.sum() < 8 or (nk == keep).all():
        break
    keep = nk
print(f'circle fit R={r:.1f}px centre=({cx:.0f},{cy:.0f}) inliers={keep.sum()}/{len(xs)}')

xi, yi = xs[keep], ys[keep]
iL, iR = np.argmin(xi), np.argmax(xi)
xL, yL, xR, yR = xi[iL], yi[iL], xi[iR], yi[iR]
chord = np.hypot(xR - xL, yR - yL)
scale = 9.63 / chord
nx, ny = -(yR - yL) / chord, (xR - xL) / chord
d = np.abs((xi - xL) * nx + (yi - yL) * ny)
h_px = d.max()
h_mm = h_px * scale
R_mm = 9.63 ** 2 / (8 * h_mm) + h_mm / 2
half = min(1.0, chord / (2 * r))
arc_len = 2 * r * np.arcsin(half) * scale
print(f'chord={chord:.1f}px  h={h_mm:.2f}mm  R={R_mm:.2f}mm  fitR={r*scale:.2f}mm  arc_len={arc_len:.2f}mm')

# visualize
vis = img.copy()
ov = vis.copy()
ov[band2 > 0] = (0, 200, 255)
ov[pulp2 > 0] = (200, 0, 200)
cv2.addWeighted(ov, 0.3, vis, 0.7, 0, vis)
for xa, ya, kp in zip(xs, ys, keep):
    cv2.circle(vis, (int(xa), int(ya)), 1, (0, 255, 255) if kp else (0, 0, 255), -1)
cv2.line(vis, (int(xL), int(yL)), (int(xR), int(yR)), (255, 200, 0), 1)
crop = vis[fy - 60:fy + fh + 60, fx - 60:fx + fw + 60]
cv2.imwrite('test/ccurve_band.png',
            cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
print('saved test/ccurve_band.png')
