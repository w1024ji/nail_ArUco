import struct, math
from collections import defaultdict

path = r"C:\nail_ArUco\results\stl_v13\nail_index_exact.stl"
with open(path, "rb") as f:
    f.read(80); n = struct.unpack("<I", f.read(4))[0]
    tris = []
    for _ in range(n):
        f.read(12)
        tris.append([struct.unpack("<3f", f.read(12)) for _ in range(3)])
        f.read(2)

print(f"Triangles: {n}")

def r3(v): return tuple(round(x,3) for x in v)
ec = defaultdict(int)
for t in tris:
    for i in range(3): ec[(r3(t[i]), r3(t[(i+1)%3]))] += 1

bad, seen = [], set()
for (v0,v1),c in ec.items():
    rev = ec.get((v1,v0),0)
    k = tuple(sorted([v0,v1]))
    if k not in seen:
        seen.add(k)
        if c!=1 or rev!=1: bad.append((c,rev,v0,v1))

print("MANIFOLD" if not bad else f"NON-MANIFOLD: {len(bad)} bad edges")
if bad:
    for c,r,v0,v1 in bad[:3]: print(f"  fwd={c} rev={r} {v0[:2]}..{v1[:2]}")

def cross3(a,b,c):
    u=(b[0]-a[0],b[1]-a[1],b[2]-a[2]); v=(c[0]-a[0],c[1]-a[1],c[2]-a[2])
    return math.sqrt((u[1]*v[2]-u[2]*v[1])**2+(u[2]*v[0]-u[0]*v[2])**2+(u[0]*v[1]-u[1]*v[0])**2)/2
degen = sum(1 for t in tris if cross3(*t)<1e-6)
print(f"Degenerate (3D): {degen}")

all_v=[v for t in tris for v in t]
for ax,nm in enumerate("XYZ"):
    vals=[v[ax] for v in all_v]
    print(f"  {nm}: {min(vals):.2f}..{max(vals):.2f}  span={max(vals)-min(vals):.2f}mm")

# Check C-curve: bottom face should have z varying across width
bot_pts = [(v[0],v[2]) for t in tris for v in t if v[2] < 0.1]
if bot_pts:
    zs = [v[1] for v in bot_pts]
    xs = [v[0] for v in bot_pts]
    print(f"  Bottom z range: {min(zs):.3f}..{max(zs):.3f}mm (should be ~0..C={1.81})")
