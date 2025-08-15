

import numpy as np

from tolerancing.geometry import GeometryType
from tolerancing.geometry.cylinder import Cylinder


c1 = Cylinder(origin=[0,0,1], u = [0,1,0], r=2)
c2 = Cylinder(origin=[0,.5,1], u = [0,1,1], r=5)
c3 = Cylinder(origin=[100,0,0], u = [.5,-1,.25], r=10)

print(c1)

urange = np.arange(-1,1,.1)
thetarange = np.arange(0, 6.28, 3.14/6)
vrange = np.cos(thetarange)
wrange = np.sin(thetarange)

for u in urange:
    print(f"radial tangent at point:")
    for v,w in zip(vrange, wrange):
        
        xyz = c1.convert(np.array([u,v,w]),forward=False)
        x,y,z = xyz
        localframe = c1.get_local_frame(u, v, w)
        
        # print(f"internal coordinate (uvw):")
        # print(f"({u:0.1g},{v:0.1g},{w:0.1g})")

        # print(f"global coordinate (xyz):")
        # print(f"({x:0.1g},{y:0.1g},{z:0.1g})")
        
        
        # print(f"axial tangent at point:")
        # print(f"u={localframe[0]}")
        
        # print(f"radial normal at point:")
        print(f"v={localframe[1]}")
        
        
        print(f"w={localframe[2]}")

    break
