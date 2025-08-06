
from tolerancing.geometry import *
import numpy as np

nullgeo = NullGeometry()
# pt1 = Point(origin=[0,0,1])
# pt2 = Point(origin=[1,1,0])
# pt3 = Point(origin=[0,1,1])

# ax = Axis(origin=[0,0,0], u=[0,1,1])
# ax2 = Axis(origin=[0,4,0], u=[0,1,0])
# ax3 = Axis(origin=[1,4,0], u=[0,1,0])

# pt_inter = ax.intersection(ax2)

# newaxs = ax.derive(mode="same", du=1, dv=2, dw=3)

# plane = Plane(origin=[0,0,0],u=[0,1,0])
# plane2 = Plane(origin=[0,2,0],u=[0,1,0])
# plane3 = Plane(origin=[3,1,2],u=[1,2,3])

# newax = plane2.intersection(plane3)

# newax2 = plane3.derive_dual()

# newplane = newax.derive_dual()

# newcyl = Cylinder(origin=[0,0,0],u=[0,1,0],r=2)

# der_cyl = newcyl.derive(mode="same", 
#                         rx = np.radians(30),
#                         ry = np.radians(30),
#                         rz = np.radians(60))


top = Plane(origin=[0,0,0],u=[0,0,1]) #top plane
boss = top.derive(new_geo=10, du=3)
boss_rotate = boss.derive(new_geo=10, du=3, dz=np.radians(45))
ax1 = boss_rotate.derive(new_geo=1,at_point=[1,1,0], direction=boss_rotate.u)
ax2 = boss_rotate.derive(new_geo=1,at_point=[-1,-1,0], direction=boss_rotate.u)
ax3=ax2.derive(new_geo=1,dv=2, dx=np.radians(30))

# print(top)
# print(boss)
# print(boss_rotate)
# print(ax1)
# print(ax2)
# print(ax3)

# print(ax3.aorigin)
# print(ax3.aframe)


