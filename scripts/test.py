
from tolerancing.geometry import *
import numpy as np

nullgeo = NullGeometry()
pt1 = Point(origin=[0,0,1])
pt2 = Point(origin=[1,1,0])
pt3 = Point(origin=[0,1,1])

# print(nullgeo, pt1, pt2)
# print(pt1.distance(pt2))
# print(nullgeo.intersection(pt1))
# print(pt1.intersection(pt3))

ax = Axis(origin=[0,0,0], u=[0,1,1])
ax2 = Axis(origin=[0,4,0], u=[0,1,0])
ax3 = Axis(origin=[1,4,0], u=[0,1,0])
# print(ax)
# print(ax2)
# print(ax3)


pt_inter = ax.intersection(ax2)
# print(pt_inter)
# print(ax.distance(ax2))
# print(ax.distance(ax3))


newaxs = ax.derive(mode="same", du=1, dv=2, dw=3)
# print(newaxs)
# print(newaxs.distance(ax))

plane = Plane(origin=[0,0,0],u=[0,1,0])
plane2 = Plane(origin=[0,2,0],u=[0,1,0])
plane3 = Plane(origin=[3,1,2],u=[1,2,3])
# print(plane, plane2, plane3)

# print(plane.distance(plane2), plane.distance(plane3))

# print(plane3.convert(np.array([1,-1,2])))
# print(plane3.convert(np.array([1,-1,2]), forward=False))

newax = plane2.intersection(plane3)
# print(newax)
# print(newax.u, newax.v, newax.w)

newax2 = plane3.derive_dual()
# print(newax2)

newplane = newax.derive_dual()
# print(newplane)

newcyl = Cylinder(origin=[0,0,0],u=[0,1,1],r=2)
print(newcyl)

print(newcyl.coordinate(2, 3.1415/3))

der_cyl = newcyl.derive(mode="same", 
                        rx = np.radians(30),
                        ry = np.radians(-30),
                        rz = np.radians(90))

print(der_cyl)

# frame = newcyl.frame

# rotated_frame = newcyl.rotate_frame(frame, 10,10,-10,deg = True)
# print(frame)
# print(rotated_frame)

