import cadquery as cq

# Create a simple sketch
sk = cq.Sketch().rect(2, 2).clean()

# Convert to workplane face
wp1 = cq.Workplane("XY").add(sk).extrude(0.001).faces("<Z")
wp2 = wp1.translate((1.5, 0, 0)) # overlap

# Try union?
try:
    wp1.union(wp2)
    print("union worked")
except Exception as e:
    print("union failed:", str(e))

# Try add then extrude?
try:
    wp_all = cq.Workplane("XY").add(wp1.val()).add(wp2.val())
    solid = wp_all.extrude(1.0)
    print("add+extrude worked! Solids count:", len(solid.solids().vals()))
except Exception as e:
    print("add+extrude failed:", str(e))
