with open("v2ecore/synthetic/particles.py", "r") as f:
    c = f.read()

# Fix floats initialized as ints
c = c.replace("self.radius: int", "self.radius: float")
c = c.replace("self.d_radius: int", "self.d_radius: float")
c = c.replace("self.x: int", "self.x: float")
c = c.replace("self.y: int", "self.y: float")
c = c.replace("self.vx: int", "self.vx: float")
c = c.replace("self.vy: int", "self.vy: float")
c = c.replace("self.time: int", "self.time: float")

with open("v2ecore/synthetic/particles.py", "w") as f:
    f.write(c)
