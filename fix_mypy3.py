import re


# synthetic_utils
with open("v2ecore/synthetic/synthetic_utils.py", "r") as f:
    c = f.read()
c = c.replace("dot_sigma: float,\n):", "dot_sigma: float,\n) -> None:")
with open("v2ecore/synthetic/synthetic_utils.py", "w") as f:
    f.write(c)

# aedat2
with open("v2ecore/output/aedat2.py", "r") as f:
    c = f.read()
c = c.replace(
    "self.numEventsWritten += int(events.shape[0])",
    "self.numEventsWritten += int(events.shape[0])",
)
with open("v2ecore/output/aedat2.py", "w") as f:
    f.write(c)

# Remove unused type ignores
for file in [
    "v2ecore/synthetic/particles.py",
    "v2ecore/synthetic/moving_dot.py",
    "v2ecore/synthetic/gradients.py",
    "v2ecore/synthetic/barberpole.py",
]:
    with open(file, "r") as f:
        c = f.read()
    c = re.sub(r"\s*# type: ignore", "", c)
    with open(file, "w") as f:
        f.write(c)

# fix particles
with open("v2ecore/synthetic/particles.py", "r") as f:
    c = f.read()
c = c.replace("radius: int =", "radius: float =")
c = c.replace("x: int =", "x: float =")
c = c.replace("y: int =", "y: float =")
c = c.replace("vx: int =", "vx: float =")
c = c.replace("vy: int =", "vy: float =")
c = c.replace("time: int =", "time: float =")
c = c.replace(
    "def fill_dot(\n    pix_arr: np.ndarray, x: float, y: float, fg: float, bg: float, radius: float\n):",
    "def fill_dot(\n    pix_arr: np.ndarray, x: float, y: float, fg: float, bg: float, radius: float\n) -> None:",
)
with open("v2ecore/synthetic/particles.py", "w") as f:
    f.write(c)
