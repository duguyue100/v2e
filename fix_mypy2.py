import re


# v2ecore/v2e_utils.py
with open("v2ecore/v2e_utils.py", "r") as f:
    c = f.read()
c = c.replace("cv2.VideoWriter(", "cv2.VideoWriter(str(")
c = c.replace("frame_size)", "frame_size))")
with open("v2ecore/v2e_utils.py", "w") as f:
    f.write(c)

# v2ecore/output/aedat2.py
with open("v2ecore/output/aedat2.py", "r") as f:
    c = f.read()
c = c.replace(
    "self.numEventsWritten += events.shape[0]",
    "self.numEventsWritten += int(events.shape[0])",
)
with open("v2ecore/output/aedat2.py", "w") as f:
    f.write(c)

# Unused ignores
files_to_check = [
    "v2ecore/synthetic/particles.py",
    "v2ecore/synthetic/moving_dot.py",
    "v2ecore/synthetic/gradients.py",
    "v2ecore/synthetic/barberpole.py",
    "v2ecore/model.py",
    "v2ecore/dataloader.py",
]

for file in files_to_check:
    with open(file, "r") as f:
        c = f.read()
    c = re.sub(r"  # type: ignore", "", c)
    c = re.sub(r"# type: ignore", "", c)
    with open(file, "w") as f:
        f.write(c)

# particles floats
with open("v2ecore/synthetic/particles.py", "r") as f:
    c = f.read()
c = c.replace("self.time: int = 0", "self.time: float = 0")
c = c.replace(
    "self.position: np.ndarray = np.array", "self.position: np.ndarray = np.array"
)
# Let's fix the int/float issue more robustly
c = re.sub(r"def fill_dot\((.*?)\):", r"def fill_dot(\1) -> None:", c)
with open("v2ecore/synthetic/particles.py", "w") as f:
    f.write(c)
