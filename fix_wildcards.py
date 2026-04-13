import glob

for f in glob.glob("v2ecore/synthetic/*.py"):
    with open(f) as file:
        c = file.read()
    c = c.replace("from v2ecore.v2e_utils import *", "from v2ecore.v2e_utils import all_images, read_image, checkAddSuffix, v2e_quit, video_writer, check_lowpass")
    with open(f, "w") as file:
        file.write(c)
