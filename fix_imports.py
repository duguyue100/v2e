import glob

for f in glob.glob("v2ecore/synthetic/*.py"):
    with open(f, "r") as file:
        c = file.read()
    
    if "from v2ecore.v2e_utils" not in c:
        c = "from v2ecore.v2e_utils import all_images, read_image, checkAddSuffix, v2e_quit, video_writer, check_lowpass, njit\n" + c
        with open(f, "w") as file:
            file.write(c)
