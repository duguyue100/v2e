import re

with open("v2ecore/emulator.py") as f:
    content = f.read()

# 1. Remove args from __init__
content = re.sub(
    r"""\s*output_folder: str = None,\s*dvs_h5: str = None,\s*dvs_aedat2: str = None,\s*dvs_aedat4: str = None,\s*dvs_text: str = None,""",
    "",
    content,
    flags=re.MULTILINE
)

# 2. Remove args docstrings
content = re.sub(
    r"""\s*dvs_aedat2, dvs_aedat4, dvs_h5, dvs_text: str\s*names of output data files or None""",
    "",
    content,
    flags=re.MULTILINE
)

# 3. Remove init bodies for files
content = re.sub(
    r"""\s*# h5 output\s*self\.output_folder = output_folder\s*self\.dvs_h5 = dvs_h5\s*self\.dvs_h5_dataset = None\s*self\.frame_h5_dataset = None\s*self\.frame_ts_dataset = None\s*self\.frame_ev_idx_dataset = None\s*# aedat or text output\s*self\.dvs_aedat2 = dvs_aedat2\s*self\.dvs_aedat4 = dvs_aedat4\s*self\.dvs_text = dvs_text""",
    "",
    content,
    flags=re.MULTILINE
)

# 4. Remove h5 file creation
content = re.sub(
    r"""\s*if self\.dvs_h5:\s*path = os\.path\.join\(self\.output_folder, self\.dvs_h5\).*?self\.frame_ev_idx_dataset = None""",
    "",
    content,
    flags=re.MULTILINE | re.DOTALL
)

# 5. Remove h5 file close
content = re.sub(
    r"""\s*if self\.dvs_h5 is not None:\s*self\.dvs_h5\.close\(\)""",
    "",
    content,
    flags=re.MULTILINE
)

# 6. Remove frame saving
content = re.sub(
    r"""\s*# like a DAVIS, write frame into the file if it's HDF5\s*if self\.frame_h5_dataset is not None:\s*# save frame data\s*self\.frame_h5_dataset\[self\.frame_counter\] = \\\s*new_frame\.astype\(np\.uint8\)""",
    "",
    content,
    flags=re.MULTILINE
)

# 7. Remove event saving block
content = re.sub(
    r"""\s*if self\.dvs_h5 is not None:\s*# convert data to uint32.*?if self\.dvs_text is not None:\s*if self\.label_signal_noise:\s*self\.dvs_text\.appendEvents\(events, signnoise_label=signnoise_label\)\s*else:\s*self\.dvs_text\.appendEvents\(events\)""",
    "",
    content,
    flags=re.MULTILINE | re.DOTALL
)

# 8. Remove frame event idx save
content = re.sub(
    r"""\s*if self\.frame_ev_idx_dataset is not None:\s*# save frame event idx\s*# determine after the events are added\s*self\.frame_ev_idx_dataset\[self\.frame_counter - 1\] = \\\s*self\.dvs_h5_dataset\.shape\[0\]""",
    "",
    content,
    flags=re.MULTILINE
)

with open("v2ecore/emulator.py", "w") as f:
    f.write(content)
