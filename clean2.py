import re

with open("v2ecore/emulator.py") as f:
    c = f.read()

# h5 creation block
c = re.sub(
    r"""\s*if dvs_h5:\s*path = os\.path\.join\(self\.output_folder, dvs_h5\)\s*path = checkAddSuffix\(path, '\.h5'\)\s*self\.dvs_h5 = h5py\.File\(path, "w"\)\s*self\.dvs_h5_dataset = self\.dvs_h5\.create_dataset\(\s*name="events",\s*shape=\(0, 4\),\s*maxshape=\(None, 4\),\s*dtype="uint32",\s*compression="gzip"\)""",
    "", c, flags=re.MULTILINE | re.DOTALL
)

c = re.sub(
    r"""\s*if self\.dvs_h5:\s*self\.frame_h5_dataset = self\.dvs_h5\.create_dataset\(\s*name="frame",\s*shape=\(0, self\.output_height,\s*self\.output_width\),\s*maxshape=\(None, self\.output_height,\s*self\.output_width\),\s*dtype="uint8",\s*compression="gzip"\)\s*self\.frame_ts_dataset = self\.dvs_h5\.create_dataset\(\s*name="frame_ts",\s*shape=\(0,\),\s*maxshape=\(None,\),\s*dtype="float32",\s*compression="gzip"\)\s*self\.frame_ev_idx_dataset = self\.dvs_h5\.create_dataset\(\s*name="frame_event_idx",\s*shape=\(0,\),\s*maxshape=\(None,\),\s*dtype="uint32",\s*compression="gzip"\)\s*else:\s*self\.frame_h5_dataset = None""",
    "", c, flags=re.MULTILINE | re.DOTALL
)

with open("v2ecore/emulator.py", "w") as f:
    f.write(c)
