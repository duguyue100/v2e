import h5py
import numpy as np
from typing import Any


class Hdf5EventWriter:
    """Writes events to HDF5 format."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.file = h5py.File(filepath, "w")
        self.dvs_h5_dataset = self.file.create_dataset(
            name="events",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype="float32",
            compression="gzip",
        )

    def write(self, events: Any) -> None:
        if events is None or len(events) == 0:
            return
        self.dvs_h5_dataset.resize(
            self.dvs_h5_dataset.shape[0] + events.shape[0],
            axis=0,
        )
        self.dvs_h5_dataset[-events.shape[0] :] = events

    def close(self) -> None:
        self.file.close()
