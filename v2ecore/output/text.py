import atexit
import logging
from typing import Any

import numpy as np
from engineering_notation import EngNumber  # only from pip


logger = logging.getLogger(__name__)


class TextEventWriter:
    """
    outputs text format DVS events to file according to events.txt format in http://rpg.ifi.uzh.ch/davis_data.html

    The RPG DVS text file datatset looks like this. Each line has (time(float s), x, y, polarity (0=off,1=on)

        0.000000000 33 39 1
        0.000011001 158 145 1
        0.000050000 88 143 0
        0.000055000 174 154 0
        0.000080001 112 139 1
        0.000123000 136 171 0
        0.000130001 173 90 0
        0.000139001 106 140 0
        0.000148001 192 79 1
    """

    def __init__(self, filepath: str, label_signal_noise: bool = False):
        """Constructs the CSV writer
        :param filepath: the full path to file
        :param label_signal_noise: set True to append column labeling signal (1) and noise (0)
        """
        self.filepath = filepath
        # edit below to match your device from https://inivation.com/support/software/fileformat/#aedat-20
        self.numEventsWritten = 0
        self.label_signal_noise = label_signal_noise
        self.flipx = (
            False  # set both flipx and flipy to rotate TODO replace with rotate180
        )
        self.flipy = False
        self.sizex = 346
        self.sizey = 260  # adjust to your needs
        logging.info("opening text DVS output file %s", filepath)
        self.file = open(filepath, "w")
        self._writeHeader()
        atexit.register(self.cleanup)

    def cleanup(self) -> None:
        self.close()

    def close(self) -> None:
        if self.file:
            logger.info(
                "Closing %s after writing %s events",
                self.filepath,
                EngNumber(self.numEventsWritten),
            )
            self.file.close()
            self.file = None  # type: ignore

    def _writeHeader(self) -> None:
        import datetime
        import getpass
        import time

        if not self.label_signal_noise:
            format = "# Format is time (float s), x, y, polarity (0=off, 1=on) as specified at http://rpg.ifi.uzh.ch/davis_data.html\n"
        else:
            format = "# Format is time (float s), x, y, polarity (0=off, 1=on), signal/noise (1/0)\n#  as specified at http://rpg.ifi.uzh.ch/davis_data.html\n"
        date = datetime.datetime.now().strftime(
            "# Creation time: %I:%M%p %B %d %Y\n"
        )  # Tue Jan 26 13:57:06 CET 2016
        time_str = (
            f"# Creation time: System.currentTimeMillis() {int(time.time() * 1000.)}\n"
        )
        user = f"# User name: {getpass.getuser()}\n"
        header = (
            "#!events.txt\n",
            "# This is a text DVS created by v2e (see https://github.com/SensorsINI/v2e)\n",
            format,
            date,
            time_str,
            user,
        )
        for s in header:
            if self.file:
                self.file.write(s)

    def write(self, events: Any, signnoise_label: Any = None) -> None:
        """Append events to text output

        Parameters
        ----------
         events: Any with N events if any events, else None
             [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
             NOTE x,y, NOT y,x.
        signnoise: Any
            [N] each entry is 1 for signal or 0 for noise

        Returns:
        -------
         None
        """
        if self.file is None:
            raise Exception("output file closed already")

        if len(events) == 0:
            return
        n = events.shape[0]
        t = (events[:, 0]).astype(float)
        x = events[:, 1].astype(np.int32)  # Issue #37, thanks Mohsi Jawaid
        if self.flipx:
            x = (self.sizex - 1) - x  # 0 goes to sizex-1
        y = events[:, 2].astype(np.int32)
        if self.flipy:
            y = (self.sizey - 1) - y
        p = ((events[:, 3] + 1) / 2).astype(np.int32)  # go from -1/+1 to 0,1
        for i in range(n):
            if signnoise_label is None:
                if self.file:
                    self.file.write(
                        f"{t[i]} {x[i]} {y[i]} {p[i]}\n"
                    )  # todo there must be vector way
            else:
                if self.file:
                    self.file.write(
                        f"{t[i]} {x[i]} {y[i]} {p[i]} {int(signnoise_label[i])}\n"
                    )  # write with additonal signal/noise label column cast to int (1=signal, 0=noise)
        self.numEventsWritten += n


# class TextEventWriterTest: # test from src.output.ae_text_output import TextEventWriterTest
#     f = TextEventWriter('aedat-text-test.txt')
#     e = [[0., 0, 0, 0], [1e-6, 0, 0, 1], [2e-6, 1, 0, 0]]
#     ne = np.array(e)
#     f.write(ne)
#     e = [[3e-6, 0, 0, 1], [5e-6, 0, 0, 1], [9e-6, 1, 0, 0]]
#     ne = np.array(e)
#     f.write(ne)
#     f.close()
