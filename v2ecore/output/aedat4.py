import dv_processing

import logging

# check https://gitlab.com/inivation/dv/dv-processing to install dv-processing-python

import numpy as np
from typing import Any
from engineering_notation import EngNumber  # only from pip

logger = logging.getLogger(__name__)


class Aedat4EventWriter:
    """
    outputs AEDAT-4.0 jAER format DVS data from v2e
    """

    def __init__(self, filepath: str, output_width: int=640, output_height: int=480) -> None:
        self.filepath = filepath
        self.numEventsWritten = 0
        self.numOnEvents=0
        self.numOffEvents=0
        logging.info(f'opening AEDAT-4.0 output file {filepath} in binary mode')

        self.flipy = False
        self.flipx = False
        self.sizex = output_width
        self.sizey = output_height

        self.store = dv_processing.EventStore()

        resolution = (640, 480)
        # Event only configuration
        config = dv_processing.io.MonoCameraWriter.EventOnlyConfig("DVXplorer_sample", resolution)

        # Create the writer instance, it will only have a single event output stream.
        self.writer = dv_processing.io.MonoCameraWriter(filepath, config)

    def cleanup(self) -> None:
        self.close()

    def close(self) -> None:
        if self.writer:
            logger.info(f"Closing {self.filepath} after writing {EngNumber(self.numEventsWritten)} events ({EngNumber(self.numOnEvents)} on, {EngNumber(self.numOffEvents)} off)")

            self.writer.writeEvents(self.store)
            self.writer = None

    def write(self, events: Any, signnoise_label: Any = None) -> None:
        """Append events to AEDAT-4.0 output

        Parameters
        ----------
        events: Any if any events, else None
            [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
            NOTE x,y, NOT y,x.
        signnoise: Any
          [N] each entry is 1 for signal or 0 for noise

        Returns
        -------
        None
        """
        if self.writer is None:
            return

        if len(events) == 0:
            return
        n = events.shape[0]
        for event in events:
            t = int(event[0] * 1e6)
            x = int(event[1])
            if self.flipx: x = (self.sizex - 1) - x  # 0 goes to sizex-1
            y = int(event[2])
            if self.flipy: y = (self.sizey - 1) - y
            p = int((event[3] + 1) / 2) # 0=off, 1=on

            try:
                 self.store.push_back(t, x, y, p)
            except RuntimeError as e:
                 logger.warning(f'caught exception event {e} to store')

            if p==1: self.numOnEvents+=1
            else: self.numOffEvents+=1
            self.numEventsWritten += 1

        # logger.info('wrote {} events'.format(n))


if __name__ == '__main__':
    class Aedat4EventWriterTt:
        f = Aedat4EventWriter('aedattest.aedat4')
        e = [[1, 400, 0, 0], [2, 0, 400, 0], [3, 300, 400, 0], [4, 400, 300, 1], [5, 400, 300, 0]]
        ne = np.array(e)
        eventsNum = 2000 * 5
        nne = np.tile(ne, (int(eventsNum/5), 1))
        nne[:, 0] = np.arange(1, eventsNum + 1)
        f.write(nne)
        print(f'wrote {nne.shape[0]} events')
        f.close()
