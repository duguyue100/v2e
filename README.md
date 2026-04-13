# v2e 2.0

v2e is a tool for generating synthetic DVS (Dynamic Vision Sensor) event streams from conventional video, using PyTorch-based Super-SloMo neural network frame interpolation and detailed DVS pixel simulation.

## Installation

```bash
pip install v2e
```

For optional dependencies (faster processing, AEDAT-4.0 output):
```bash
pip install v2e[numba,aedat4]
```

Requires Python 3.11+.

## Library Usage

`v2e` is now a cleanly separated library:

```python
import numpy as np
from v2ecore import EventEmulator, V2EConfig

config = V2EConfig()
emulator = EventEmulator(
    pos_thres=config.dvs.pos_thres,
    neg_thres=config.dvs.neg_thres,
    sigma_thres=config.dvs.sigma_thres,
    cutoff_hz=config.dvs.cutoff_hz,
    leak_rate_hz=config.dvs.leak_rate_hz,
    refractory_period_s=config.dvs.refractory_period_s,
    shot_noise_rate_hz=config.dvs.shot_noise_rate_hz,
    photoreceptor_noise=config.dvs.photoreceptor_noise or False,
    leak_jitter_fraction=config.dvs.leak_jitter_fraction,
    noise_rate_cov_decades=config.dvs.noise_rate_cov_decades,
    seed=config.dvs.seed,
    output_width=346,
    output_height=260,
    device=config.device,
)

frame1 = np.random.rand(260, 346).astype(np.float32) * 128
frame2 = frame1 + 30  # brightness increase

events1 = emulator.generate_events(frame1, t_frame=0.0)
events2 = emulator.generate_events(frame2, t_frame=0.001)

if events2 is not None:
    print(f"Generated {len(events2)} events")
```

## Development

```bash
git clone https://github.com/SensorsINI/v2e.git
cd v2e
pip install -e ".[dev]"
pre-commit install
```
