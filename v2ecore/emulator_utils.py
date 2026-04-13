"""Collections of emulator utilities.

Author: Yuhuang Hu, Tobi Delbruck
Email : yuhuang.hu@ini.uzh.ch, tobi@ini.uzh.ch
"""
import logging
import math
from typing import Any, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def lin_log(x: Any, threshold: int = 20) -> torch.Tensor:
    """
    Linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    f = (1./threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x*f, torch.log(x))

    rounding = 1e8
    y = torch.round(y*rounding)/rounding

    return y.float()


def rescale_intensity_frame(new_frame: Any) -> Any:
    """Rescale intensity frames."""
    return (new_frame+20)/275.


def low_pass_filter(
        log_new_frame: Any,
        lp_log_frame: Any,
        inten01: Any,
        delta_time: Any,
        cutoff_hz: float = 0) -> Any:
    """Compute intensity-dependent low-pass filter."""
    if cutoff_hz <= 0:
        return log_new_frame

    tau = 1/(math.pi*2*cutoff_hz)

    if inten01 is not None:
        eps = inten01*(delta_time/tau)
        max_eps = torch.max(eps)
        if max_eps >0.3:
            IIR_MAX_WARNINGS = 10
            if low_pass_filter.iir_warning_count<IIR_MAX_WARNINGS:  # type: ignore
                logger.warning(f'IIR lowpass filter update has large maximum update eps={max_eps:.2f} from delta_time/tau={delta_time:.3g}/{tau:.3g}')
                low_pass_filter.iir_warning_count+=1  # type: ignore
                if low_pass_filter.iir_warning_count==IIR_MAX_WARNINGS:  # type: ignore
                    logger.warning('Supressing further warnings about inaccurate IIR lowpass filtering')

        eps = torch.clamp(eps, max=1)
    else:
        eps=delta_time/tau

    new_lp_log_frame = (1-eps)*lp_log_frame+eps*log_new_frame

    return new_lp_log_frame

low_pass_filter.iir_warning_count=0  # type: ignore


def subtract_leak_current(base_log_frame: Any,
                          leak_rate_hz: float,
                          delta_time: float,
                          pos_thres: Any,
                          leak_jitter_fraction: float,
                          noise_rate_array: Any) -> Any:
    """Subtract leak current from base log frame."""
    rand = torch.randn(
        noise_rate_array.shape, dtype=torch.float32,
        device=noise_rate_array.device)

    curr_leak_rate = \
        leak_rate_hz*noise_rate_array*(1-leak_jitter_fraction*rand)

    delta_leak = delta_time*curr_leak_rate*pos_thres  

    return base_log_frame-delta_leak


def compute_event_map(diff_frame: Any, pos_thres: Any, neg_thres: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute event maps"""
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)

    pos_evts_frame = torch.div(
        pos_frame, pos_thres, rounding_mode="floor").type(torch.int32)
    neg_evts_frame = torch.div(
        neg_frame, neg_thres, rounding_mode="floor").type(torch.int32)

    return pos_evts_frame, neg_evts_frame


def compute_photoreceptor_noise_voltage(shot_noise_rate_hz: float, f3db: float, sample_rate_hz: float, pos_thr: float, neg_thr: float, sigma_thr: float) -> float:
    """Computes the necessary photoreceptor noise voltage"""

    def compute_vn_from_log_rate_per_hz(thr: float, x: float) -> float:
        y = -0.0026 * x ** 3 - 0.036 * x ** 2 - 0.1949 * x + 0.321
        thr_per_vn = 10 ** y  
        vn = thr / thr_per_vn  
        return vn

    if compute_photoreceptor_noise_voltage.last_sample_rate is not None:  # type: ignore
        diff=np.abs(sample_rate_hz/compute_photoreceptor_noise_voltage.last_sample_rate-1)  # type: ignore
        if diff<0.1:
            return float(compute_photoreceptor_noise_voltage.last_vn) # type: ignore

    rate_per_bw= (shot_noise_rate_hz / f3db) / 2 
    if rate_per_bw>0.5:
        logger.warning(f'shot noise rate per hz of bandwidth is larger than 0.1')
    x=math.log10(rate_per_bw)
    if x<-5.0:
        logger.warning(f'desired noise rate is too low')
    elif x>0.0:
        logger.warning(f'desired noise rate is too large')

    N=300 
    pos_samps=pos_thr+sigma_thr*np.random.default_rng().standard_normal(N)
    neg_samps=neg_thr+sigma_thr*np.random.default_rng().standard_normal(N)
    thrs=np.vstack((pos_samps,neg_samps))
    mins=np.min(thrs,axis=0)
    vns=np.zeros_like(mins)
    for i in range(N):
        thr=float(mins[i])
        vn = compute_vn_from_log_rate_per_hz(thr, x)
        vns[i]=vn

    vn=float(np.mean(vns))
    
    compute_photoreceptor_noise_voltage.last_sample_rate=sample_rate_hz  # type: ignore
    tau=1/(f3db*2*math.pi)
    dt=1/sample_rate_hz
    t=np.arange(0,1000*tau,dt)
    rin = vn*np.random.default_rng().standard_normal(t.shape) 
    rms_in=np.std(rin) 
    rout=np.zeros_like(rin)
    
    eps=dt/tau
    eps_limit=.1
    if eps>eps_limit:
        logger.warning(f'eps={eps:.3f} for IIR lowpass is >{eps_limit}')
    rout[0]=0 
    
    for i in range(1,len(rin)):
        rout[i]=rout[i-1]*(1-eps)+rin[i]*eps
    rms_out=np.std(rout) 
    scale=rms_in/rms_out 
    vnscaled=float(scale*vn) 
    new_rms_out=np.std(scale*rin) 

    compute_photoreceptor_noise_voltage.last_vn=vnscaled  # type: ignore
    
    if not compute_photoreceptor_noise_voltage.vrms_computation_printed:  # type: ignore
        logger.info(f'For desired shot_noise_rate_hz={shot_noise_rate_hz} Hz, computed photoreceptor_noise_rms={vn:.3f}')
        compute_photoreceptor_noise_voltage.vrms_computation_printed=True  # type: ignore
    return vnscaled

compute_photoreceptor_noise_voltage.vrms_computation_printed=False  # type: ignore
compute_photoreceptor_noise_voltage.last_sample_rate=None  # type: ignore
compute_photoreceptor_noise_voltage.last_vn=None  # type: ignore

def generate_shot_noise(
        shot_noise_rate_hz: float,
        delta_time: float,
        shot_noise_inten_factor: float,
        inten01: Any,
        pos_thres_pre_prob: Any,
        neg_thres_pre_prob: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate shot noise."""
    if shot_noise_rate_hz*delta_time>1:
        logger.warning(f'shot_noise_rate_hz*delta_time={shot_noise_rate_hz:.2f}*{delta_time:.2g}={shot_noise_rate_hz*delta_time:.2f} is too large')

    shot_noise_factor = (
        (shot_noise_rate_hz/2)*delta_time) * \
        ((shot_noise_inten_factor-1)*inten01+1) 

    one_minus_shot_ON_prob_this_sample = \
        1 - shot_noise_factor*pos_thres_pre_prob 
    shot_OFF_prob_this_sample = \
        shot_noise_factor*neg_thres_pre_prob 

    rand01 = torch.rand(
        size=inten01.shape,
        dtype=torch.float32,
        device=inten01.device)  

    shot_on_cord = torch.gt(
        rand01, one_minus_shot_ON_prob_this_sample)
    shot_off_cord = torch.lt(
        rand01, shot_OFF_prob_this_sample)

    return shot_on_cord, shot_off_cord

if __name__ == "__main__":
    temp_input = torch.randint(0, 256, (1280, 720), dtype=torch.float32).cuda()
    for i in range(1000):
        temp_out = lin_log(temp_input, threshold=20)
