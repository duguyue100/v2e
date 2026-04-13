import re


def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Add typing imports
    if "from typing import" not in content:
        content = re.sub(
            r"import numpy as np\n",
            r"import numpy as np\nfrom typing import Optional, List, Dict, Tuple, Any, Callable, Union\n",
            content,
        )

    # 1. scidvs_dvdt
    content = re.sub(
        r"def scidvs_dvdt\(self, v, tau=None\):",
        r"def scidvs_dvdt(self, v: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:",
        content,
    )

    # 2. __init__ parameters
    content = re.sub(
        r"output_folder: str = None,", r"output_folder: Optional[str] = None,", content
    )
    content = re.sub(
        r"show_dvs_model_state: str = None,",
        r"show_dvs_model_state: Optional[str] = None,",
        content,
    )
    content = re.sub(
        r"output_width: int = None,", r"output_width: Optional[int] = None,", content
    )
    content = re.sub(
        r"output_height: int = None,", r"output_height: Optional[int] = None,", content
    )
    content = re.sub(
        r"cs_lambda_pixels: float = None,",
        r"cs_lambda_pixels: Optional[float] = None,",
        content,
    )
    content = re.sub(
        r"cs_tau_p_ms: float = None,", r"cs_tau_p_ms: Optional[float] = None,", content
    )
    content = re.sub(
        r"record_single_pixel_states=None,",
        r"record_single_pixel_states: Optional[Tuple[int, int]] = None,",
        content,
    )
    content = re.sub(
        r"label_signal_noise=False", r"label_signal_noise: bool = False", content
    )
    content = re.sub(
        r"label_signal_noise: bool = False\n    \):",
        r"label_signal_noise: bool = False\n    ) -> None:",
        content,
    )

    # 3. Lists and Dicts in __init__
    content = re.sub(
        r"self\.dont_show_list = \[\]", r"self.dont_show_list: List[str] = []", content
    )
    content = re.sub(
        r"self\.show_list = \[\]", r"self.show_list: List[str] = []", content
    )
    content = re.sub(
        r"self\.photoreceptor_noise_samples = \[\]",
        r"self.photoreceptor_noise_samples: List[float] = []",
        content,
    )
    content = re.sub(
        r"self\.video_writers: dict\[str, video_writer\] = \{\}",
        r"self.video_writers: Dict[str, Any] = {}",
        content,
    )
    content = re.sub(
        r"self\.cs_steps_taken = \[\]", r"self.cs_steps_taken: List[int] = []", content
    )
    content = re.sub(
        r"self\.show_norms = \{\}",
        r"self.show_norms: Dict[str, Tuple[float, float]] = {}",
        content,
    )

    # single_pixel_states type
    content = re.sub(
        r"self\.single_pixel_states=None",
        r"self.single_pixel_states: Optional[Dict[str, Any]] = None",
        content,
    )

    # 4. Method signatures
    content = re.sub(r"def cleanup\(self\):", r"def cleanup(self) -> None:", content)
    content = re.sub(
        r"def save_recorded_single_pixel_states\(self\):",
        r"def save_recorded_single_pixel_states(self) -> None:",
        content,
    )
    content = re.sub(
        r"def _init\(self, first_frame_linear\):",
        r"def _init(self, first_frame_linear: np.ndarray) -> None:",
        content,
    )
    content = re.sub(
        r"def set_dvs_params\(self, model: str\):",
        r"def set_dvs_params(self, model: str) -> None:",
        content,
    )
    content = re.sub(r"def reset\(self\):", r"def reset(self) -> None:", content)
    content = re.sub(
        r"def _show\(self, inp: torch\.Tensor, name: str\):",
        r"def _show(self, inp: torch.Tensor, name: str) -> None:",
        content,
    )
    content = re.sub(
        r"def generate_events\(self, new_frame, t_frame\):",
        r"def generate_events(self, new_frame: np.ndarray, t_frame: float) -> Optional[np.ndarray]:",
        content,
    )
    content = re.sub(
        r"def get_event_list_from_coords\(self, pos_event_xy, neg_event_xy, ts\):",
        r"def get_event_list_from_coords(self, pos_event_xy: Tuple[torch.Tensor, ...], neg_event_xy: Tuple[torch.Tensor, ...], ts: torch.Tensor) -> Optional[torch.Tensor]:",
        content,
    )
    content = re.sub(
        r"def _update_csdvs\(self, delta_time\):",
        r"def _update_csdvs(self, delta_time: float) -> None:",
        content,
    )

    # 5. Fix numpy typing and clamping
    content = re.sub(r"np\.ndarray \| None", r"Optional[np.ndarray]", content)
    content = re.sub(
        r"self\.pos_thres = torch\.clamp\(self\.pos_thres, min=0\.01\)",
        r"self.pos_thres = torch.clamp(self.pos_thres, min=0.01) # type: ignore",
        content,
    )
    content = re.sub(
        r"self\.neg_thres = torch\.clamp\(self\.neg_thres, min=0\.01\)",
        r"self.neg_thres = torch.clamp(self.neg_thres, min=0.01) # type: ignore",
        content,
    )

    # 6. missing type args for generic type ndarray -> replace `np.ndarray` with `Any` where mypy complains or ignore it
    content = re.sub(r": np\.ndarray", r": Any", content)
    content = re.sub(r"Optional\[np\.ndarray\]", r"Optional[Any]", content)

    # Optional[Any] replacements that were np.ndarray:
    content = re.sub(
        r"self\.new_frame: Optional\[Any\] = None",
        r"self.new_frame: Optional[Any] = None",
        content,
    )

    with open(filepath, "w") as f:
        f.write(content)


fix_file("v2ecore/emulator.py")
