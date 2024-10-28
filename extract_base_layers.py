import sys

import numpy as np
import torch
from absl import flags

sys.path.append("./src")

from src.base_model import PatchedTimeSeriesEncoder, TimesFMConfig

FLAGS = flags.FLAGS

_MODEL_PATH = flags.DEFINE_string(
    "model_path", "./checkpoint/timesfm-1.0-200m.pth", "The path to model."
)
_SAVE_PATH = flags.DEFINE_string(
    "save_path", "./checkpoint/timesfm-1.0-200m-base.pth", "The path to save model."
)

QUANTILES = list(np.arange(1, 10) / 10.0)
EPS = 1e-7


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    model_path = _MODEL_PATH.value

    state_dict = torch.load(model_path, weights_only=True)

    base_config = TimesFMConfig()
    base_model = PatchedTimeSeriesEncoder(base_config)
    base_state_dict = base_model.state_dict()
    for key in base_state_dict.keys():
        if key not in state_dict:
            print(f"Could not find: {key}")
        elif base_state_dict[key].shape != state_dict[key].shape:
            print(
                f"Shape mismatch: {key}, expected: {base_state_dict[key].shape}, got: {state_dict[key].shape}"
            )
        else:
            base_state_dict[key].data.copy_(state_dict[key])
            assert torch.equal(
                base_state_dict[key], state_dict[key]
            ), f"Failed to copy: {key}"
    base_model.load_state_dict(base_state_dict)
    torch.save(base_model.state_dict(), _SAVE_PATH.value)
