import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append("./src")

from src.full_model import PatchedTimeSeriesDecoder, TimesFMConfig

ROOT = Path("model_states")
all_files = set(ROOT.rglob("*.npy"))


def route(key: str) -> torch.Tensor | None:
    path = ROOT
    parts = key.split(".")[::-1]
    transposed = False
    reshape = False
    custom = []
    match parts.pop():
        case "freq_emb":
            path = path / "freq_emb" / "emb_var.npy"
        case "horizon_ff_layer" | "input_ff_layer":
            path = path / key.split(".", maxsplit=1)[0]
            match parts.pop():
                case "hidden_layer":
                    path /= "hidden_layer"
                case "output_layer":
                    path /= "output_layer"
                case "residual_layer":
                    path /= "residual_layer"
            match parts.pop():
                case "weight":
                    transposed = True
                    path = path / "linear" / "w.npy"
                case "bias":
                    path = path / "bias" / "b.npy"
        case "stacked_transformer":
            parts.pop()
            idx = "x_layers_" + parts.pop()
            path = path / "stacked_transformer_layer" / idx
            match parts.pop():
                case "self_attn":
                    path = path / "self_attention"
                    is_linear = True
                    match parts.pop():
                        case "scaling":
                            is_linear = False
                            path = path / "per_dim_scale" / "per_dim_scale.npy"
                        case "q_proj":
                            transposed = True
                            path /= "query"
                        case "k_proj":
                            transposed = True
                            path /= "key"
                        case "v_proj":
                            transposed = True
                            path /= "value"
                        case "o_proj":
                            path /= "post"
                    if is_linear:
                        reshape = True
                        match parts.pop():
                            case "weight":
                                path /= "w.npy"
                            case "bias":
                                transposed = False
                                path /= "b.npy"
                case "mlp":
                    path = path / "ff_layer"
                    is_norm = False
                    match parts.pop():
                        case "gate_proj":
                            path /= "ffn_layer1"
                        case "down_proj":
                            path /= "ffn_layer2"
                        case "layer_norm":
                            is_norm = True
                            path /= "layer_norm"
                    match parts.pop():
                        case "weight":
                            transposed = not is_norm
                            if is_norm:
                                custom.append(lambda x: x + 1)
                            path = (
                                path / "linear" / "w.npy"
                                if not is_norm
                                else path / "scale.npy"
                            )
                        case "bias":
                            path = (
                                path / "bias" / "b.npy"
                                if not is_norm
                                else path / "bias.npy"
                            )
                case "input_layernorm":
                    path = path / "layer_norm" / "scale.npy"
    if path.exists() and path.is_file():
        all_files.remove(path)
        data: np.ndarray = np.load(path)
        if reshape:
            shape = data.shape
            match len(shape):
                case 2:
                    data = data.reshape(-1)
                case 3:
                    data = data.reshape(shape[0], -1)
        if transposed:
            data = data.T
        for func in custom:
            data = func(data)
        return torch.tensor(data)
    return None


if __name__ == "__main__":
    config = TimesFMConfig(
        quantiles=list(np.arange(1, 10) / 10), use_positional_embedding=True
    )
    model = PatchedTimeSeriesDecoder(config)

    state_dict = model.state_dict()

    for key in state_dict.keys():
        params = route(key)
        if params is None:
            print(f"Could not find: {key}")
        elif params.shape != state_dict[key].shape:
            print(
                f"Shape mismatch: {key}, expected: {state_dict[key].shape}, got: {params.shape}"
            )
        else:
            state_dict[key].data.copy_(params)
            assert torch.equal(state_dict[key], params), f"Failed to copy: {key}"

    if len(all_files) > 0:
        print("Unused files:")
        for path in all_files:
            print(path)

    torch.save(state_dict, "checkpoint/timesfm-1.0-200m.pth")
