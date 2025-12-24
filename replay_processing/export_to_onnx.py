import os
import torch
import numpy as np

from replay_processing.get_base_dir import get_base_dir
from replay_processing.Modules import (
    PosVel_convSig_futureball_duel_deep,
    PosVel_convSig_futureball_doubles_deep,
    PosVel_convSig_futureball_standard_deep,
)

MODELS_DIR = "D:/python/replay_processing/models"


def export_model(model_class, ckpt_name, onnx_name, input_dim):
    model = model_class()
    ckpt_path = os.path.join(MODELS_DIR, ckpt_name)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # dummy input: batch_size=2, features=input_dim
    dummy = torch.randn(2, input_dim, dtype=torch.float32)

    onnx_path = os.path.join(MODELS_DIR, onnx_name)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
        opset_version=13,  # 11+ is generally fine; 13 is a good default
    )
    print(f"Exported {onnx_path}")


if __name__ == "__main__":
    export_model(PosVel_convSig_futureball_duel_deep,    "duel.pk1",    "duel.onnx",    144)
    export_model(PosVel_convSig_futureball_doubles_deep, "doubles.pk1", "doubles.onnx", 208)
    export_model(PosVel_convSig_futureball_standard_deep,"standard.pk1","standard.onnx",272)
