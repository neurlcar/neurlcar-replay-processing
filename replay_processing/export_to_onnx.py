import os
import torch
import numpy as np

from graph_streamer import xyz_to_graph
from replay_processing.Modules import TokenAttnDualHead, TokenAttnDualHeadOnnxWrap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "D:/python/replay_processing/models"


def export_model(
    ckpt_name: str,
    onnx_name: str,
    replay_path: str = "tests/npys/3v3.npy",
):
    # --------------------------------------------------
    # 1) Load trained checkpoint into the BASE model
    # --------------------------------------------------
    base = TokenAttnDualHead().to(device)
    ckpt_path = os.path.join(MODELS_DIR, ckpt_name)

    state_dict = torch.load(ckpt_path, map_location=device)
    base.load_state_dict(state_dict)  # keys match (no "base." prefix)
    base.eval()

    # Wrap only after loading
    model = TokenAttnDualHeadOnnxWrap(base).to(device)
    model.eval()

    # --------------------------------------------------
    # 2) Build real example inputs from a replay
    # --------------------------------------------------
    xyz = np.load(replay_path)

    # Drop last 2 cols if those are labels (y,z)
    x = torch.from_numpy(xyz[:, :-2]).float().to(device)

    # Convert to graph tokens
    tokens_list, token_mask = xyz_to_graph(x, gamemode="standard", device=device)

    # Ensure mask dtype/device is sane
    token_mask = token_mask.to(device).bool()

    # --------------------------------------------------
    # 3) Prepare ONNX export inputs
    #    Wrapper forward expects: (*token_0..token_36, token_mask)
    # --------------------------------------------------
    inputs = tuple(tokens_list) + (token_mask,)

    input_names = [f"token_{i}" for i in range(len(tokens_list))] + ["token_mask"]
    output_names = ["p_wsn", "p_imm"]

    # Dynamic batch axis for token tensors and outputs.
    # Do NOT set dynamic axes for token_mask (it's [37] fixed).
    dynamic_axes = {name: {0: "batch_size"} for name in input_names if name != "token_mask"}
    dynamic_axes.update({name: {0: "batch_size"} for name in output_names})

    onnx_path = os.path.join(MODELS_DIR, onnx_name)

    # --------------------------------------------------
    # 4) Export
    # --------------------------------------------------
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    export_model(
        ckpt_name="CombinedModel_hiscore.pk1",
        onnx_name="CombinedModel_hiscore.onnx",
        replay_path="../tests/npys/3v3.npy",
    )
