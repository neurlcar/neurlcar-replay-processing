import os
import numpy as np
import onnxruntime as ort

from replay_processing.get_base_dir import get_base_dir
from replay_processing.graph_streamer import xyz_to_graph
from replay_processing.invert_frames import invert_x


# ------------------------------------------------------------
# Helpers (NumPy-only)
# ------------------------------------------------------------
def _prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable enough for typical logit ranges; if you ever see overflow, we can add a branchy stable sigmoid
    return 1.0 / (1.0 + np.exp(-x))


def _compose_eval(p_wsn: np.ndarray, p_imm: np.ndarray) -> np.ndarray:
    """
    p_wsn ∈ [0,1]   (0.5 neutral)
    p_imm ∈ [0,1]
    returns final_eval ∈ [0,1]

    Current composition:
      w = 2*p_wsn - 1     # in [-1,1]
      e(g,w) = sign(w) * (g + |w| - g*|w|)
      final_eval = (e + 1)/2  ∈ [0,1]
    """
    w = 2.0 * p_wsn - 1.0
    aw = np.abs(w)
    s = np.sign(w)
    e_signed = s * (p_imm + aw - p_imm * aw)
    return 0.5 * (e_signed + 1.0)


def _infer_gamemode(numcols: int) -> str:
    if numcols == 144:
        return "duel"
    elif numcols == 208:
        return "doubles"
    elif numcols == 272:
        return "standard"
    raise ValueError(f"Unexpected x width {numcols}")


# ------------------------------------------------------------
# Main function (kept name/signature)
# ------------------------------------------------------------
def y_wsn_imm_from_x(x: np.ndarray, model_path=None, use_cuda=True):
    """
    Runs one replay (x array) through your ONNX TokenAttnDualHead model and
    computes the same symmetry-enforced outputs you used in training.

    Args:
        x : np.ndarray of shape [B, D]
        model_path : path to .onnx file, defaults to CombinedModel_hiscore.onnx
        use_cuda : if True and CUDA available, uses CUDAExecutionProvider

    Returns:
        y_preds   (np.ndarray) : composed final_eval, shape [B]
        wsn_preds (np.ndarray) : symmetry-enforced who-scores-next, shape [B]
        imm_preds (np.ndarray) : symmetry-enforced imminence, shape [B]
    """
    if model_path is None:
        model_path = os.path.join(get_base_dir(), "models", "CombinedModel_hiscore.onnx")

    # ONNX Runtime provider selection
    avail = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers)

    # --------------------------
    # Ensure numpy float32
    # --------------------------
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [B,D], got shape {x.shape}")
    x = x.astype(np.float32, copy=False)

    gamemode = _infer_gamemode(x.shape[1])

    # Build inverted version (NumPy invert_x)
    Tx = invert_x(x, gamemode=gamemode)

    # Convert both to graph tokens (NumPy xyz_to_graph)
    tokens_x, token_mask = xyz_to_graph(x, gamemode=gamemode)
    tokens_Tx, _ = xyz_to_graph(Tx, gamemode=gamemode)

    # --------------------------
    # Prepare ONNX inputs
    # --------------------------
    def _get_predictions(tokens_list):
        ort_inputs = {}
        for i, t in enumerate(tokens_list):
            # be defensive on dtype
            ort_inputs[f"token_{i}"] = np.asarray(t, dtype=np.float32)
        ort_inputs["token_mask"] = np.asarray(token_mask, dtype=np.bool_)

        p_wsn, p_imm = session.run(["p_wsn", "p_imm"], ort_inputs)
        p_wsn = np.asarray(p_wsn, dtype=np.float32).reshape(-1)
        p_imm = np.asarray(p_imm, dtype=np.float32).reshape(-1)
        return p_wsn, p_imm

    # Run both x and inverted Tx
    p_wsn_x, p_imm_x = _get_predictions(tokens_x)
    p_wsn_Tx, p_imm_Tx = _get_predictions(tokens_Tx)

    """
    Enforce:
      g_wsn(Tx)  = 1 - g_wsn(x)
      g_imm(Tx)  =     g_imm(x)

    Implementation:
      Let model output probabilities p(x), p(Tx).
      h(x)   = logit(p(x))
      h(Tx)  = logit(p(Tx))

      s_wsn(x)  = h_wsn(x)  - h_wsn(Tx)
      s_imm(x)  = h_imm(x)  + h_imm(Tx)

      g_wsn(x)  = σ(s_wsn(x))
      g_imm(x)  = σ(s_imm(x))
    """
    h_wsn_x = _prob_to_logit(p_wsn_x)
    h_wsn_Tx = _prob_to_logit(p_wsn_Tx)
    h_imm_x = _prob_to_logit(p_imm_x)
    h_imm_Tx = _prob_to_logit(p_imm_Tx)

    s_wsn = h_wsn_x - h_wsn_Tx
    s_imm = h_imm_x + h_imm_Tx

    wsn_preds = _sigmoid(s_wsn).astype(np.float32, copy=False)
    imm_preds = _sigmoid(s_imm).astype(np.float32, copy=False)

    y_preds = _compose_eval(wsn_preds, imm_preds).astype(np.float32, copy=False)

    return y_preds, wsn_preds, imm_preds
