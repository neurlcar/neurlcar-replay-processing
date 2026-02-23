import numpy as np

# before analysis is done, replay is trimmed of non-play frames.
# Incorporate these frames back in so that it displays in the replay seamlessly
def pad_preds_and_goalchanges(preds, goalchanges, prelength, game):
    kept_frames = game.frames.index.values - 1
    pad_husk = np.zeros(prelength)

    preds_padded = pad_husk + 0.5
    preds_padded[kept_frames] = preds.squeeze()

    padded_goalchanges = []
    for gc in goalchanges:
        pad_gc = pad_husk
        pad_gc[kept_frames] = gc.squeeze()
        padded_goalchanges.append(pad_gc.copy())

    padded_preds_array = np.vstack([preds_padded, np.vstack(padded_goalchanges)])

    return padded_preds_array

def pad_preds(preds, prelength, game):
    kept_frames = game.frames.index.values - 1
    pad_husk = np.zeros(prelength)
    preds_padded = pad_husk + 0.5
    preds_padded[kept_frames] = preds.squeeze()
    return preds_padded