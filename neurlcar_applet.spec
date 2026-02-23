# -*- mode: python ; coding: utf-8 -*-
# to create a new exe:
# pyinstaller neurlcar_applet.spec --noconfirm --clean

import os
from pathlib import Path
import carball

# In .spec execution, __file__ may be undefined; cwd is reliable.
ROOT = Path(os.getcwd()).resolve()

CARBALL_DIR = Path(carball.__file__).resolve().parent  # .../site-packages/carball

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(CARBALL_DIR), "carball"),
        (str(ROOT / "rlutilities"), "rlutilities"),
        (str(ROOT / "replay_processing"), "replay_processing"),
        (str(ROOT / "models"), "models"),
    ],
    hiddenimports=[
        "boxcars_py",
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "google.protobuf",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torch._C", "torchvision", "torchaudio"],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="neurlcar_applet",
    console=True,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="neurlcar_applet",
)
