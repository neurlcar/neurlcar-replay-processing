# neuRLcar applet (PyInstaller EXE)

This folder builds a standalone Windows EXE that:
- takes a Rocket League `.replay` file
- runs my replay → features → ONNX inference pipeline
- writes predictions to an output `.csv`

The EXE is meant to be launched by a BakkesMod plugin, but you should always test it from the command line first.

---

## What you get

After a successful build, you’ll have:

- `dist/neurlcar_applet/neurlcar_applet.exe`
- `dist/neurlcar_applet/_internal/` (PyInstaller one-folder bundle)
- bundled `models/` + `replay_processing/` + `rlutilities/`

The EXE entrypoint is `run.py`. Usage:

```bash
neurlcar_applet.exe <replay_filepath> <output_csv_filepath>
