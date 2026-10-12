# Local AuDrA scoring

`score.py` loads the supplied checkpoint and scores images in `user_images/`.
It keeps AuDrA's inference packages separate from `backend/.venv` and does not
change the deployed Flask app.

Use Python 3.12 and run commands from the repository root. On Linux or WSL
(x86-64), install the CPU wheels from PyTorch's index:

```bash
python3.12 -m venv AuDrA/.venv
AuDrA/.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r AuDrA/requirements-inference.txt
AuDrA/.venv/bin/python AuDrA/score.py
```

On Windows PowerShell (64-bit Python):

```powershell
py -3.12 -m venv AuDrA\.venv
AuDrA\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu -r AuDrA\requirements-inference.txt
AuDrA\.venv\Scripts\python.exe AuDrA\score.py
```

On Apple Silicon macOS 14 or newer, use PyPI's macOS wheels:

```bash
python3.12 -m venv AuDrA/.venv
AuDrA/.venv/bin/python -m pip install -r AuDrA/requirements-inference.txt
AuDrA/.venv/bin/python AuDrA/score.py
```

The last command writes `AuDrA/AuDrA_predictions.csv`, replacing that file if
it already exists. Use `--input-dir` and `--output` to choose different paths.
The score is the model's predicted normalized Judge Response Theory rating for
drawings like those used to train AuDrA; it is not a general-purpose creativity
measure.

The original `AuDrA_run.py` and Linux GPU environment snapshot are retained as supplied.
The tutorial's Conda environment includes training-only packages; this inference
path needs only CPU PyTorch, torchvision, and Pillow. Do not use the same
`AuDrA/.venv` from both Windows Python and WSL Python. The checkpoint is not
tracked in Git, so other contributors need to obtain `AuDrA_trained.ckpt`
separately and place it in this folder. Intel Macs are not covered by the
pinned PyTorch wheels above.

The supplied AuDrA code identifies its license as CC BY-NC-SA 4.0; review its
terms before redistributing or using the model commercially.
