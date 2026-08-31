# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

desktop_dir = Path(SPEC).resolve().parent
root = desktop_dir.parent
datas = [
    (str(root / "web"), "web"),
    (str(root / "core" / "migrations"), "core/migrations"),
]

a = Analysis(
    [str(root / "app.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "datasets", "matplotlib", "numpy", "pandas", "pyarrow", "ragas",
        "scipy", "sentence_transformers", "sklearn", "torch",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="knowledge-garden-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
