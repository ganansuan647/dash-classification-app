# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['classification_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('dataset/dataset2_RC_Pier_Column_Failure_Mode.xlsx', 'dataset'),
        ('dataset/dataset6_Bridge_Post_Earthquake_Damage_State.xlsx', 'dataset'),
        ('assets/TBS-logo.png','assets'),
        ('assets/base-styles.css','assets'),
        ('assets/custom-styles.css','assets')
    ],
    hiddenimports=['sklearn.datasets.data','sklearn.datasets.descr'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='classification_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    icon="assets/icon.ico",
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
