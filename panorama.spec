# -*- mode: python ; coding: utf-8 -*-

import os
import sys

path = os.path.normpath(os.path.abspath('./src'))
if path not in sys.path:
  sys.path.insert(0, path)

from kivymd import hooks_path as kivymd_hooks_path

block_cipher = None

a = Analysis(['panorama.py'],
             pathex=['D:\\Python\\ThermalImage'],
             binaries=[],
             datas=[
                 ('src\\', 'src\\'),
                 ('data\\fonts\\', 'data\\fonts\\'),
                 ('data\\cmap\\', 'data\\cmap\\'),
             ],
             hiddenimports=[
                 'scipy.special.cython_special',
                 'skimage.io',
                 'src.flir.extract',
                 'src.tools.ivimages',
                 'src.tools.stitcher',
                 'kivymd.stiffscroll',
             ],
             hookspath=[kivymd_hooks_path],
             runtime_hooks=['D:\\Python\\ThermalImage\\runtime_hook.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts, [],
          exclude_binaries=True,
          name='ThermalPanorama',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ThermalPanorama')
