import os
import sys

path = os.path.normpath(os.path.abspath('./lib'))

if path not in sys.path:
  sys.path.insert(0, path)
