# Custom hook to bypass the standard setuptools hook; the app does not use
# setuptools at runtime, so collect nothing.
datas, binaries, hiddenimports = [], [], []
