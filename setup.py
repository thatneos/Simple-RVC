[build-system]
requires = [
  "setuptools>=61",
  "wheel"
]
build-backend = "setuptools.build_meta"

[project]
name = "simple-rvc"
version = "0.1.0"
description = "A simple RVC Inference Python wrapper."
readme = "README.md"
requires-python = ">=3.6"
license = { text = "MIT License" }
authors = [
  { name = "Thatneos", email = "" }
]
urls = { Homepage = "https://github.com/thatneos/Simple-RVC/tree/main" }
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent"
]
dependencies = [
  "requests",
  "mega.py",
  "gdown",
  "wget",
  "gradio==4.29.0",
  "audio-separator[gpu]==0.30.1",
  "sox",
  "pedalboard",
  "torch<2.6",
  "torchaudio<2.6",
  "torchvision<0.21",
  "fairseq2",
  "yt-dlp",
  "ffmpeg-python>=0.2.0",
  "praat-parselmouth>=0.4.2",
  "torchcrepe==0.0.23",
  "pyworld==0.3.4",
  "faiss-cpu==1.7.3",
  "numpy==1.23.5",
  "einops",
  "local-attention",
  "tensorboardX"
]

[tool.setuptools]
package-dir = { "" = "srvc" }
include-package-data = true
zip-safe = false

[tool.setuptools.packages.find]
where = ["srvc"]
