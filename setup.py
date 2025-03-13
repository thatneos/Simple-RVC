#!/usr/bin/env python
from setuptools import setup, find_packages
import sys

# Ensure a supported Python version (3.6+)
if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")

# Attempt to read the long description from README.md, if available
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="srvc",  # Package name is now srvc
    version="0.1.0",
    author="Thatneos",
    author_email="",
    description="A simple RVC Inference Python wrapper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thatneos/Simple-RVC/tree/main",
    packages=find_packages(),
    install_requires=[
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
        "tensorboardX",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
