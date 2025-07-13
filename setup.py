from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Predict event from photo album using CLIP + RAM + Attention"

setup(
    name="event-inference",
    version="0.1.0",
    description="Predict event from photo album using CLIP + RAM + Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tqth/event-inference",
    author="Quoc Thinh",
    packages=["dataset", "models", "utils"],  # Tìm package trong thư mục gốc
    package_dir={"": "."},  # Định nghĩa gốc package là thư mục hiện tại
    py_modules=["predict_event"],  # Bao gồm module riêng lẻ predict_event.py
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "tqdm>=4.0.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.4.0",
        "clip @ git+https://github.com/openai/CLIP.git",
        "ram @ git+https://github.com/tqth/recognize-anything.git",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)