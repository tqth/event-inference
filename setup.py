from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Predict event from photo album using CLIP + RAM + Attention"

setup(
    name='event_inference',
    version='0.1.0',
    description='Predict event from photo album using CLIP + RAM + Attention',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tqth/event-inference',
    author='Quoc Thinh',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'numpy',
        'Pillow',
        'scikit-learn',
        'tensorflow',
        'git+https://github.com/openai/CLIP.git',
        'git+https://github.com/tqth/recognize-anything.git',
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
