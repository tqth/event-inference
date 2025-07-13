from setuptools import setup, find_packages

setup(
    name='event_inference',
    version='0.1.0',
    description='Predict event from photo album using CLIP + RAM + Attention',
    long_description=open('README.md').read(),
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
)
