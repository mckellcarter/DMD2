from setuptools import setup, find_packages
setup(
    name="DMD2",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'diffviews.adapters': [
            'dmd2-imagenet-64 = visualizer.adapters.dmd2_imagenet:DMD2ImageNetAdapter',
        ],
    },
)