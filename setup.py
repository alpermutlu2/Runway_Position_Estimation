
from setuptools import setup, find_packages

setup(
    name='Runway_Position_Estimation',
    version='0.1.0',
    description='Probabilistic Monocular Depth Estimation with SAM-based Masking',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
        'torchvision',
        'opencv-python',
        'numpy'
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'train-depth=Runway_Position_Estimation.train_cli:main',
            'visualize-depth=Runway_Position_Estimation.visualize_depth:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
