from setuptools import setup, find_packages

try:
    import torch
except ImportError:
    print('PyTorch not found! please install torch/torchvision before proceeding to install the foresight package.')
    exit(1)

setup(
    name="epsinas",
    version="0.1.0",
    author="Ekaterina Gracheva",
    author_email="gracheva.ekaterina@nims.go.jp",  # Replace with the actual email
    description="A Python package for epsinas zero-cost NAS proxy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/egracheva/epsinas",
    download_url="https://github.com/egracheva/epsinas",
    python_requires='>=3.6.0',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'h5py>=2.10.0',
        'jupyter>=1.0.0',
        'matplotlib>=3.2.1',
        'nas-bench-201==2.0',
        'numpy>=1.18.4',
        'prettytable>=2.0.0',
        'pytorch-ignite>=0.3.0',
        'pytorchcv>=0.0.58',
        'scikit-learn>=0.23.2',
        'scipy>=1.4.1',
        'tqdm>=4.46.0',
        'seaborn'

      ],
    entry_points={
        "console_scripts": [
            "epsinas=epsinas.cli:main",  # Adjust based on your package structure
        ]
    },
)
