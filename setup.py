from setuptools import setup, find_packages

setup(
    name='RetroMAE',
    version='0.0.1',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch>=1.6.0',
        'transformers==4.34.0',
        'datasets==1.18.3',
        'faiss-gpu>=1.6'
    ],
)
