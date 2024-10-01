
from setuptools import setup, find_packages

setup(
    name="fraud_detection_simulator",
    version="1.0",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    description="Fraud detection simulator for benchmarking anomaly detection models.",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
)
