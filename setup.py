from setuptools import setup, find_packages

setup(
    name="cheetahopt",
    version="0.1.0",
    author="Your Name",
    description="Optimization library using hybrid WOA and GWO for MLP tuning.",
    packages=find_packages(),
    install_requires=["pandas", "tensorflow", "scikit-learn"]
)
