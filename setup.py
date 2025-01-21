from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="price_optimization_rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning solution for price optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/price-optimization-rl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "python-dateutil>=2.8.2",
        "streamlit>=1.20.0",
        "plotly>=5.10.0",
    ],
    entry_points={
        "console_scripts": [
            "train-pricing=src.train:main",
            "evaluate-pricing=src.evaluate:main",
            "run-experiment=src.run_experiment:main",
            "run-all=run_all:main",
        ],
    },
) 