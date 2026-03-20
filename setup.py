from setuptools import setup, find_packages

setup(
    name="custos",
    version="0.1.0",
    description="Adaptive Immune Defense for Multi-Agent LLM Networks",
    author="Yugesh Sappidi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.30.0",
        "boto3>=1.34.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "sentence-transformers>=3.0.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0"],
    },
)
