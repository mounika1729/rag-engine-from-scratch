from setuptools import setup, find_packages

setup(
    name="rag-from-scratch",
    version="1.0.0",
    description="Production-grade RAG system built from scratch",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
