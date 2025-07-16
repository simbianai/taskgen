from setuptools import setup, find_packages

setup(
    name="taskgen",
    version="3.3.10",
    packages=find_packages(),
    install_requires=[
        "openai>=1.59.6",
        "langchain",
        "dill>=0.3.9",
        "termcolor>=3.1.0",
        "requests",
        "pypdf~=5.8.0",
        "python-docx",
        "pandas",
        "chromadb",
        "xlrd",
        "chromadb>=1.0.15",
        "asyncio",
    ],
)
