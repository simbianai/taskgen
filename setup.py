from setuptools import setup, find_packages

setup(
    name="taskgen",
    version="3.4.3",
    packages=find_packages(),
    install_requires=[
        "openai>=1.59.6",
        "langchain",
        "dill>=0.3.9",
        "termcolor>=3.1.0",
        "requests",
        "python-docx",
        "pandas",
        "xlrd",
        "asyncio",
    ],
)
