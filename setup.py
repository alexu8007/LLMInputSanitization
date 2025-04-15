from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-input-sanitizer",
    version="0.2.0",
    author="Alex",
    author_email="alexu8007@gmail.com",
    description="A package for sanitizing and securing user inputs to LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexu8007/llm-input-sanitizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "nltk>=3.6.0",
        "better-profanity>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8", "better-profanity>=0.7.0"],
    },
)