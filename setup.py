from pathlib import Path

from setuptools import find_packages, setup

README = Path("README.md")
long_description = README.read_text(encoding="utf-8") if README.exists() else "Global Macro Data package"

setup(
    name="global-macro-data",
    version="3.0.0",
    packages=find_packages(),
    package_data={"global_macro_data": ["*.csv", "*.txt"]},
    install_requires=["requests", "pandas", "numpy", "matplotlib", "openpyxl", "pyshp"],
    author="Bojun Geng",
    description="Global Macro Database Python implementation by Bojun Geng",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KMueller-Lab/Global-Macro-Database-Python",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
