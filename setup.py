import os

from setuptools import find_packages, setup

# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


requirements = [
    "gymnasium<1.0.0",
    "matplotlib",
    "numpy",
    "pandas",
    "pygame",
    "shapely",
    "svgpath2mpl",
]

setup(
    name="MetaLore",
    version="2.0.1",
    author="Elif Ebru Ohri",
    description="MetaLore: DRL simulation environment for joint communication and computation resoruce allocation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elifohri/MetaLore-simulator",
    packages=find_packages(),
    python_requires=">=3.9.0",
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)