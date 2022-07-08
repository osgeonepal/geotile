import os
from typing import Dict

from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))

about = dict()

with open(os.path.join(HERE, "geotile", "__version__.py"), "r") as f:
    exec(f.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="geotile",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description="Package for working with geographic raster tiles",
    py_modules=["geotile"],
    license="MIT License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamtekson/geotile",
    packages=["geotile"],
    keywords=[
        "geotile",
        "geotiling",
        "geoTiler",
        "geospatial",
        "geospatial data",
        "geospatial raster tiles",
        "raster tiles",
        "raster",
        "tiles",
        "tile",
        "tiling python",
        "python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "gdal",
        "numpy",
        "geopandas",
        "rasterio",
    ],
    extras_require={"dev": [
        "pytest",
        "black",
        "flake8",
        "sphinx>=1.7",
        "pydata-sphinx-theme"
    ]},
    python_requires=">=3.6",
)
