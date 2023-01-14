[![Downloads](https://pepy.tech/badge/geotile)](https://pepy.tech/project/geotile)
[![PyPI version](https://badge.fury.io/py/geotile.svg)](https://pypi.org/project/geotile/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/iamtekson/geotile/blob/master/LICENSE)

# GeoTile

GeoTile is an open-source python library for creating and manipulating the tiles of the raster dataset. The package will be very useful for managing the raster tiles which can be used for deep learning traing dataset.

## Full documentation

The complete documentation of this package is available here: https://geotile.readthedocs.io/en/latest/

## Installation

The easy installation of `geotile` is by using `conda` environment,

```shell
conda install -c conda-forge geotile
```

If you want to install it for `pip`, check the documentation here: https://geotile.readthedocs.io/en/latest/pages/install.html

## Some basic examples

Please check the complete documentation here: https://geotile.readthedocs.io/en/latest/

```shell
from geotile import GeoTile
gt = GeoTile(r"path/to/raster/data.tif")

# to generate the tiles of raster
gt.generate_tiles(r'/path/to/output/folder', tile_x=256, tile_y=256, stride_x=256, stride_y=256)

# to generate the tiles of selected bands only
gt.generate_tiles(r'/path/to/output/folder', bands=[4, 3, 2], tile_x=256, tile_y=256, stride_x=256, stride_y=256)

# to merge the tiles
from geotile import mosaic
mosaic('/path/to/input/folder/tiles', output_file='path/to/output/file.tif')

# to generate the raster mask from shapefile
gt.mask('/path/to/shapefile.shp', '/path/to/output/file.tif')

# to rasterize the shapefile based on column value,
gt.rasterization(input_vector='path/to/shp.shp', out_path='path/to/output.tif' value_col="value_col")

# to close the file
gt.close()
```

