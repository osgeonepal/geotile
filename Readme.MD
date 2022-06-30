# GeoTiler

GeoTile is an open-source python library for creating and manipulating the tiles of the raster dataset. The package will be very useful for managing the raster tiles which can be used for deep learning traing dataset.

## Full documentation

The complete documentation of this package is available here: https://geotile.readthedocs.io/en/latest/

## Installation

If all of the dependencies ([rasterio](https://rasterio.readthedocs.io/en/latest/) and [geopandas](https://geopandas.org/en/stable/index.html)) are installed already, the library can be installed with pip as,

```shell
pip install geotile
```

## Some basic examples

Please check the complete documentation here: https://geotile.readthedocs.io/en/latest/

```shell
from geotile import GeoTile
gt = GeoTile(r"path/to/raster/data.tif")

# to generate the tiles of raster
gt.generate_raster_tiles(r'/path/to/output/folder', tile_x=512, tile_y=512, stride_x=0, stride_y=0)

# to generate the raster mask from shapefile
gt.generate_raster_mask_from_shapefile('/path/to/shapefile.shp', '/path/to/output/file.tif')
```

### TO DO

- Vectorize raster data
