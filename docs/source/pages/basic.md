# Getting Started With Basic Functions

Now it is time to initialize the library and work with your raster data. First of all, you need to import the library and initialize the library your raster file as below,

```python
from geotile import GeoTile
gt = GeoTile('/path/to/raster/file.tif')
```

After initializing the `GeoTile` class, the library will be able to read the file. Now you have access to use the `GeoTile` methods.

## Create the raster tiles

Lets create the tiles of our raster data,

```python
gt.generate_raster_tiles(r'/path/to/output/folder')
```

By default, the above function will create the tiles having square shape of `256` and overlapping of `128`. But you can overwrite the shape as your requirements,

```python
gt.generate_raster_tiles(r'/path/to/output/folder', tile_x=512, tile_y=512, stride_x=0, stride_y=0)
```

Also, if you are working with multi-band dataset, it will help you to create the tiles with only specific bands as well,

```python
gt.generate_raster_tiles(r'/path/to/output/folder', bands=[3,2,1], tile_x=512, tile_y=512, stride_x=0, stride_y=0)
```

## Merge tiles

To create the combined image with tiled images, the `mosaic_rasters` function will be useful,

```python
from geoTile import mosaic_rasters
mosaic_rasters('/path/to/input/folder', '/path/to/output/file.tif')
```

**Note: ** This function will be valid only for the georeferenced rasters.

## Generate mask from shapefile

To generate the raster mask from shapefile, you need to initialize the GeoTile class first,

```python
from geotile import GeoTile
gt = GeoTile('/path/to/raster/file.tif')

# generate shapefile mask
gt.generate_raster_mask_from_shapefile('/path/to/shapefile.shp', '/path/to/output/file.tif')
```

The output raster will have simillar metadata as the input raster.

## More functions

**TO DO**
