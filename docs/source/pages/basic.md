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
gt.generate_tiles(r'/path/to/output/folder')
```

By default, the above function will create the tiles having square shape of `256` and overlapping of `128`. But you can overwrite the shape as your requirements,

```python
# to generate the tiles having no overlapping
gt.generate_tiles(r'/path/to/output/folder', tile_x=512, tile_y=512, stride_x=512, stride_y=512)
```

Also, if you are working with multi-band dataset, it will help you to create the tiles with only specific bands as well,

```python
# only band 3,2,1 will be included in the output tiles
gt.generate_tiles(r'/path/to/output/folder', bands=[3,2,1], tile_x=512, tile_y=512, stride_x=512, stride_y=512)
```

## Merge tiles

To create the combined image with tiled images, the `mosaic` function will be useful,

```python
from geotile import mosaic
mosaic('/path/to/input/folder', '/path/to/output/file.tif')
```

**Note: ** This function will be valid only for the georeferenced rasters.

## Generate mask from shapefile

To generate the raster mask from shapefile, you need to initialize the GeoTile class first,

```python
from geotile import GeoTile
gt = GeoTile('/path/to/raster/file.tif')

# generate shapefile mask
gt.mask('/path/to/shapefile.shp', '/path/to/output/file.tif')
```

The output raster will have simillar metadata as the input raster.

## Rasterization of shapefile

To create the raster surface from the shapefile, you can do following thin. If value_col is None, the rasterization will be binary otherwise the rasterization will be the based on value of the column

```python
gt.rasterization(input_vector='path/to/shp.shp', out_path='path/to/output.tif' value_col=None)
```

> The generated raster will have same meta information as raster used in `GeoTile` class.

## Close the raster file

```python
gt.close()
```

## More functions

Some of the other functionalities of this packages are as below,

```python
# reprojection of raster
gt.reprojection(out_path='path/to/output.tif', out_crs='EPSG:4326', resampling_method='nearest')

#resampling of raster
gt.resample('/path/to/output/file.tif', upscale_factor=2, resampling_method='bilinear')
```

