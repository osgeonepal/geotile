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
gt.generate_tiles(output_folder=r'/path/to/output/folder', prefix='_img')
```

By default, the above function will create the tiles having square shape of `256` and overlapping of `128`. But you can overwrite the shape as your requirements,

```python
# to generate the tiles having no overlapping
gt.generate_tiles(output_folder=r'/path/to/output/folder', tile_x=512, tile_y=512, stride_x=512, stride_y=512)
```

Also, if you are working with multi-band dataset, it will help you to create the tiles with only specific bands as well,

```python
# only band 3,2,1 will be included in the output tiles
gt.generate_tiles(output_folder=r'/path/to/output/folder', bands=[3,2,1], tile_x=512, tile_y=512, stride_x=512, stride_y=512)
```

If you don't want to save tiles on the disk, you can get the tiles as numpy array as well,

```python
gt.generate_tiles(save_tiles=False)

# get the tiles as numpy array; The shape of the array will be (num_tiles, tile_x, tile_y, bands)
gt.save_numpy(output_folder=r'/path/to/output/folder')
```

### Preprocess tiles before saving

If you want to preprocess (such as normalization, suffling, etc.) the tiles before saving, you can do it as below,

```python
# generate the tiles
gt.generate_tiles(save_tiles=False)

# suffle tiles
gt.suffel_tiles()

# normalize tiles
gt.normalize_tiles()

# convert nan to zero
gt.convert_nan_to_zero()

# save the tiles in numpy format; shape of the array will be (num_tiles, tile_x, tile_y, bands)
gt.save_numpy(output_folder=r'/path/to/output/folder')

# save the tiles in tif format
gt.save_tiles(output_folder=r'/path/to/output/folder', prefix='tile_')
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

> **Note:** The above function will generate the raster mask from shapefile. Which is similar to [extract by mask](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/extract-by-mask.htm) function in ArcGIS. If you want to make the mask raster as binary, you have to use `gt.rasterization()` function.

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
