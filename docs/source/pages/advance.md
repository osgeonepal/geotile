# Advance Use Cases

The main motivation of this library is to create the easy pipeline to handle the geospatial raster dataset in machine learning and deep learning projects. The library is able to create the required label imagery from the vector data as well as the tiles of the raster data. The library is also able to create the tiles of the predicted data and merge them into single tif file. The below are the some key features/use cases of the library,

## 1. Generate masks from vector data

The library is able to create the labels or masks from the vector data. The vector data can be in any format such as geojson, shapefile, etc. The library is able to create the masks of the vector data in the raster format. The below is the example of creating the mask from the shapefile,

```python
from geotile import GeoTile
gt = GeoTile('/path/to/raster/file.tif')
gt.rasterization('/path/to/vector/file.shp', output_folder='/path/to/output/file.tif')
```

If you didn't pass the `value_col` parameter, the library will create the mask of the vector data with binary values, i.e. 0 and 1. If you want to create the mask with the specific values, you can pass the `value_col` parameter as below,

```python
from geotile import GeoTile
gt = GeoTile('/path/to/raster/file.tif')
gt.rasterization('/path/to/vector/file.shp', out_path='/path/to/output/file.tif', value_col='class')
```

Using the rasterization function, you don't need to worry about the metadata and extend of the output raster. The output raster will have the same metadata and extend as the input raster assigned in `gt` class.

> **Note**: Many people confused with the `gt.mask()` function. The `gt.mask()` function is used to clip out the raster data by vector extend which is similar to [extract by mask](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/extract-by-mask.htm) in ArcGIS. The `gt.rasterization()` function is used to create the mask of the vector data.

## 2. Generate tiles from both imagery and masks

The library is able to create the tiles from both images and masks. The below is the example of creating the tiles from the image,

```python
from geotile import GeoTile

# create the tiles of the raster imagery
gt_img = GeoTile('/path/to/raster/file.tif')
gt_img.generate_tiles(output_folder='/path/to/output/folder')

# create the tiles of the raster mask
gt_mask = GeoTile('/path/to/raster/file.tif')
gt_mask.generate_tiles(output_folder='/path/to/output/folder')
```

If you don't want to save the tiles, you have to pass the `save_tiles=False` parameter in the `generate_tiles()` function. In this case, the raster values will be stored in the `gt_img.tile_data` and `gt_mask.tile_data` variables.
The below is the example of creating the tiles without saving them,

```python
from geotile import GeoTile

# create the tiles of the raster imagery
gt_img = GeoTile('/path/to/raster/file.tif')
gt_img.generate_tiles(save_tiles=False)

# create the tiles of the raster mask
gt_mask = GeoTile('/path/to/raster/file.tif')
gt_mask.generate_tiles(save_tiles=False)
```

### Preprocess tiles before saving

If you want to preprocess (such as normalization, removing nan, etc.) the tiles before saving, you can do it as below,

```python
# generate the tiles
gt_img.generate_tiles(save_tiles=False)
gt_img.convert_nan_to_zero()
gt_img.normalize_tiles()

# save the tiles in numpy format; shape of the array will be (num_tiles, tile_x, tile_y, bands)
gt_img.save_numpy(output_folder=r'/path/to/output/folder')

# save the tiles in tif format
gt_img.save_tiles(output_folder=r'/path/to/output/folder', prefix='tile_')
```

If your main goal is to train the deep learning model, you can only save the tiles as a numpy array and ignore saving tiles as a tif file. The numpy array will be more efficient and faster to load in the deep learning model.

## 3. Model prediction on tiles and merge them into single tif file

You can now use your deep learning model to predict the tiles. The below is the example of predicting the tiles,

```python
from geotile import GeoTile
from geotile import mosaic

# create the tiles of the raster imagery
gt_img = GeoTile('/path/to/raster/file.tif')
gt_img.generate_tiles(save_tiles=False)

# predict the tiles
pred = model.predict(gt_img.tile_data)

# you can do the post processing such as thresholding, changing datatype etc on the predicted tiles as well. After that you can assign the predicted tile values to the gt class and save georeferenced tile as below,
gt_img.tile_data = pred
gt_img.save_tiles(output_folder='/path/to/output/folder', prefix='pred_')

# merge the predicted tiles into single tif file
mosaic('/path/to/output/folder', '/path/to/output/file.tif')
```

## Discussion on other advance operations

Some of the advance operations were discussing in following issues, please check them out for more details,

1. [Issue #18: Merge/mosaic predicted tiles into single tif file](https://github.com/iamtekson/geotile/issues/18)
2. [Issue #15: Data augmentation technique](https://github.com/iamtekson/geotile/issues/18)
