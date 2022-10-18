# inbuilt libraries
import os
import itertools
import glob
from typing import Optional
import pathlib

import numpy as np

# rasterio library
import rasterio as rio
from rasterio import windows
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.features import rasterize

# geopandas library
import geopandas as gpd


class GeoTile:
    """GeoTile class

            Attributes
            ----------
                path : str, python path
                    Path to the raster file (str)
                ds : rasterio.DatasetReader, object
                    Raster dataset
                meta : dict
                    Raster metadata
                height : int
                    Raster height
                width : int
                    Raster width
                crs : str
                    Raster crs (e.g. 'EPSG:4326'). The will be generated automatically.
                    In case of non-geographic raster, the crs will be None.
                stride_x : int
                    The stride of the x axis (int), default is 128
                stride_y : int
                    The stride of the y axis (int), default is 128
                tile_x : int
                    The size of the tile in x axis (int), default is 256
                tile_y : int
                    The size of the tile in y axis (int), default is 256
    """

    def __init__(self, path):
        """
        Read the raster file

            Parameters
            ----------
                path: the path of the raster file

            Returns
            -------
                None: Read raster and assign metadata of raster to the class
        """
        self._read_raster(path)

    def __del__(self):
        self.ds.close()

    def _read_raster(self, path):
        """Read the raster file

            Parameters
            ----------
                path: the path of the raster file

            Returns
            -------
                None: Read raster and assign metadata of raster to the class
        """
        self.path = path
        self.ds = rio.open(path)
        self.meta = self.ds.meta
        self.height = self.meta['height']
        self.width = self.meta['width']
        self.meta['crs'] = self.ds.crs

    def _calculate_offset(self, stride_x: Optional[int] = None, stride_y: Optional[int] = None) -> tuple:
        """Calculate the offset for the whole dataset

            Parameters
            ----------
                tile_x: the size of the tile in x axis
                tile_y: the size of the tile in y axis
                stride_x: the stride of the x axis
                stride_y: the stride of the y axis

            Returns
            -------
                tuple: (offset_x, offset_y)
        """
        self.stride_x = stride_x
        self.stride_y = stride_y

        # offset x and y values calculation
        X = [x for x in range(0, self.width, stride_x)]
        Y = [y for y in range(0, self.height, stride_y)]
        offsets = list(itertools.product(X, Y))
        return offsets

    def generate_tiles(
            self,
            output_folder: str,
            out_bands: Optional[list] = None,
            image_format: Optional[str] = None,
            dtype: Optional[str] = None,
            tile_x: Optional[int] = 256,
            tile_y: Optional[int] = 256,
            stride_x: Optional[int] = 128,
            stride_y: Optional[int] = 128
    ):
        """
        Save the tiles to the output folder

            Parameters
            ----------
                output_folder : str
                    Path to the output folder
                out_bands : list
                    The bands to save (eg. [3, 2, 1]), if None, the output bands will be same as the input raster bands
                image_format : str
                    The image format (eg. tif), if None, the image format will be the same as the input raster format (eg. tif)
                dtype : str, np.dtype
                    The output dtype (eg. uint8, float32), if None, the dtype will be the same as the input raster
                tile_x: int
                    The size of the tile in x axis, Default value is 256
                tile_y: int
                    The size of the tile in y axis, Default value is 256
                stride_x: int
                    The stride of the x axis, Default value is 128 (1/2 overalapping)
                    If you want to ignore the overlap, keep it same as tile_x
                stride_y: int
                    The stride of the y axis, Default value is 128 (1/2 overalapping)
                    If you want to ignore the overlap, keep it same as tile_y

            Returns
            -------
                None: save the tiles to the output folder

            Examples
            --------
                >>> from geotile import GeoTile
                >>> tiler = GeoTile('/path/to/raster/file.tif')
                >>> tiler.generate_raster_tiles('/path/to/output/folder')
                    # save the specific bands with other than default size
                >>> tiler.generate_raster_tiles('/path/to/output/folder', [3, 2, 1], tile_x=512, tile_y=512, stride_x=512, stride_y=512)
        """

        self.tile_x = tile_x
        self.tile_y = tile_y
        self.stride_x = stride_x
        self.stride_y = stride_y

        if image_format is None:
            image_format = pathlib.Path(self.path).suffix

        # create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # offset calculation
        offsets = self._calculate_offset(self.stride_x, self.stride_y)

        # iterate through the offsets
        for col_off, row_off in offsets:
            window = windows.Window(
                col_off=col_off, row_off=row_off, width=self.tile_x, height=self.tile_y)
            transform = windows.transform(window, self.ds.transform)
            meta = self.ds.meta.copy()
            nodata = meta['nodata']

            # update the meta data
            meta.update(
                {"width": window.width, "height": window.height, "transform": transform})

            # if the output bands is not None, add all bands to the output dataset
            # out_bands starts from i+1 because rasterio bands start from 1
            if out_bands is None:
                out_bands = [i+1 for i in range(0, self.ds.count)]

            else:
                meta.update({"count": len(out_bands)})

            # if data_type, update the meta
            if dtype:
                meta.update({"dtype": dtype})

            else:
                dtype = self.ds.meta['dtype']

            # tile name and path
            tile_name = 'tile_' + str(col_off) + '_' + \
                str(row_off) + '.' + image_format
            tile_path = os.path.join(output_folder, tile_name)

            # save the tiles with new metadata
            with rio.open(tile_path, 'w', **meta) as outds:
                outds.write(self.ds.read(
                    out_bands, window=window, fill_value=nodata, boundless=True).astype(dtype))

    def mask(self, input_vector: str, out_path: str, crop=False, invert=False, **kwargs):
        """Generate a mask raster from a vector
            This tool is similar to QGIS clip raster by mask layer (https://docs.qgis.org/2.8/en/docs/user_manual/processing_algs/gdalogr/gdal_extraction/cliprasterbymasklayer.html)

            Parameters
            ----------
                input_vector: str, python path
                    Path to the input vector (supports: shp, geojson, zip)
                    All the vector formats supported by geopandas are supported
                out_path: Str, python Path
                    Path to the output location of the mask raster
                crop: bool
                    If True, the mask will be cropped to the extent of the vector
                    If False, the mask will be the same size as the raster
                invert: bool
                    If True, the mask will be inverted, pixels outside the mask will be filled with 1 and pixels inside the mask will be filled with 0
                kwargs: dict
                    # rasterio.mask.mask (e.g. bounds, res, nodataetc.)
                    The kwargs from rasterio.mask.mask can be used here: https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html

            Returns
            -------
                out_path
                    Save the mask as a out_path

            Examples:
                >>> from geotile import GeoTile
                >>> tiler = GeoTile('/path/to/raster/file.tif')
                >>> tiler.generate_raster_mask('/path/to/vector.shp', '/path/to/output/file.tif')
        """

        # open the input vector
        df = gpd.read_file(input_vector)

        # check the coordinate system for both raster and vector and reproject vector if necessary
        raster_crs = self.meta['crs']
        if raster_crs != df.crs:
            df = df.to_crs(raster_crs)

        # get the bounds of the vector
        with rio.open(self.path) as src:
            out_image, out_transform = mask(
                src, df["geometry"], crop=crop, invert=invert, **kwargs)
            out_meta = src.meta.copy()

        # update the metadata
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform})

        # write the output raster
        with rio.open(out_path, 'w', **out_meta) as outds:
            outds.write(out_image)

    def rasterization(self, input_vector: str, out_path: str, value_col=None, **kwargs):
        """Convert vector shapes into raster 
            The metadata of the raster will be the same as the raster from GeoTile class.
            The raster will be filled with the value of the value_col of the vector.

            Parameters
            ----------
                input_vector: str, python path
                    Path to the input vector (supports: shp, geojson, zip)
                    All the vector formats supported by geopandas are supported
                out_path: str, python path
                    Path to the output location of the rasterized vector
                value_col: str
                    The column name of the vector to be rasterized
                    If None, the rasterization will be binary otherwise the rasterization will be the based on value of the column
                kwargs: dict
                    # rasterio.rasterize.rasterize (e.g. fill, transform etc.)
                    The kwargs from rasterio.rasterize can be used here: https://rasterio.readthedocs.io/en/latest/api/rasterio.rasterize.html


            Returns
            -------
                None: save the rasterized vector as a out_path

            Examples:
                >>> from geotile import GeoTile
                >>> tiler = GeoTile('/path/to/raster/file.tif')
                >>> tiler.rasterize_vector('/path/to/vector.shp', '/path/to/output/file.tif')
        """

        # open the input vector
        df = gpd.read_file(input_vector)

        # check the coordinate system for both raster and vector and reproject vector if necessary
        raster_crs = self.meta['crs']
        if raster_crs != df.crs:
            df = df.to_crs(raster_crs)

        # if value column is specified, rasterize the vector based on value column else bianary classification
        dataset = zip(df['geometry'], df[value_col]
                      ) if value_col else df['geometry']

        # rasterize the vector based on raster metadata
        mask = rasterize(dataset, self.ds.shape,
                         transform=self.meta['transform'], **kwargs)
        mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))

        # update the metadata
        meta = self.meta.copy()
        meta.update({'count': 1, "dtype": "uint8"})

        # write the output raster
        with rio.open(out_path, 'w', **meta) as outds:
            outds.write(mask)

    def reprojection(self, out_path: str, out_crs: str, resampling_method: str = 'nearest'):
        """Reproject a raster to a new coordinate system

            Parameters:
                out_path: str, python path
                    Path to the output location of the reprojected raster
                out_crs: str
                    The coordinate system of the output raster (e.g. 'EPSG:4326')
                resampling_method: str
                    The resampling method to use (e.g. 'bilinear')
                    It should be one of following,
                    "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average",
                    "mode", "gauss", "max", "min", "median", "q1", "q3", "std", "sum", "rms"

            Returns:
                out_path: str
                    Path to the output location of the reprojected raster

            Examples:
                >>> from geotile import GeoTile
                >>> tiler = GeoTile('/path/to/raster/file.tif')
                >>> tiler.reprojection('/path/to/output/file.tif', 'EPSG:4326')
        """
        # reproject raster to project crs
        with rio.open(self.path) as src:
            src_crs = src.crs
            transform, width, height = calculate_default_transform(
                src_crs, out_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()

            kwargs.update({
                'crs': out_crs,
                'transform': transform,
                'width': width,
                'height': height})

            with rio.open(out_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=out_crs,
                        resampling=Resampling[resampling_method])
        return(out_path)

    def resample(self, out_path: str, upscale_factor: int, resampling_method: str = 'bilinear'):
        """Resample a raster to a new resolution


            Parameters:
                out_path: str, python path
                    Path to the output location of the resampled raster
                upscale_factor: int
                    The upscale factor of the output raster (e.g. 2, i.e. 10x10 cell size to 5x5 cell size)
                    If you want to downscale by 2, that mean upscale_factor = 0.5
                resampling_method: str
                    The resampling method to use (e.g. 'bilinear')
                    It should be one of following,
                    "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average",
                    "mode", "gauss", "max", "min", "median", "q1", "q3", "std", "sum", "rms"

            Returns:
                out_path: str
                    Path to the output location of the resampled raster

            Examples:
                >>> from geotile import GeoTile
                >>> tiler = GeoTile('/path/to/raster/file.tif')
                >>> tiler.resample('/path/to/output/file.tif', 2)
        """
        # target dataset
        data = self.ds.read(
            out_shape=(
                self.ds.count,
                int(self.ds.height * upscale_factor),
                int(self.ds.width * upscale_factor)
            ),
            resampling=Resampling[resampling_method]
        )

        # scale image transform
        transform = self.ds.transform * self.ds.transform.scale(
            (self.ds.width / data.shape[-1]),
            (self.ds.height / data.shape[-2])
        )

        # update metadata
        meta = self.meta.copy()
        meta.update({
            "transform": transform,
            "width": int(self.ds.width * upscale_factor),
            "height": int(self.ds.height * upscale_factor),
        })

        # write the output raster
        with rio.open(out_path, 'w', **meta) as outds:
            outds.write(data)

        return(out_path)

    def close(self):
        """Close the dataset
        """
        self.ds.close()
