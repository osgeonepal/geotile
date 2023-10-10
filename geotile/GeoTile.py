# inbuilt libraries
import os
import itertools
from typing import Optional
import pathlib

# numpy library
import numpy as np

# rasterio library
import rasterio as rio
from rasterio import windows
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine

# geopandas library
import geopandas as gpd

# to check if the input raster dtype is int based or float based
_int_dtypes = ["uint8", "uint16", "uint32", "int8", "int16", "int32"]


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
        self.height = self.meta["height"]
        self.width = self.meta["width"]
        self.meta["crs"] = self.ds.crs

    def _calculate_offset(
        self, stride_x: Optional[int] = None, stride_y: Optional[int] = None
    ) -> tuple:
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
        self.offsets = offsets

    def _windows_transform_to_affine(self, window_transform: Optional[tuple]):
        """Convert the window transform to affine transform

        Parameters
        ----------
            window_transform: tuple
                tuple of window transform

        Returns
        -------
            tuple: tuple of affine transform
        """
        a, b, c, d, e, f, _, _, _ = window_transform
        return Affine(a, b, c, d, e, f)

    def shuffle_tiles(self, random_state: Optional[int] = None):
        """Shuffle the tiles

        Parameters
        ----------
            random_state: int
                Random state for shuffling the tiles

        Returns
        -------
            None: Shuffle the tiles. The offsets will be shuffled in place

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.shuffle_tiles()
        """
        # check if random_state is not None
        if random_state is not None:
            self.random_state = random_state
            np.random.seed(self.random_state)

        assert (
            len(self.offsets) == len(self.tile_data) == len(self.window_transform)
        ), "The number of offsets and window data should be same"

        # shuffle the offsets and window data
        p = np.random.permutation(len(self.offsets))
        self.offsets = np.array(self.offsets)[p]
        self.tile_data = np.array(self.tile_data)[p]
        self.window_transform = np.array(self.window_transform)[p]

    def tile_info(self):
        """Get the information of the tiles

        Returns
        -------
            dict: (tile_x, tile_y, stride_x, stride_y)

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.tile_info()
                {'tile_x': 256, 'tile_y': 256, 'stride_x': 128, 'stride_y': 128}
        """
        return {
            "tile_x": self.tile_x,
            "tile_y": self.tile_y,
            "stride_x": self.stride_x,
            "stride_y": self.stride_y,
        }

    def get_dtype(self, data_array: np.ndarray):
        """Get the appropriate dtype for the data array

        Parameters
        ----------
            data_array: np.ndarray
                The data array for which the dtype will be calculated

        Returns
        -------
            str: The appropriate dtype for the data array

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> data_array = np.array([1, 2, 3, 4])
            >>> gt.get_dtype(data_array)
                'uint8'
        """
        if isinstance(data_array, np.ndarray):
            return str(data_array.dtype)
        
        else:
            return 'Input is not a NumPy array.'

    def generate_tiles(
        self,
        output_folder: Optional[str] = "tiles",
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        save_tiles: Optional[bool] = True,
        save_transform: Optional[bool] = False,
        out_bands: Optional[list] = None,
        image_format: Optional[str] = None,
        dtype: Optional[str] = None,
        tile_x: Optional[int] = 256,
        tile_y: Optional[int] = 256,
        stride_x: Optional[int] = 128,
        stride_y: Optional[int] = 128,
    ):
        """
        Save the tiles to the output folder

            Parameters
            ----------
                output_folder : str
                    Path to the output folder
                save_tiles : bool
                    If True, the tiles will be saved to the output folder else the tiles will be stored in the class
                save_transform : bool
                    If True, the transform will be saved to the output folder in txt file else it will only generate the tiles
                suffix : str
                    The suffix of the tile name (eg. _img)
                prefix : str
                    The prefix of the tile name (eg. img_)
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
                >>> gt = GeoTile('/path/to/raster/file.tif')
                >>> gt.generate_raster_tiles('/path/to/output/folder', prefix='img_')
                    # save the specific bands with other than default size
                >>> gt.generate_raster_tiles('/path/to/output/folder', [3, 2, 1], tile_x=512, tile_y=512, stride_x=512, stride_y=512)
        """

        self.tile_x = tile_x
        self.tile_y = tile_y
        self.stride_x = stride_x
        self.stride_y = stride_y

        # create the output folder if it doesn't exist
        if not os.path.exists(output_folder) and save_tiles is True:
            os.makedirs(output_folder)

        # offset calculation
        self._calculate_offset(self.stride_x, self.stride_y)

        # store all the windows data as a list, windows shape: (band, tile_y, tile_x)
        self.tile_data = []

        # store all the transform data as a list
        self.window_transform = []

        # iterate through the offsets and save the tiles
        for i, (col_off, row_off) in enumerate(self.offsets):
            window = windows.Window(
                col_off=col_off, row_off=row_off, width=self.tile_x, height=self.tile_y
            )
            transform = windows.transform(window, self.ds.transform)

            # convert the window transform to affine transform and append to the list
            transform = self._windows_transform_to_affine(transform)
            self.window_transform.append(transform)

            # copy the meta data
            meta = self.ds.meta.copy()
            nodata = meta["nodata"]

            # update the meta data
            meta.update(
                {"width": window.width, "height": window.height, "transform": transform}
            )

            # if the output bands is not None, add all bands to the output dataset
            # out_bands starts from i+1 because rasterio bands start from 1
            if out_bands is None:
                out_bands = [i + 1 for i in range(0, self.ds.count)]

            else:
                meta.update({"count": len(out_bands)})

            # read the window data and append to the list
            single_tile_data = self.ds.read(
                out_bands, window=window, fill_value=nodata, boundless=True
            )
            self.tile_data.append(single_tile_data)

            # if data_type, update the meta
            if dtype:
                meta.update({"dtype": dtype})

            else:
                dtype = self.ds.meta["dtype"]

            if save_tiles:
                # check if image_format is None
                image_format = image_format or pathlib.Path(self.path).suffix

                # create the output folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Set default values for suffix and prefix if they are None
                suffix = suffix or ""
                prefix = prefix or ""

                # Construct the tile name and path
                tile_name = (
                    f"{prefix}{str(i)}{suffix}{image_format}"
                    if suffix or prefix
                    else f"tile_{col_off}_{row_off}{image_format}"
                )

                tile_path = os.path.join(output_folder, tile_name)

                if save_transform:
                    # tile georeference data to reconstruct spatial location from output inference bboxes
                    geo_reference_tile_worldfile = (
                        f"{prefix}{str(i)}{suffix}.txt"
                        if suffix or prefix
                        else f"tile_{col_off}_{row_off}.txt"
                    )

                    geo_reference_tile_worldfile_path = os.path.join(
                        output_folder, geo_reference_tile_worldfile
                    )

                    crs = meta["crs"]
                    crs = crs.to_proj4()

                    # raster affine transform information
                    with open(geo_reference_tile_worldfile_path, "w") as f:
                        f.write(str(transform.to_gdal()))
                        f.write("\n")
                        f.write(crs)

                # save the tiles with new metadata
                with rio.open(tile_path, "w", **meta) as outds:
                    outds.write(
                        self.ds.read(
                            out_bands, window=window, fill_value=nodata, boundless=True
                        ).astype(dtype)
                    )

        if not save_tiles:
            # convert list to numpy array
            self.tile_data = np.array(self.tile_data)

            # dtype conversion
            self.tile_data = self.tile_data.astype(dtype)

            # move axis to (n, tile_y, tile_x, band)
            self.tile_data = np.moveaxis(self.tile_data, 1, -1)

    def save_tiles(
        self,
        output_folder: str,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        image_format: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """Save the tiles to the output folder

        Parameters
        ----------
            output_folder : str
                Path to the output folder
            prefix : str
                The prefix of the tile name (eg. img_)
            suffix : str
                The suffix of the tile name (eg. _img)
            image_format : str
                The image format (eg. tif), if None, the image format will be the same as the input raster format (eg. tif)

            dtype : str, np.dtype
                The output dtype (eg. uint8, float32), if None, the dtype will be the same as the input raster

        Returns
        -------
            None: save the tiles to the output folder

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.save_tiles('/path/to/output/folder', prefix='img_')
        """
        # create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # meta data
        meta = self.meta.copy()
        # nodata = meta['nodata']
        meta.update(
            {
                "width": self.tile_x,
                "height": self.tile_y,
            }
        )

        # check if image_format is None
        image_format = image_format or pathlib.Path(self.path).suffix

        # Set default values for suffix and prefix if they are None
        suffix = suffix or ""
        prefix = prefix or ""

        # if data_type, update the meta
        meta.update({"dtype": dtype or self.get_dtype(self.tile_data)})

        # iterate through the offsets and windows_data and save the tiles
        for i, ((col_off, row_off), wd, wt) in enumerate(
            zip(self.offsets, self.tile_data, self.window_transform)
        ):
            # update meta data with transform
            meta.update({"transform": tuple(wt)})

            # Construct the tile name and path
            tile_name = (
                f"{prefix}{str(i)}{suffix}{image_format}"
                if suffix or prefix
                else f"tile_{col_off}_{row_off}{image_format}"
            )

            tile_path = os.path.join(output_folder, tile_name)

            # move axis to (band, tile_y, tile_x)
            wd = np.moveaxis(wd, -1, 0)

            # update the meta with number of bands
            meta.update({"count": wd.shape[0]})

            # save the tiles with new metadata
            with rio.open(tile_path, "w", **meta) as outds:
                outds.write(wd.astype(meta["dtype"]))

    def merge_tiles(
        self,
        output_path: str,
        out_bands: Optional[list] = None,
        image_format: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """Merge the tiles and save the merged raster.
        Make sure the tiles are generated before merging and all the tiles having similar properties (eg. dtype, crs, transform)

        Parameters
        ----------
            output_path : str
                Path to the output raster
            out_bands : list
                The bands to save (eg. [3, 2, 1]), if None, the output bands will be same as the input raster bands
            image_format : str
                The image format (eg. tif), if None, the image format will be the same as the input raster format (eg. tif)
            dtype : str, np.dtype
                The output dtype (eg. uint8, float32), if None, the dtype will be the same as the input raster
            meta: dict
                The metadata of the output raster.
                If provided, the output raster will be created with the provided metadata
                else the output raster will be created with the metadata of the input raster or the other given parameters

        Returns
        -------
            None: save the merged raster to the output folder

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.generate_raster_tiles(save_tiles=False)
            >>> gt.merge_tiles('/path/to/output/file.tif')
        """
        # if self.tile_data is list, convert it to numpy array
        if isinstance(self.tile_data, list):
            self.tile_data = np.array(self.tile_data)

        # if data_type, update the meta
        if dtype:
            self.meta.update({"dtype": dtype})

        # if out_bands is None, update the meta with number of bands
        if out_bands is None:
            self.meta.update({"count": self.tile_data.shape[-1]})
            out_bands = [i + 1 for i in range(0, self.ds.count)]

        else:
            self.meta.update({"count": len(out_bands)})

        # check if image_format is None
        image_format = image_format or pathlib.Path(self.path).suffix

        # change numpy shape to (n, bands, x_tile, y_tile)
        tile_data = np.moveaxis(self.tile_data, -1, 1)

        # write the output raster
        with rio.open(output_path, "w", **self.meta) as outds:
            outds.write(tile_data[:, out_bands, :, :].astype(self.meta["dtype"]))

    def normalize_tiles(self):
        """Normalize the tiles between 0 and 1 (MinMaxScaler)

        Returns
        -------
            None: Normalize the tiles. The normalized tiles will be stored in the class

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.generate_raster_tiles(save_tiles=False)
            >>> gt.normalize_tiles()
        """
        # normalize the tiles
        # if self.tile_data is list, convert it to numpy array
        if isinstance(self.tile_data, list):
            self.tile_data = np.array(self.tile_data)

        # if datatype is int based (eg. uint8, uint16, int8, int16), convert those to float32 for normalization
        # if not changed, the normalization will only generate 0 and 1 values for the tiles
        if self.tile_data.dtype in _int_dtypes:
            self.tile_data = self.tile_data.astype("float32")

        # find max and min values in whole tiles on each channel
        # my windows_data shape: (n, tile_y, tile_x, band)
        max_values = np.max(self.tile_data, axis=(0, 1, 2))
        min_values = np.min(self.tile_data, axis=(0, 1, 2))

        # Normalize the tiles and update the tile_data for each channel independently
        for channel in range(self.tile_data.shape[-1]):
            self.tile_data[:, :, :, channel] = (
                self.tile_data[:, :, :, channel] - min_values[channel]
            ) / (max_values[channel] - min_values[channel])

    def convert_nan_to_zero(self):
        """Convert nan values to zero

        Returns
        -------
            None: Convert nan values to zero. The converted tiles will be stored in the class

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.generate_raster_tiles(save_tiles=False)
            >>> gt.convert_nan_to_zero()
        """
        # if self.tile_data is list, convert it to numpy array
        if isinstance(self.tile_data, list):
            self.tile_data = np.array(self.tile_data)

        # convert nan values to zero
        self.tile_data = np.nan_to_num(self.tile_data)

    def drop_nan_tiles(self):
        """Drop the tiles with nan values

        Returns
        -------
            None: Drop the tiles with nan values. The dropped tiles will be lost forever

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.generate_raster_tiles(save_tiles=False)
            >>> gt.drop_nan_tiles()
        """
        # if self.tile_data is list, convert it to numpy array
        if isinstance(self.tile_data, list):
            self.tile_data = np.array(self.tile_data)

        # drop the tiles with nan values and also drop corresponding window transform and offsets
        nan_index = np.argwhere(np.isnan(self.tile_data).any(axis=(1, 2, 3)))
        self.tile_data = np.delete(self.tile_data, nan_index, axis=0)
        self.window_transform = np.delete(self.window_transform, nan_index, axis=0)
        self.offsets = np.delete(self.offsets, nan_index, axis=0)

    def drop_zero_value_tiles(self):
        """Drop the tiles with all zero values

        Returns
        -------
            None: Drop the tiles with all zero values. The dropped tiles will lost forever

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.generate_raster_tiles(save_tiles=False)
            >>> gt.drop_all_zero_value_tiles()
        """
        # if self.tile_data is list, convert it to numpy array
        if isinstance(self.tile_data, list):
            self.tile_data = np.array(self.tile_data)

        # drop the tiles with nan values and also drop corresponding window transform and offsets
        zero_index = np.argwhere(np.all(self.tile_data == 0, axis=(1, 2, 3)))
        self.tile_data = np.delete(self.tile_data, zero_index, axis=0)
        self.window_transform = np.delete(self.window_transform, zero_index, axis=0)
        self.offsets = np.delete(self.offsets, zero_index, axis=0)

    def save_numpy(self, file_name: str, dtype: Optional[str] = None):
        """Save the tiles to the output folder

        Parameters
        ----------
            file_name : str
                Path or name of the output numpy file

            dtype : str, np.dtype
                The output dtype (eg. uint8, float32), if None, the dtype will be the same as the input raster

        Returns
        -------
            None: save the tiles to the output folder, the shape of the numpy file will be (n, tile_x, tile_y, band)

        Examples
        --------
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.generate_raster_tiles(save_tiles=False)
            >>> gt.save_numpys('/folder/to/output/file.npy')
        """
        # check if the file name path exists or not, if not, create the folder
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        # if self.tile_data is list, convert it to numpy array
        if isinstance(self.tile_data, list):
            self.tile_data = np.array(self.tile_data)

        # if data_type is none, get the appropriate dtype
        if dtype is None:
            dtype = self.get_dtype(self.tile_data)

        # save the numpy file
        np.save(file_name, self.tile_data.astype(dtype))

    def mask(
        self, input_vector: str, out_path: str, crop=False, invert=False, **kwargs
    ):
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
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.generate_raster_mask('/path/to/vector.shp', '/path/to/output/file.tif')
        """

        # open the input vector
        df = gpd.read_file(input_vector)

        # check the coordinate system for both raster and vector and reproject vector if necessary
        raster_crs = self.meta["crs"]
        if raster_crs != df.crs:
            df = df.to_crs(raster_crs)

        # get the bounds of the vector
        with rio.open(self.path) as src:
            out_image, out_transform = mask(
                src, df["geometry"], crop=crop, invert=invert, **kwargs
            )
            out_meta = src.meta.copy()

        # update the metadata
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        # write the output raster
        with rio.open(out_path, "w", **out_meta) as outds:
            outds.write(out_image)

    def rasterization(
        self,
        input_vector: str,
        out_path: str,
        value_col=None,
        no_data: Optional[int] = None,
        **kwargs,
    ):
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
            no_data: int
                The no data value of the raster.
                Default value is None.
            kwargs: dict
                # rasterio.rasterize.rasterize (e.g. fill, transform etc.)
                The kwargs from rasterio.rasterize can be used here: https://rasterio.readthedocs.io/en/latest/api/rasterio.rasterize.html


        Returns
        -------
            None: save the rasterized vector as a out_path

        Examples:
            >>> from geotile import GeoTile
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.rasterize_vector('/path/to/vector.shp', '/path/to/output/file.tif', fill=0)
        """

        # open the input vector
        df = gpd.read_file(input_vector)

        # check the coordinate system for both raster and vector and reproject vector if necessary
        raster_crs = self.meta["crs"]
        if raster_crs != df.crs:
            print(
                f"CRS of raster doesn't match with vector. Reprojecting the vector ({df.crs}) to the raster coordinate system ({raster_crs})"
            )
            df = df.to_crs(raster_crs)

        # if value column is specified, rasterize the vector based on value column else bianary classification
        dataset = zip(df["geometry"], df[value_col]) if value_col else df["geometry"]

        # rasterize the vector based on raster metadata
        mask = rasterize(
            dataset,
            self.ds.shape,
            transform=self.meta["transform"],
            **kwargs,
        )
        mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))

        # update the metadata
        meta = self.meta.copy()
        meta.update({"count": 1, "dtype": self.get_dtype(mask), "nodata": no_data})

        # write the output raster
        with rio.open(out_path, "w", **meta) as outds:
            outds.write(mask)

    def reprojection(
        self, out_path: str, out_crs: str, resampling_method: str = "nearest"
    ):
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
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.reprojection('/path/to/output/file.tif', 'EPSG:4326')
        """
        # reproject raster to project crs
        with rio.open(self.path) as src:
            src_crs = src.crs
            transform, width, height = calculate_default_transform(
                src_crs, out_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()

            kwargs.update(
                {
                    "crs": out_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            with rio.open(out_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=out_crs,
                        resampling=Resampling[resampling_method],
                    )
        return out_path

    def resample(
        self, out_path: str, upscale_factor: int, resampling_method: str = "bilinear"
    ):
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
            >>> gt = GeoTile('/path/to/raster/file.tif')
            >>> gt.resample('/path/to/output/file.tif', 2)
        """
        # target dataset
        data = self.ds.read(
            out_shape=(
                self.ds.count,
                int(self.ds.height * upscale_factor),
                int(self.ds.width * upscale_factor),
            ),
            resampling=Resampling[resampling_method],
        )

        # scale image transform
        transform = self.ds.transform * self.ds.transform.scale(
            (self.ds.width / data.shape[-1]), (self.ds.height / data.shape[-2])
        )

        # update metadata
        meta = self.meta.copy()
        meta.update(
            {
                "transform": transform,
                "width": int(self.ds.width * upscale_factor),
                "height": int(self.ds.height * upscale_factor),
            }
        )

        # write the output raster
        with rio.open(out_path, "w", **meta) as outds:
            outds.write(data)

        return out_path

    def close(self):
        """Close the dataset"""
        self.ds.close()
