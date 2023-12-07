import os
import glob
from typing import Optional, Union

import rasterio as rio
from rasterio.merge import merge

import fiona


# mosaic the tiles
def mosaic(input_folder: str, output_file: str, image_format: Optional[str] = 'tif', **kwargs):
    """Mosaic the rasters inside the input folder

        This method is used to merge the tiles into single file

        Parameters
        ----------
            input_folder: str, python path 
                Path to the input folder
            output_file: str, python path
                Path to the output file
            image_format: str
                The image format (eg. tif), if None, the image format will be the same as the input raster format.
            kwargs: dict 
                The kwargs from rasterio.merge.merge can be used here: https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html#rasterio.merge.merge (e.g. bounds, res, nodata etc.)

        Returns
        -------
            output_file
                Save the mosaic as a output_file. Returns the output_file path

        Examples
        --------
            >>> from geotile import mosaic
            >>> mosaic('/path/to/input/folder', '/path/to/output/file.tif')
    """

    # get the list of input rasters to merge
    search_criteria = "*.{}".format(image_format)
    q = os.path.join(input_folder, search_criteria)
    input_files = sorted(glob.glob(q))

    # Open and add all the input rasters to a list
    src_files_to_mosaic = []
    for files in input_files:
        src = rio.open(files)
        src_files_to_mosaic.append(src)

    # Merge the rasters
    mosaic, out_trans = merge(src_files_to_mosaic, **kwargs)

    # update the metadata
    meta = src.meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })

    # write the output raster
    with rio.open(output_file, 'w', **meta) as outds:
        outds.write(mosaic)

    return output_file


# vectorize the tiles
def vectorize(input_raster: str, output_file:str, band: Optional[int] = 1, raster_values: Optional[Union[str, list]] = 'all', mask: Optional[str] = None):
    """Vectorize the raster

        This method is used to vectorize the raster

        Parameters
        ----------
            input_raster: str, python path 
                Path to the input raster
            output_file: str, python path
                Path to the output file
            band: int
                The band to be vectorized
            raster_values: str, list
                The values to be vectorized. Default is 'all'

        Returns
        -------
            output_file
                Save the vectorized raster as a output_file. Returns the output_file path

        Examples
        --------
            >>> from geotile import vectorize
            >>> vectorize('/path/to/input/raster.tif', '/path/to/output/file.shp')
            >>> vectorize('/path/to/input/raster.tif', '/path/to/output/file.shp', raster_values=[1])
    """

    # Open the raster
    with rio.open(input_raster) as src:
        raster = src.read(band)

    # Vectorize the raster
    shapes = rio.features.shapes(raster, transform=src.transform, mask=mask)

    # if remove_values is not 'all'; filter out the required records
    records = []
    if isinstance(raster_values, list):
        for i, (geom, value) in enumerate(shapes):
            if value in raster_values:
                records.append({
                    'geometry': geom,
                    'properties': {'value': value},
                })
    
    # if remove_values is None, add all shapes to the record
    elif raster_values=='all':
        for i, (geom, value) in enumerate(shapes):
            records.append({
                'geometry': geom,
                'properties': {'value': value},
            })

    # else raise the exception
    else:
        raise ValueError("remove_values either should be 'all' or list of integers")


    # Save the vectorized raster
    with fiona.open(output_file, 'w', crs=src.crs, driver='ESRI Shapefile', schema={'geometry': 'Polygon', 'properties': [('value', 'int')]}) as dst:
        dst.writerecords(records)

    return output_file