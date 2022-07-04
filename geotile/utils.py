import os
import glob
from typing import Optional

import rasterio as rio
from rasterio.merge import merge


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
            >>> from geotile import GeoTile
            >>> tiler = GeoTile('/path/to/raster/file.tif')
            >>> tiler.mosaic_rasters('/path/to/input/folder', '/path/to/output/file.tif')
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
