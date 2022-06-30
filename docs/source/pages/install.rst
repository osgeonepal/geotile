GeoTile Installation
====================

Conda Installation 
------------------

The easy and best way to install the GeoTile library is using ``conda`` distribution,

.. code-block:: python 

    conda install -c conda-forge geotile

Pip Installation 
----------------

To install the package with pip installation, you have to install its dependencies first. Following packages are the dependencies of GeoTile:

- gdal (http://www.gdal.org/)
- numpy (http://www.numpy.org/)
- geopandas (https://pypi.python.org/pypi/geopandas)
- rasterio (https://pypi.python.org/pypi/rasterio)

Windows Pip Installation 
^^^^^^^^^^^^^^^^^^^^^^^^

In windows, the dependencies can be install using ``pipwin`` command,

.. code:: shell

    pip install pipwin
    pipwin install gdal numpy geopandas rasterio


Now you can install the library using pip install command,

.. code:: shell 

    pip install geotile


Linux Installation
^^^^^^^^^^^^^^^^^^

In Debian/Ubuntu, the dependencies can be install using ``apt-get`` command,

.. code:: shell 

    sudo add-apt-repository ppa:ubuntugis/ppa
    sudo apt update -y; sudo apt upgrade -y;
    sudo apt install gdal-bin libgdal-dev
    pip3 install pygdal=="`gdal-config --version`.*"


More Instruction for Dependencies Installation 
----------------------------------------------

The following links are the instructions for installing the dependencies of GeoTile:

1. `GeoPandas installation <https://geopandas.org/en/stable/getting_started/install.html>`_
2. `Rasterio Installation <https://rasterio.readthedocs.io/en/latest/installation.html>`_

