# iohub

This library provides io utilites for ND image data. 

Supported formats: 

Read (iohub.reader): 
- single-page TIFF, OME-TIFF, NDTiff written by micro-manager, 
- custom data formats used by Biohub microscopes (e.g., PTI, DaXi, mantis).
- all the formats writte by this library.

Write (iohub.writer): 
- OME-TIFF, 
- OME-zarr, 
- TIFF stacks that mimic OME-zarr structure. This format provides benefits of a chunked format like zarr for visualizaion tools and [analysis pipelines that only support TIFF](https://github.com/mehta-lab/recOrder/issues/276).

Data access API (under discussion):
- to visualize data in above formats using napari and Fiji.
- enable efficient, paraellel, and lazy loading of data in deconvolution and DL pipelines via iohub.reader module.
 
