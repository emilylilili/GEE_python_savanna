Objective: Using the Python API to access data from the Google Earth Engine (GEE) platform, identify and export the savanna distribution map for a specified area, and prompt “Success, result saved to GEE” upon completion.

The overall structure is composed of the following parts:

(Function load_shapefiles) Import the shapefile for the study area (shapefile data), UAV image classification-derived cover training data (labeldatatrain), and validation data (labeldatatest) as labels for the cover regression model training.

(Function preprocess_data) Load SRTM terrain data, Sentinel-2, and Sentinel-1 imagery, and select bands; image dates can be modified as needed. Compute terrain features, texture, and vegetation indices as features for the cover regression model training.

(Function create_dataset) Align the label data to the corresponding geographic pixel features.

(Function train_model) Train separate cover models for woody plants and herbaceous plants and estimate the cover of woody and herbaceous plants in the study area.

(Function predict_evaluate_model) Validate model accuracy using the RMSE metric.

(Function identify_savanna) Extract pixels that meet the defined woody and herbaceous plant cover ranges for savanna and generate a savanna distribution map.

(Function save_results) Save the results. You will need to log in to GEE to check the result saving progress and download from Google Cloud.

Data Description:

The ground-truth training and validation data loaded are shapefile point files with attributes for woody and herbaceous plant cover.

Features for cover regression model training include:

a) Band reflectance (Sentinel-1 VV/VH, Sentinel-2 B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12);
b) Terrain features (elevation/aspect/slope);
c) Vegetation indices (NDVI/DVI/GNDVI/GRI/MTCI/EVI/LSWI/BI);
d) Texture (Blue-mean/Green-mean/Red-mean/NIR-mean/RedEdge1-mean/RedEdge2-mean/RedEdge1-mean/Purple-mean/Yellow-mean/NIR-var/NIR-homo/NIR-con/NIR-asm/NIR-diss/NIR-cor).
Data export default resolution is 70m; use save_results to set scale if modification is needed.

Installation Requirements:

Initially, install the required GEE API packages:

conda install earthengine-api
conda install geemap -c conda-forge
conda install gdal
Each run requires GEE Python API authentication. Validate the GEE account and initialize using ee.Initialize() each time.

Authentication will be saved once successful. If it expires later, re-authentication will be needed.
