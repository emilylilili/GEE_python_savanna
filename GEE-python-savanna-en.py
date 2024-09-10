#Objective: Using the Python API to access data from the Google Earth Engine (GEE) platform, identify and export the savanna distribution map for a specified area, and prompt “Success, result saved to GEE” upon completion.

#The overall structure is composed of the following parts:

#(Function load_shapefiles) Import the shapefile for the study area (shapefile data), UAV image classification-derived cover training data (labeldatatrain), and validation data (labeldatatest) as labels for the cover regression model training.

#(Function preprocess_data) Load SRTM terrain data, Sentinel-2, and Sentinel-1 imagery, and select bands; image dates can be modified as needed. Compute terrain features, texture, and vegetation indices as features for the cover regression model training.

#(Function create_dataset) Align the label data to the corresponding geographic pixel features.

#(Function train_model) Train separate cover models for woody plants and herbaceous plants and estimate the cover of woody and herbaceous plants in the study area.

#(Function predict_evaluate_model) Validate model accuracy using the RMSE metric.

#(Function identify_savanna) Extract pixels that meet the defined woody and herbaceous plant cover ranges for savanna and generate a savanna distribution map.

#(Function save_results) Save the results. You will need to log in to GEE to check the result saving progress and download from Google Cloud.

#Data Description:

#The ground-truth training and validation data loaded are shapefile point files with attributes for woody and herbaceous plant cover.

#Features for cover regression model training include:

#a) Band reflectance (Sentinel-1 VV/VH, Sentinel-2 B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12);
#b) Terrain features (elevation/aspect/slope);
#c) Vegetation indices (NDVI/DVI/GNDVI/GRI/MTCI/EVI/LSWI/BI);
#d) Texture (Blue-mean/Green-mean/Red-mean/NIR-mean/RedEdge1-mean/RedEdge2-mean/RedEdge1-mean/Purple-mean/Yellow-mean/NIR-var/NIR-homo/NIR-con/NIR-asm/NIR-diss/NIR-cor).
#Data export default resolution is 70m; use save_results to set scale if modification is needed.

#Installation Requirements:

#Initially, install the required GEE API packages:

#conda install earthengine-api
#conda install geemap -c conda-forge
#conda install gdal
#Each run requires GEE Python API authentication. Validate the GEE account and initialize using ee.Initialize() each time.

#Authentication will be saved once successful. If it expires later, re-authentication will be needed.

# Data to be imported
shpdata = 'D:/mapping_data/shp/area.shp'  # Shapefile polygon data of the study area
labeldatatrain = r'D:/mapping_data/data/train.shp'  # training data
labeldatatest = r'D:/mapping_data/data/test.shp'  # testing data

#Define the dates for Sentinel-2 and Sentinel-1 imagery
start_date = '2022-06-01'
end_date = '2022-09-30'


import ee
import os
import geemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up proxy
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
geemap.set_proxy(port=10809)

# Initialize Earth Engine
ee.Authenticate()
geemap.ee_initialize()

# Set up the map
Map = geemap.Map(center=[44.448, 116.550], zoom=3, height=600)

def main():
    """Load shapefiles and convert to Earth Engine objects."""
    shp, labeltrain, labeltest = load_shapefiles(shpdata, labeldatatrain, labeldatatest)
    
    """Load Sentinel-2 and Sentinel-1 data, select bands, and calculate texture indices and vegetation indices."""
    features, bands = preprocess_data(shp, start_date, end_date)

    """Construction and validation of the fractional woody vegetation cover model"""
    woody_training, woody_testing,woody_validated = create_dataset('woody', features, bands, labeltrain, labeltest)
    woody_model = train_model('woody', woody_training, bands)
    woody_predict = predict_evaluate_model('woody', features, bands, woody_model, woody_validated, return_type='regression')
    
    """Construction and validation of the fractional herbaceous vegetation cover model"""
    herb_training, herb_testing,herb_validated = create_dataset('herb', features, bands, labeltrain, labeltest)
    herb_model = train_model('herb', herb_training, bands)
    herb_predict = predict_evaluate_model('herb', features, bands, herb_model, herb_validated, return_type='regression')
    
    """savanna identification """
    savanna = identify_savanna(woody_predict, herb_predict,shp)
    
    """result export (FWVC map\FHVC map\ savanna map)"""
    save_results(woody_predict, shp, 'woody_predict', scale = 20)
    save_results(herb_predict, shp, 'herb_predict', scale = 20)
    save_results(savanna, shp, 'savanna')
    print(f"Success, result saved to GEE.")


def load_shapefiles(shpdata, labeldatatrain, labeldatatest):
    """Load shapefiles and convert to Earth Engine objects."""
    shp = geemap.shp_to_ee(shpdata)
    labeltrain = geemap.shp_to_ee(labeldatatrain)
    labeltest = geemap.shp_to_ee(labeldatatest)
    return shp, labeltrain, labeltest

def preprocess_data(shp, start_date, end_date):
    #removing cloud
    def cloudfunction_ST2(s2):
        quality = s2.select('QA60')
        RecorteNubes = 1 << 10
        RecorteCirros = 1 << 11
        qa = quality.bitwiseAnd(RecorteNubes).eq(0) and (quality.bitwiseAnd(RecorteCirros).eq(0))
        return s2.updateMask(qa)
    #load sentinel-2 imagery
    senl2 = (
        ee.ImageCollection('COPERNICUS/S2_SR')
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .map(cloudfunction_ST2)
        .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
        .median()
        .clipToCollection(shp)
        .round()
    )

    #load sentinel-1 imagery
    sen1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(shp) 
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) 
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )

    desc = sen1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    asc = sen1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

    senl1 = ee.Image.cat([
        ee.ImageCollection(asc.select('VH').merge(desc.select('VH'))).mean(),
        ee.ImageCollection(asc.select('VV').merge(desc.select('VV'))).mean(),
    ]).focal_median().clipToCollection(shp)
    
    #load dem imagery
    dem = ee.Image("USGS/SRTMGL1_003")
    elevation = dem.clip(shp)
    topo = ee.Algorithms.Terrain(elevation)
    
    #calculate vegetation indices.
    def vegetation_index(blue, green, red, RedEdge1, RedEdge2, RedEdge3, nir, RedEdge4, SWIR1, SWIR2):

        ndvi = senl2.normalizedDifference(['B8','B4']).rename("ndvi")
        dvi_exp = senl2.expression("B8-B4",{"B4":red,"B8":nir}).rename("dvi")
        gndvi_exp = senl2.expression("(B8-B3)/(B8+B3)",{"B3":green,"B8":nir}).rename("gndvi")
        gri_exp = senl2.expression("B8/B3-1",{"B3":green,"B8":nir}).rename("gri")
        mtci_exp = senl2.expression("(B7-B5)/(B6-B4)",{"B4":red,"B5":RedEdge1,"B6":RedEdge2,"B7":RedEdge3}).rename("mtci")
        evi_exp = senl2.expression("2.5*(B8-B4)/B8+6*B4-7.5*B2+1",{"B2":blue,"B4":red,"B8":nir}).rename("evi")
        lswi_exp = senl2.expression("(B8-B11)/(B8+B11)",{"B8":nir,"B11":SWIR1}).rename("lswi")
        bi_exp = senl2.expression("sqrt(B3*B3+B4*B4+B8*B8)",{"B3":green,"B4":red,"B8":nir}).rename("bi")

        vegindex_features = ndvi.addBands([dvi_exp, gndvi_exp, gri_exp, mtci_exp, evi_exp, lswi_exp, bi_exp])
        return vegindex_features
        
    #calculate texture
    def texture_index(blue, green, red, RedEdge1, RedEdge2, RedEdge3, nir, RedEdge4, SWIR1, SWIR2):
        glcmblue = blue.glcmTexture(4)
        glcmgreen = green.glcmTexture(4)
        glcmred = red.glcmTexture(4)
        glcmRedEdge1 = RedEdge1.glcmTexture(4)
        glcmRedEdge2 = RedEdge2.glcmTexture(4)
        glcmRedEdge3 = RedEdge3.glcmTexture(4)
        glcmNIR = nir.glcmTexture(4)
        glcmRedEdge4 = RedEdge4.glcmTexture(4)
        glcmSWIR1 = SWIR1.glcmTexture(4)
        glcmSWIR2 = SWIR2.glcmTexture(4)

        B2savg = glcmblue.select('B2_savg')
        B3savg = glcmgreen.select('B3_savg')
        B4savg = glcmred.select('B4_savg')
        B5savg = glcmRedEdge1.select('B5_savg')
        B6savg = glcmRedEdge2.select('B6_savg')
        B7savg = glcmRedEdge3.select('B7_savg')
        B8savg = glcmNIR.select('B8_savg')
        B8Asavg = glcmRedEdge4.select('B8A_savg')
        B11savg = glcmSWIR1.select('B11_savg')
        B12savg = glcmSWIR2.select('B12_savg')

        B8contrast = glcmNIR.select('B8_contrast')
        B8corr = glcmNIR.select('B8_corr')
        B8diss = glcmNIR.select('B8_diss')
        B8asm = glcmNIR.select('B8_asm')
        B8var = glcmNIR.select('B8_var')
        B8idm = glcmNIR.select('B8_idm')
        B8ent = glcmNIR.select('B8_ent')

        texture_features = B2savg.addBands([B3savg, B4savg, B5savg, B6savg, B7savg, B8savg, B8Asavg, B11savg, B12savg, B8contrast, B8corr, B8diss, B8asm, B8var, B8idm, B8ent])
        return texture_features

    blue = senl2.select(['B2']).toUint16()
    green = senl2.select(['B3']).toUint16()
    red = senl2.select(['B4']).toUint16()
    nir = senl2.select(['B8']).toUint16()
    RedEdge1 = senl2.select(['B5']).toUint16()
    RedEdge2 = senl2.select(['B6']).toUint16()
    RedEdge3 = senl2.select(['B7']).toUint16()
    RedEdge4 = senl2.select(['B8A']).toUint16()
    SWIR1 = senl2.select(['B11']).toUint16()
    SWIR2 = senl2.select(['B12']).toUint16()

    veg_index = vegetation_index(blue, green, red, RedEdge1, RedEdge2, RedEdge3, nir, RedEdge4, SWIR1, SWIR2)
    tex_index = texture_index(blue, green, red, RedEdge1, RedEdge2, RedEdge3, nir, RedEdge4, SWIR1, SWIR2)
    topo10 = topo.select(['elevation', 'slope', 'aspect'])  # Assuming topo is the Terrain output with these bands

    #合并特征
    composite_features = senl2.addBands([senl1, veg_index, tex_index, topo10])
    features = composite_features.reproject(crs='EPSG:4326', scale=20)
    bands = features.bandNames()
    
    return features, bands

def create_dataset(vegetation_type, features, bands, labeltrain, labeltest):
    """Create Training and Validation Datasets"""
    if vegetation_type == 'woody':
        training = features.sampleRegions(
            collection=labeltrain,
            properties=['woody'],
            scale=20,
            geometries=True)
        validateddata = features.sampleRegions(
            collection=labeltest,
            properties=['woody'],
            scale=20,
            geometries=True)
    elif vegetation_type == 'herb':
        training = features.sampleRegions(
            collection=labeltrain,
            properties=['herb'],
            scale=20,
            geometries=True)
        validateddata = features.sampleRegions(
            collection=labeltest,
            properties=['herb'],
            scale=20,
            geometries=True)
    else:
        raise ValueError("Invalid vegetation type. Choose 'woody' or 'herb'.")

    trainingdata = training.randomColumn()
    training = trainingdata.filter(ee.Filter.lt('random', 0.8))
    testing = trainingdata.filter(ee.Filter.gte('random',0.8))
    return training, testing,validateddata

def train_model(vegetation_type, training, bands):
    """random forest regression"""
    if vegetation_type == 'woody':
        label = 'woody'
    elif vegetation_type == 'herb':
        label = 'herb'
    else:
        raise ValueError("Invalid vegetation type. Choose 'woody' or 'herb'.")
        
    RFmodel = ee.Classifier.smileRandomForest(numberOfTrees=160).setOutputMode('REGRESSION').train(**{
        'features': training,
        'classProperty': label,
        'inputProperties': bands
    })
    return RFmodel

def predict_evaluate_model(vegetation_type, features, bands, RFmodel, validateddata, return_type='regression'):
    """Calculate FVC and Compute RMSE"""
    if vegetation_type == 'woody':
        label = 'woody'
    elif vegetation_type == 'herb':
        label = 'herb'
    else:
        raise ValueError("Invalid vegetation type. Choose 'woody' or 'herb'.")
        
    regression = features.select(bands).classify(RFmodel, 'predicted')#Generating FVC map

    # RMSE
    predictedValidation = regression.sampleRegions(**{
        'collection': validateddata,
        'properties': [label],
        'scale': 20})
    sampleValidation = predictedValidation.select([label, 'predicted'])
    observationValidation = ee.Array(sampleValidation.aggregate_array(label))
    predictionValidation = ee.Array(sampleValidation.aggregate_array('predicted'))
    residualsValidation = observationValidation.subtract(predictionValidation)
    rmseValidation = residualsValidation.pow(2).reduce('mean', [0]).sqrt()
    rmse_value = ee.Number(rmseValidation).getInfo()
    print(f"{label} RMSE:{rmse_value}")
    return regression
       
def save_results(img, shp,description, scale=70):
    """save result map to Google Drive."""
    geemap.ee_export_image_to_drive(
        img, 
        description=description, 
        folder=description, 
        region=shp.geometry(), 
        scale=scale, 
        crs='EPSG:4326',
        maxPixels=350000000
    )

def identify_savanna(woodyFVC, herbFVC,shp):
    """savanna identification"""
    # Define Criteria for Identifying Savanna
    savanna_identify = woodyFVC.gt(10) and woodyFVC.lt(40).And(herbFVC.gt(40))

    # Apply the Criteria to FWVC and FHVC results
    savanna = woodyFVC.where(savanna_identify, 1).clip(shp)

    return savanna 

if __name__ == "__main__":
    main()