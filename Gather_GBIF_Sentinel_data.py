#Import libraries

from pystac_client import Client

import shapely.geometry as geom
from shapely.geometry import box
from shapely.geometry import Point
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon


from pygbif import species as species
from pygbif import occurrences as occ
from datetime import datetime

import time
import os
from zipfile import ZipFile
import random

import pycountry

import geopandas as gpd
import numpy as np
import pandas as pd

import rasterio

import re

import boto3

from PIL import Image

import sys

log_file = open("data/run.log", "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

#set minimum sample size per tile
occurrences_sample=200

cwd=os.getcwd()


#Get credentials for GBIf and CDSE from runtime variables
access_key=os.getenv("ACCESS_KEY")
secret_key = os.getenv("SECRET_KEY")
passw=os.getenv('GBIF_PASSWORD')
username=os.getenv('GBIF_USERNAME')
emailadd=os.getenv('GBIF_EMAIL')

print(access_key)
print(secret_key)
print(passw)
print(username)
print(emailadd)

#Define functions

 #Biodiversity functions

def shannon(df):
    species_counts=df.groupby(['species'])['individualCount'].sum()
    proportions=species_counts/species_counts.sum()
    shannon_index= -sum(proportions * np.log(proportions))
    return shannon_index
    
#Function for simpson index
def simpson(df):
    species_counts = df.groupby('species')['individualCount'].sum()
    N = species_counts.sum()
    D = (species_counts * (species_counts - 1)).sum() / (N * (N - 1))
    return 1 - D



#CDSE functions


#Function for extracting polygons from MGRS dataframe rows
def get_polygon(geom):
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    
    if isinstance(geom, GeometryCollection):
        polygons = [
            g for g in geom.geoms
            if isinstance(g, (Polygon, MultiPolygon))
        ]

        if len(polygons) == 1:
            return polygons[0]
        elif len(polygons) > 1:
            return MultiPolygon(polygons)

    return None

#Function for filtering tiles list to only those in a given country
def find_tiles_in_country(country, mgrs):

    country_geom = country.geometry.iloc[0]

    tiles = mgrs[mgrs.intersects(country_geom)]

    tile_names = tiles["Name"].tolist()

    return tile_names
    
#Function for searching CDSE STAC catalogue for data relating to a particular tile
def cdse_search(mgrs_tile):

    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1")

    #Reformat tile name to match
    if 'MGRS-' not in mgrs_tile:
        mgrs_tile='MGRS-'+mgrs_tile


    search_results = catalog.search(
    collections=["sentinel-2-l2a"],
    datetime="2024-01-01/2024-12-31",
    query={
        "grid:code": {
            "eq": mgrs_tile
        }
    },
    #Sort by least cloud cover and return only 1 result = get best scene in terms of cloud cover
    limit=1,
    sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
)


    print('SEARCH DONE')

    #Try up to 10 times to pull data back relating to the returned scene. 
    item=None
    item_found=0
    for attempt in range(10):
        #Keep retrying if not found yet
        if item_found==0:
            attempt=attempt+1
            try:
                #Iterate to first (only) item
                item = next(search_results.items(),None)
                item_found=1
                print('Item returned')
            except Exception as e:
                print(f"Error returning search item:: {e}")
                #Exponential backoff to prevent API throttling
                wait=2**attempt
                time.sleep(wait)
    
    if item != None:
        print('Tile returned')

    print(item.properties)

    #Save key information from returned item
    try:
        id=item.id
        bbox=item.bbox
        vegetation=item.properties['statistics']['vegetation']
        red=item.assets["B04_10m"].href
        green=item.assets["B03_10m"].href
        blue=item.assets["B02_10m"].href

        print(id)
        print(bbox)
        print(red)
        print(green)
        print(blue)

        return {
            'id': id,
            'bbox': bbox,
            'vegetation':vegetation,
            'red': red,
            'green': green,
            'blue': blue
        }
    except:
        print('Not found')
        return None

#Function for downloading j2 files from the scene identified from CDSE
def download_cdse_file(href, access_key,secret_key, target_dir="."):

    s3_resource = boto3.resource(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    key = href.replace("s3://eodata/", "")
    os.makedirs(target_dir, exist_ok=True)

    local_path = os.path.join(target_dir, os.path.basename(key))

    #Try up to 10 times to download j2 file
    file_downloaded=0
    for attempt in range(10):
        if file_downloaded==0:
            attempt=attempt+1
            try:
                s3_resource.Bucket("eodata").download_file(key, local_path)
                file_downloaded=1
                print('J2 downloaded')
                return local_path
            except Exception as e:
                print(f"Error returning search item:: {e}")
                #Exponential backoff for API throttling
                wait=2**attempt
                time.sleep(wait)
    
    if file_dowloaded==0:
        return None

#Function for combining j2 files into jpeg
def assemble_jpeg(tile):

    image_created=0

    #Get tile name from filepatch using regular expression (tile names all have same format)
    tile_pattern = re.compile(r"_T\d{2}[A-Z]{3}_")
    tile_id=tile_pattern.search(tile).group().strip('_')

    #get list of all j2 files (will only be most recent 3 downloaded)
    files=os.listdir(cwd+'/data/j2_files/')

    #Make sure only 3 j2 files
    if len(files)==3:
        try:
            #Get all bands
            bands = []
            for col_band in ['_B04_','_B03_','_B02_']:
                for file in files:
                    if col_band in file and tile_id in file:
                        with rasterio.open(os.path.join(cwd+'/data/j2_files/', file)) as src:
                            bands.append(src.read(1))  # Read the first (and only) band

            # Stack into an (H, W, N) array
            rgb = np.stack(bands, axis=-1)

            print(rgb.shape)

            rgb=rgb.astype(np.float32)

            #Adjust brightness of image and set correct pixel value bounds
            rgb /= 3000.0  
            rgb = np.clip(rgb,0,1)
            rgb = rgb ** 0.8
            rgb = (rgb*255).astype(np.uint8)

            Image.fromarray(rgb).save(cwd+f"/data/jpeg_files/{tile_id}.jpg")

            image_created=1
        except Exception as e:
            print(f"Error creating jpeg: {e}") 
    
    else:
        print('Not correct number of J2 files, skipping')

    #Remove j2 files as no longer needed
    for file in files:
        os.remove(os.path.join(cwd+'/data/j2_files/', file))
    
    return image_created


#Function for downloading gbif data for a given country
def gbif_download(countrycode, username,passw,emailadd):

    #Create query string for country
    queries = [
        'country = '+str(countrycode),
        'year = 2024',
        'hasCoordinate = TRUE',
        'occurrenceStatus = PRESENT',
        'hasGeospatialIssue = FALSE'

    ]

    #Submit request to GBIF
    request = occ.download(queries=queries,user=username, pwd=passw,email=emailadd)
    #get key of request
    key=request[0]
    print(f'Request created: {key}')

    #Set initial status
    status='Not ready'
    #Create timer
    mins=0

    #Go into wait, depending on status
    while status != 'SUCCEEDED':

        #get metadata
        meta = occ.download_meta(key)

        #Get status
        status = meta["status"]

        print(f"{key}: {status}")
        print('Minutes: '+str(mins))
        mins=mins+1

        #Download if successful
        if status == "SUCCEEDED":
            #Download zip
            zipdownload=occ.download_get(key, path='data/GBIF_downloads')

        #Print error if not
        if status in [
            "FAILED",
            "KILLED",
            "CANCELLED"
        ]:
            raise RuntimeError(
                f"GBIF download failed: {status}"
            )

        #Wait a minute
        time.sleep(60)


    #Get downloaded zip file name
    zip = os.listdir(cwd+'/data/GBIF_downloads')[0]
    print('zip file = '+str(zip))

    # loading the temp.zip and creating a zip object
    with ZipFile(cwd+'/data/GBIF_downloads/'+str(zip), 'r') as zObject:

        # Extracting all the members of the zip into the GBIF downloads file
        zObject.extractall(
            path=cwd+'/data/GBIF_downloads')
        

    #Delete zip file
    os.remove(cwd+'/data/GBIF_downloads/'+str(zip))

#Function to find and cancel any existing gbif jobs
def cancel_gbif_jobs(username,passw):
    downloads=occ.download_list(user=username, pwd=passw)
    print(f'Total previousjobs: {len(downloads)}')
    count=0
    for item in downloads['results']:
        if item['status']=='RUNNING' or item['status']=='PREPARING':
            count=count+1
        occ.download_cancel(item['key'],user=username, pwd=passw)
    print(f'Total running GBIF jobs: {count}')
    downloads=occ.download_list(user=username, pwd=passw)
    count=0
    for item in downloads['results']:
        if item['status']=='RUNNING' or item['status']=='PREPARING':
            count=count+1
    print('GBIF Jobs cancelled')
    print(f'Total GBIF running jobs: {count}')

    
#Function for counting how many occurrence records there are in each country
def create_country_dataframe(world):
    
    #Download file of geojsons for countries 
    print('Creating country observation counts dataframe')

    #Project to equal area projection for accurate area calculations
    world_proj = world.to_crs("EPSG:6933")

    #Create dataframe of country names and codes with number of gbif observations in each
    try:
        countrydf=pd.read_csv(cwd+'data/country_total_records.csv')
    except: 
        countrydf=pd.DataFrame({'country':[],'code':[],'observations':[], 'Area':[]})
        for index, row in world_proj.iterrows():
            try:
                count=occ.count(country = row['ISO_A2'])
            except:
                count=0
            countrydf.loc[len(countrydf)]=[row['SOVEREIGNT'],row['ISO_A2'], count, world_proj.area.iloc[index]]

        print('Country observation counts retrieved')
        countrydf.to_csv(cwd+'/data/country_total_records.csv',index=False)
    
    return countrydf

#Function for spatially joining GBIF occurrences and the MGRS tile grid, to identify tiles of each observation
def gbif_tile_join():

    #Read tile grid
    mgrs = gpd.read_file(cwd+"/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml")
    #Use custom function to extract polygons from kml
    mgrs["geometry"] = mgrs.geometry.apply(get_polygon)
    # remove features where no polygon was found
    mgrs = mgrs.dropna(subset=["geometry"])

    mgrs=mgrs[['Name','geometry']]

    print('MRGS tile shapefile read into dataframe')

    #Filter to a predefined list of gbif columns - rest not needed
    cols=['occurrenceID','kingdom','phylum','order','class','family','genus','species',
                'decimalLatitude','decimalLongitude','dateIdentified', 'eventDate', 'countryCode',
                'coordinateUncertaintyInMeters','recordedBy','identifiedBy',
                'individualCount']

    #read csv into dataframe (tab delimited) in chunks to save memory
    chunk_size = 100_000
    chunks = []
    for chunk in pd.read_csv(cwd+'/data/GBIF_downloads/'+os.listdir(cwd+'/data/GBIF_downloads')[0], sep='\t', chunksize=chunk_size):
        #Filter down to columns, and ensure all columns are present, if not add blank column
        for col in cols:
            if col not in chunk.columns:
                chunk[col] = np.nan
        chunk = chunk[cols]
        chunks.append(chunk)

    gdf = pd.concat(chunks, ignore_index=True)

    print('GBIF data read into dataframe')

    #Clean individual count data - make sure always present with at least count of 1
    try:
        gdf['individualCount']=gdf['individualCount'].fillna(1)
    except:
        gdf['individualCount']=1

    #Convert to geodataframe
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.decimalLongitude, gdf.decimalLatitude), crs="EPSG:4326")

    print('Converted to geodataframe')

    #Join to MGRS dataframe to find which tile each is in (in chunks to avoid memory issues)
    chunks=[]
    for start in range(0, len(gdf), chunk_size):
        print(f"Spatial-joining rows {start} to {start + chunk_size}...")
        chunk = gdf.iloc[start:start + chunk_size]

        joined_chunk = gpd.sjoin(
            chunk,
            mgrs,
            predicate='within',
            how='left'
        )
        chunks.append(joined_chunk)
    #Delete gdf before assembling chunks to save memory
    del gdf
    joined = pd.concat(chunks, ignore_index=True)
    joined.dropna(subset=['Name'],inplace=True)

    print('Spatial join complete')
    print(joined.columns)

    return joined

#Function for calculating biodiversity metrics of tile
def compile_bio_data(df,row,tile):

    new_row=[]

    #Tile
    new_row.append(df.iloc[0]['Name'])
    #bbox
    new_row.append(tile['bbox'])
    #Country
    new_row.append(row['country'])
    #code
    new_row.append(row['code'])
    #vegetation
    new_row.append(tile['vegetation'])

     #Set up tracking whether full sample 
    all_full=0

    #Go through each taxa/filter
    for filter in ['All', 'Plants','Animals','Birds','Insects']:


        print(f'Filter: {filter}')
                    
        #ADD FILTERS CONDITIONALLY
        if filter == 'All':
            filterdf=subdf
        elif filter=='Animals':
            filterdf=subdf[subdf['kingdom']=='Animalia']
        elif filter == 'Plants':
            filterdf=subdf[subdf['kingdom']=='Plantae']
        elif filter == 'Birds':
            filterdf=subdf[subdf['class']=='Aves']
        elif filter=='Insects':
            filterdf=subdf[subdf['class']=='Insecta']
    
        #Record observations before sampling
        occurence_num=len(filterdf)

        #Check if enough data
        if len(filterdf)>occurrences_sample:
            print('Sufficient data')

            #Randomly subsample to 50
            filterdf=filterdf.sample(n=occurrences_sample, random_state=1)


            species_num=filterdf['species'].nunique()
            genus_num=filterdf['genus'].nunique()
            family_num=filterdf['family'].nunique()

            #Custom functions for shannon and simpson
            shannon_num=shannon(filterdf)
            simpson_num=simpson(filterdf)

            new_row.append(occurence_num)
            new_row.append(species_num)
            new_row.append(genus_num)
            new_row.append(family_num)
            new_row.append(shannon_num)
            new_row.append(simpson_num)
            print('data added')

            full=1
        
        else:
            print('insufficient data')
            #Add blanks if less than 50 rows
            new_row.append(occurence_num)
            new_row.append(np.nan)
            new_row.append(np.nan)
            new_row.append(np.nan)
            new_row.append(np.nan)
            new_row.append(np.nan)

            full=0

         #Only return if sample is sufficient
        if filter == 'All' and full==1:
           all_full=1
    if all_full==1:
        return new_row
    else:
        return None

#Check required directory structure exists, if not, create directories

if not os.path.isdir(cwd+'/data'):
    os.makedirs(cwd+'/data')

if not os.path.isdir(cwd+'/data/GBIF_downloads'):
    os.makedirs(cwd+'/data/GBIF_downloads')

if not os.path.isdir(cwd+'/data/gbif_tile_files_saved'):
    os.makedirs(cwd+'/data/gbif_tile_files_saved')

if not os.path.isdir(cwd+'/data/j2_files'):
    os.makedirs(cwd+'/data/j2_files')

if not os.path.isdir(cwd+'/data/jpeg_files'):
    os.makedirs(cwd+'/data/jpeg_files')

if not os.path.isdir(cwd+'/data/Outputs'):
    os.makedirs(cwd+'/data/Outputs')


#Begin full run through


#Import existing gbif output file, if it exists, otherwise create new output dataframe from scratch

try:
    #Try to import existing outputs (use chunks in case large)
    chunk_size = 100000
    chunks = []
    for chunk in pd.read_csv(cwd+'/data/GBIF_data_output.csv', chunksize=chunk_size):
        chunks.append(chunk)
    output = pd.concat(chunks, ignore_index=True)
    #output=pd.read_csv('Outputs/GBIF_data_output.csv')
    print('Existing output file found, continuing to add to it')
except:
    #If not possible, create new output dataframe from scratch
    #Set up output dataframe
    output=pd.DataFrame({'id':[],'bbox':[],'country':[],'code':[],'vegetation':[]})
    #Add columns for biodiveristy metrics of each taxa
    for col in ['All', 'Plants', 'Animals', 'Birds', 'Insects']:
        output[f'{col} occurances']=[]
        output[f'{col} species']=[]
        output[f'{col} genus']=[]   
        output[f'{col} family']=[]
        output[f'{col} shannon']=[]
        output[f'{col} simpson']=[]

    print('Output dataframe set up')

print(f'Output has {len(output.columns)} columns and {len(output)} rows')


#CANCEL ALL EXISTING RUNNING GBIF JOBS
cancel_gbif_jobs(username,passw)


#Create dataframe of countries with count of gbif observations
world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )
#Use predefined function to create dataframe of counts
countrydf=create_country_dataframe(world)
#Remove antarctic
countrydf=countrydf[countrydf['country']!= 'Antarctica']
#Get countries with 10-100 million observations and sort by area to get most tiles
med_countries=countrydf[(countrydf['observations']>10000000)&(countrydf['observations']<100000000)].sort_values(by='Area',ascending=False).reset_index()
print('Medium sized countries identified')
print(med_countries['country'])


#Import shapefile of all mgrs tiles, and convert to polygons
print('Getting shapefile of all mgrs tiles')
mgrs = gpd.read_file(cwd+"/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml")
mgrs["geometry"] = mgrs.geometry.apply(get_polygon)
# remove features where no polygon was found
mgrs = mgrs.dropna(subset=["geometry"])

#Create dictionary for all tiles data
tiles={}

#Set up list of total files saved
saved=len(os.listdir(cwd+'/data/gbif_tile_files_saved'))
print('GBIF tile files saved: '+str(saved))


tiles_checked=0

print('Iterating through countries')
#Go through the identified medium sized countries
for index, row in med_countries.iterrows():
    country_name=row['country']
    country_code=row['code']

    print(country_name)

    country = world[world["NAME"] == country_name]


    #set up count of saved gbif dataframes for country
    country_saved=0

    #check if country completed already
    if country_name not in list(output['country'].unique()):

    
        #Check if already gbif file present 
        if len(os.listdir(cwd+'/data/GBIF_downloads/')) == 0:

            #Download gbif country file for country using predefined function
            print('Downloading GBIF country file', country_code)
            gbif_download(country_code, username,passw,emailadd)
            print('GBIF country file downloaded')
        else:
            print('GBIF country file already present, skipping download')

        #Get geometry of the country
        country.geometry = country.geometry.simplify(
            0.01,
            preserve_topology=True
        )

        print(country.geometry.iloc[0])

        #Find all mgrs tiles that intersect the country, using predefined function
        tiles_in_country=find_tiles_in_country(country, mgrs)
        print('Tiles in country:', len(tiles_in_country))


        #Join tiles to gbif data downloaded for country
        joined=gbif_tile_join()


        #Go through all the discovered tiles in the country...
        for tile in tiles_in_country:

            #Check tile not already in output file
            if tile not in list(output['id'].unique()):

                print('Output dataframe now has '+str(len(output))+' rows')
                tiles_checked=tiles_checked+1
                print(f"{tiles_checked} tiles checked so far")

                print(country_name+': '+ tile)
                #Search the STAC catalogue for the details of the tile
                tiles[tile]=cdse_search(tile)

                if tiles[tile] is not None:

                    #filter gbif mgrs dataframe to just current tile
                    subdf=joined[joined['Name']==tile]
                    print('GBIF records in tile: ', len(subdf))

                    #Proceed only if records in tile
                    if len(subdf)==0:
                        print('No GBIF records in tile, skipping')
                    else:


                        #Do biodiversity calcs and add to output dataframe
                        new_row=compile_bio_data(subdf,row,tiles[tile])
                        print('New row:', new_row)
                        
                        #Only continue if sample is sufficient and bio data is returned
                        if new_row is not None:

                            #see whether gbif file should be saved (all not null and not too many saved already)
                            if all(i is not None for i in new_row) and saved <20 and country_saved<2:
                                subdf.to_csv(cwd+'/data/gbif_tile_files_saved/'+str(row['country']+'_'+str(tile)+'.csv'))
                                saved=saved+1
                                country_saved=country_saved+1


                            #download the jp2 files for tile

                            s3 = boto3.resource(
                                "s3",
                                endpoint_url="https://eodata.dataspace.copernicus.eu",
                                aws_access_key_id=access_key,
                                aws_secret_access_key=secret_key,
                                region_name="default"
                            )

                            #Delete any previously existing j2 files
                            if len(os.listdir(cwd+'/data/j2_files/')) > 0:
                                for item in os.listdir(cwd+'/data/j2_files/'):
                                    os.remove(os.path.join(cwd+'/data/j2_files/', item))

                            #Go through all tiles and assemble images
                            print('Downloading jp2 files for best tiles')
                            download_cdse_file(tiles[tile]['red'], access_key,secret_key, target_dir=cwd+"/data/j2_files/")
                            download_cdse_file(tiles[tile]['green'],access_key,secret_key, target_dir=cwd+"/data/j2_files/")
                            download_cdse_file(tiles[tile]['blue'], access_key,secret_key, target_dir=cwd+"/data/j2_files/")


                            #Combine j2 files into a single jpeg
                            print('Assembling jp2 files into jpeg')
                            if assemble_jpeg(tiles[tile]['id'])==1:
                                print('Jpeg assembled')
                                #Add row to dataframe and save it
                                output.loc[len(output)] = new_row
                                output.to_csv(cwd+'/data/Outputs/GBIF_data_output.csv', index=False)       


                                print('Row added, dataframe now has '+str(len(output))+' rows')


                        
                        print('Insufficient GBIF data in tile, skipping')
                    print('No scenes found, skipping')



                
            
        #Delete gbif download file, once all tiles completed
        downloaded_file = os.listdir(cwd+'/data/GBIF_downloads')[0]
        os.remove(cwd+'/data/GBIF_downloads/'+downloaded_file)

        #delete joined for saving ram
        del joined

print('COMPLETE')




