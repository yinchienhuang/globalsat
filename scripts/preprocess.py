"""
Process settlement layer

Written by Ed Oughton.

December 2020

"""
import os
import math
import configparser
import json
import math
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, shape, mapping, box
from shapely.ops import unary_union, nearest_points, transform
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterstats import zonal_stats, gen_zonal_stats
from tqdm import tqdm
from IPython.display import display
import random
import pymap3d as pm

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH)
DATA_INTERMEDIATE = os.path.join(BASE_PATH, 'intermediate')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')


def find_country_list(continent_list):
    """
    This function produces country information by continent.

    Parameters
    ----------
    continent_list : list
        Contains the name of the desired continent, e.g. ['Africa']

    Returns
    -------
    countries : list of dicts
        Contains all desired country information for countries in
        the stated continent.

    """
    path = os.path.join(DATA_RAW, 'gadm36_levels_shp', 'gadm36_0.shp')
    countries = gpd.read_file(path)


    glob_info_path = os.path.join(DATA_RAW, 'global_information.csv')
    load_glob_info = pd.read_csv(glob_info_path, encoding = "ISO-8859-1")
    countries = countries.merge(load_glob_info, left_on='GID_0',
        right_on='ISO_3digit')

    if len(continent_list) > 0:
        selected_countries = countries.loc[countries['continent'].isin(continent_list)]
    else:
        selected_countries = countries.loc[countries['global'] == 1]

    countries = []

    for index, country in selected_countries.iterrows():

        countries.append({
            'country_name': country['country'],
            'iso3': country['GID_0'],
            'iso2': country['ISO_2digit'],
            'regional_level': country['gid_region'],
        })

    return countries


def process_country_shapes(country):
    """
    Creates a single national boundary for the desired country.

    Parameters
    ----------
    country : string
        Three digit ISO country code.

    """
    iso3 = country['iso3']

    path = os.path.join(DATA_INTERMEDIATE, iso3)

    if os.path.exists(os.path.join(path, 'national_outline.shp')):
        return 'Completed national outline processing'

    if not os.path.exists(path):
        # print('Creating directory {}'.format(path))
        os.makedirs(path)

    shape_path = os.path.join(path, 'national_outline.shp')

    # print('Loading all country shapes')
    path = os.path.join(DATA_RAW, 'gadm36_levels_shp', 'gadm36_0.shp')
    countries = gpd.read_file(path)

    # print('Getting specific country shape for {}'.format(iso3))
    single_country = countries[countries.GID_0 == iso3]

    # print('Excluding small shapes')
    single_country['geometry'] = single_country.apply(
        exclude_small_shapes, axis=1)

    # print('Adding ISO country code and other global information')
    glob_info_path = os.path.join(DATA_RAW, 'global_information.csv')
    glob_info_path = "data/global_information.csv"
    load_glob_info = pd.read_csv(glob_info_path, encoding = "ISO-8859-1")
    single_country = single_country.merge(
        load_glob_info,left_on='GID_0', right_on='ISO_3digit')

    single_country.to_file(shape_path, driver='ESRI Shapefile')

    return


def process_regions(country):
    """
    Function for processing the lowest desired subnational regions for the
    chosen country.

    Parameters
    ----------
    country : string
        Three digit ISO country code.

    """
    regions = []

    iso3 = country['iso3']
    level = country['regional_level']

    for regional_level in range(1, level + 1):

        filename = 'regions_{}_{}.shp'.format(regional_level, iso3)
        folder = os.path.join(DATA_INTERMEDIATE, iso3, 'regions')
        path_processed = os.path.join(folder, filename)

        if os.path.exists(path_processed):
            continue

        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = 'gadm36_{}.shp'.format(regional_level)
        path_regions = os.path.join(DATA_RAW, 'gadm36_levels_shp', filename)
        regions = gpd.read_file(path_regions)

        regions = regions[regions.GID_0 == iso3]

        regions['geometry'] = regions.apply(exclude_small_shapes, axis=1)
        #print(regions)

        try:
            regions.to_file(path_processed, driver='ESRI Shapefile')
        except:
            pass

    return


def process_settlement_layer(country):
    """
    Clip the settlement layer to the chosen country boundary and place in
    desired country folder.

    Parameters
    ----------
    country : string
        Three digit ISO country code.

    """
    iso3 = country['iso3']

    path_settlements = os.path.join(DATA_RAW,'settlement_layer',
        'ppp_2020_1km_Aggregated.tif')
    path_settlements = "data\ppp_2020_1km_Aggregated.tif"

    settlements = rasterio.open(path_settlements, 'r+')
    settlements.nodata = 255
    settlements.crs = {"init": "epsg:4326"}

    iso3 = country['iso3']
    path_country = os.path.join(DATA_INTERMEDIATE, iso3,
        'national_outline.shp')

    if os.path.exists(path_country):
        country = gpd.read_file(path_country)
    else:
        print('Must generate national_outline.shp first for {}'.format(iso3) )

    path_country = os.path.join(DATA_INTERMEDIATE, iso3)
    shape_path = os.path.join(path_country, 'settlements.tif')

    if os.path.exists(shape_path):
        return

    bbox = country.envelope
    geo = gpd.GeoDataFrame()

    geo = gpd.GeoDataFrame({'geometry': bbox})

    coords = [json.loads(geo.to_json())['features'][0]['geometry']]

    #chop on coords
    out_img, out_transform = mask(settlements, coords, crop=True)

    # Copy the metadata
    out_meta = settlements.meta.copy()

    out_meta.update({"driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "crs": 'epsg:4326'})

    with rasterio.open(shape_path, "w", **out_meta) as dest:
            dest.write(out_img)

    return


def exclude_small_shapes(x):
    """
    Remove small multipolygon shapes.

    Parameters
    ---------
    x : polygon
        Feature to simplify.

    Returns
    -------
    MultiPolygon : MultiPolygon
        Shapely MultiPolygon geometry without tiny shapes.

    """
    # if its a single polygon, just return the polygon geometry
    if x.geometry.geom_type == 'Polygon':
        return x.geometry

    # if its a multipolygon, we start trying to simplify
    # and remove shapes if its too big.
    elif x.geometry.geom_type == 'MultiPolygon':

        area1 = 0.01
        area2 = 50

        # dont remove shapes if total area is already very small
        if x.geometry.area < area1:
            return x.geometry
        # remove bigger shapes if country is really big

        if x['GID_0'] in ['CHL','IDN']:
            threshold = 0.01
        elif x['GID_0'] in ['RUS','GRL','CAN','USA']:
            threshold = 0.01

        elif x.geometry.area > area2:
            threshold = 0.1
        else:
            threshold = 0.001

        # save remaining polygons as new multipolygon for
        # the specific country
        new_geom = []
        for y in x.geometry:
            if y.area > threshold:
                new_geom.append(y)

        return MultiPolygon(new_geom)


def create_pop_regional_lookup(country,online_user_for_sat_sys1, online_user_for_sat_sys2,collaborated = 0):
    """
    Extract regional luminosity and population data.

    Parameters
    ----------

    country : string
        Three digit ISO country code.

    """
    level = country['regional_level']
    iso3 = country['iso3']
    GID_level = 'GID_{}'.format(level)

    filename = 'population_lookup_level_{}.csv'.format(level)
    path_output = os.path.join(DATA_INTERMEDIATE, iso3, filename)

    update_satellite_position_only = 1 
    collaborated = collaborated
    random.seed(9001)

    if os.path.exists(path_output):
        output = pd.read_csv(path_output).to_dict('records')
        if update_satellite_position_only == 1:
            for regions in output:
                area_km = regions['area_km']
                population = regions["population"]
                ct = centroid(float(regions['centroid_lon']), float(regions['centroid_lat']))
                potential_customer_rate = 0.002 #adaption rate
                potential_customer_total = population * potential_customer_rate
                market_share_sys1 = 0.1 #random.random()/10  #randomly generated market share, 1~10%
                market_share_sys2 = 0.1 #random.random()/10

                satellite_position1 = pd.read_csv("data\satellite_position1.csv",dtype=float) #read satellite positiion in latitude and longitude generated randomly
                satellite_position2 = pd.read_csv("data\satellite_position2.csv",dtype=float)

                if collaborated == 1:
                    #Calculate the customer that are not served for each constellation
                    x,left_potential_customer_sys1, x, x = cal_left_customer(satellite_position1, online_user_for_sat_sys1, potential_customer_total, market_share_sys1, ct, 0)
                    x,left_potential_customer_sys2, x, x = cal_left_customer(satellite_position2, online_user_for_sat_sys2, potential_customer_total, market_share_sys2, ct, 0)
                    #Add those customers to another constellation's potential customer set, calculate the serviced customer
                    online_customer_sys2,x,online_user_for_sat_sys2, transfered_customer_sys2 = cal_online_customer(satellite_position2, online_user_for_sat_sys2, potential_customer_total, market_share_sys2, ct, left_potential_customer_sys1)
                    online_customer_sys1,x,online_user_for_sat_sys1, transfered_customer_sys1 = cal_online_customer(satellite_position1, online_user_for_sat_sys1, potential_customer_total, market_share_sys1, ct, left_potential_customer_sys2)

                elif collaborated == 0:
                    #Directly calculate the online customers
                    online_customer_sys1,left_potential_customer_sys1,online_user_for_sat_sys1, transfered_customer_sys1 = cal_online_customer(satellite_position1, online_user_for_sat_sys1, potential_customer_total, market_share_sys1, ct, 0)
                    online_customer_sys2,left_potential_customer_sys2,online_user_for_sat_sys2, transfered_customer_sys2 = cal_online_customer(satellite_position2, online_user_for_sat_sys2, potential_customer_total, market_share_sys2, ct, 0)
                
                regions['market_share_sys1']= market_share_sys1
                regions['online_customer_sys1']= online_customer_sys1
                regions['online_customer_density_sys1']=online_customer_sys1/area_km
                regions['left_potential_customer_sys1']=left_potential_customer_sys1
                regions['transfered_customer_sys1']=transfered_customer_sys1
                regions['market_share_sys2']= market_share_sys2
                regions['online_customer_sys2']= online_customer_sys2
                regions['online_customer_density_sys2']=online_customer_sys2/area_km
                regions['left_potential_customer_sys2']=left_potential_customer_sys2
                regions['transfered_customer_sys2']=transfered_customer_sys2
            #print(output)
            output_pandas = pd.DataFrame(output)
            output_pandas.to_csv(path_output, index=False)
            
        return output,online_user_for_sat_sys1,online_user_for_sat_sys2

    filename = 'settlements.tif'
    path_settlements = os.path.join(DATA_INTERMEDIATE, iso3, filename)

    filename = 'regions_{}_{}.shp'.format(level, iso3)
    folder = os.path.join(DATA_INTERMEDIATE, iso3, 'regions')
    regions = gpd.read_file(os.path.join(folder, filename), crs='epsg:4326')

    output = []

    #When running the code for the first time, generate the population and geometry data
    #Not needed after the first time
    for index, region in regions.iterrows():

        area_km = get_area(region['geometry'])

        population = find_population(region, path_settlements)

        if not isinstance(population, float):
            continue

        if population > 0:
            pop_density_km2 = population / area_km
        else:
            pop_density_km2 = 0

        centroid_lat = region.geometry.centroid.y
        centroid_lon = region.geometry.centroid.x
        ct = centroid(centroid_lon,centroid_lat)
        
        potential_customer_rate = 0.005
        potential_customer_total = population * potential_customer_rate
        market_share_sys1 = random.random()/10 # 1%~10% market share
        market_share_sys2 = random.random()/10


        #Code for single satellite constellation customers
        satellite_position1 = pd.read_csv("data\satellite_position1.csv",dtype=float)
        satellite_position2 = pd.read_csv("data\satellite_position2.csv",dtype=float)

        #print("start regions")
        online_customer_sys1,left_potential_customer_sys1,online_user_for_sat_sys1, transfered_customer_sys1 = cal_online_customer(satellite_position1, online_user_for_sat_sys1, potential_customer_total, market_share_sys1, ct, 0)
        online_customer_sys2,left_potential_customer_sys2,online_user_for_sat_sys2, transfered_customer_sys2 = cal_online_customer(satellite_position2, online_user_for_sat_sys2, potential_customer_total, market_share_sys2, ct, left_potential_customer_sys1)
        online_customer_sys1,left_potential_customer_sys1,online_user_for_sat_sys1, transfered_customer_sys1 = cal_online_customer(satellite_position1, online_user_for_sat_sys1, potential_customer_total, market_share_sys1, ct, left_potential_customer_sys2)
 

        online_customer_density_sys1 = online_customer_sys1/area_km
        online_customer_density_sys2 = online_customer_sys2/area_km

        output.append({
            'iso3': iso3,
            'regions': region[GID_level],
            'population': population,
            'area_km': area_km,
            'pop_density_km2': pop_density_km2,
            'potential_custumer_total': potential_customer_total,
            'centroid_lat' : centroid_lat,
            "centroid_lon" : centroid_lon,

            'market_share_sys1': market_share_sys1,
            'online_customer_sys1': online_customer_sys1,
            'online_customer_density_sys1':online_customer_density_sys1,
            'left_potential_customer_sys1':left_potential_customer_sys1,
            'transfered_customer_sys1':transfered_customer_sys1,
            'market_share_sys2': market_share_sys2,
            'online_customer_sys2': online_customer_sys2,
            'online_customer_density_sys2':online_customer_density_sys2,
            'left_potential_customer_sys2':left_potential_customer_sys2,
            'transfered_customer_sys2':transfered_customer_sys2
        })

    output_pandas = pd.DataFrame(output)

    output_pandas.to_csv(path_output, index=False)

    return output,online_user_for_sat_sys1, online_user_for_sat_sys2

class centroid():
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

def cal_left_customer(satellite_position, online_user_for_sat, potential_customer_total, market_share, centroid, left_potential_customer=0):
    #print("cal_left_customer")
    #potential_customer += left_potential_customer
    potential_customer = potential_customer_total*market_share
    online_customer = 0 #at each regions
    min_elevation_angle = 35
    max_user_count = 10000 #1Gb/100 Kb(per person) =  10000
    transfered_customer = 0

    for i, sat in satellite_position.iterrows():
        #print("i: "+str(i))
        elevation_angle = cal_evevation_angle(sat,centroid) 
        #print("elevation angle: "+str(elevation_angle))
        if elevation_angle>min_elevation_angle:
            #print("satellite "+str(i)+" in range, "+"elevation angle = "+str(elevation_angle))
            if online_user_for_sat[i] + potential_customer + left_potential_customer > max_user_count:
                if online_user_for_sat[i] + potential_customer > max_user_count:
                    potential_customer = float(potential_customer -(max_user_count - online_user_for_sat[i]))
                    online_customer += (max_user_count - float(online_user_for_sat[i]))
                    #print("Add" + str(max_user_count-online_user_for_sat[i]) +" people to satellite" + str(i))
                    transfered_customer += 0
                else:
                    online_customer += (max_user_count - float(online_user_for_sat[i]))
                    transfered_customer += left_potential_customer - (float(online_user_for_sat[i]) + potential_customer + left_potential_customer - max_user_count)
                    left_potential_customer = float(online_user_for_sat[i]) + potential_customer + left_potential_customer - max_user_count
                    potential_customer = 0
            else:
                online_customer += potential_customer + left_potential_customer
                potential_customer = 0
                transfered_customer += left_potential_customer
                left_potential_customer = 0
        #print("online user for satellite"+str(i)+":"+str(online_user_for_sat[i]))
    left_potential_customer += potential_customer
    return online_customer, left_potential_customer, online_user_for_sat, transfered_customer


def cal_online_customer(satellite_position, online_user_for_sat, potential_customer_total, market_share, centroid, left_potential_customer=0):
    #print("left_potential_customer: "+str(left_potential_customer))
    #potential_customer += left_potential_customer
    potential_customer = potential_customer_total*market_share
    online_customer = 0 #at each regions
    min_elevation_angle = 35
    max_user_count = 10000 #1Gb/100 Kb(per person) =  10000
    transfered_customer = 0

    for i, sat in satellite_position.iterrows():
        #print("i: "+str(i))
        elevation_angle = cal_evevation_angle(sat,centroid) 
        #print("elevation angle: "+str(elevation_angle))
        if elevation_angle>min_elevation_angle:
            #print("satellite "+str(i)+" in range, "+"elevation angle = "+str(elevation_angle))
            if online_user_for_sat[i] + potential_customer + left_potential_customer > max_user_count:
                if online_user_for_sat[i] + potential_customer > max_user_count:
                    potential_customer = float(potential_customer -(max_user_count - online_user_for_sat[i]))
                    online_customer += (max_user_count - float(online_user_for_sat[i]))
                    #print("Add" + str(max_user_count-online_user_for_sat[i]) +" people to satellite" + str(i))
                    online_user_for_sat[i] = max_user_count
                    transfered_customer += 0
                else:
                    online_customer += (max_user_count - float(online_user_for_sat[i]))
                    #print("case1")
                    #print("online customer added: "+str((max_user_count - float(online_user_for_sat[i]))) + "people")
                    transfered_customer += left_potential_customer - (float(online_user_for_sat[i]) + potential_customer + left_potential_customer - max_user_count)
                    left_potential_customer = float(online_user_for_sat[i]) + potential_customer + left_potential_customer - max_user_count
                    #print("Add" + str(max_user_count-online_user_for_sat[i]) +" people to satellite" + str(i))
                    online_user_for_sat[i] = max_user_count
                    potential_customer = 0
            else:
                #print("case2")
                online_user_for_sat[i] += potential_customer + left_potential_customer
                #print("online customer added: "+str(potential_customer + left_potential_customer) + " people")
                #print("Add " + str(potential_customer + left_potential_customer) +" people to satellite " + str(i))
                online_customer += potential_customer + left_potential_customer
                potential_customer = 0
                transfered_customer += left_potential_customer
                left_potential_customer = 0
                #print("satellite i user count: "+str(online_user_for_sat[i]))
        #print("online user for satellite"+str(i)+":"+str(online_user_for_sat[i]))
    left_potential_customer += potential_customer
    #print("left_potential_customer: "+str(left_potential_customer))
    return online_customer, left_potential_customer, online_user_for_sat, transfered_customer



def find_population(region, path_settlements):
    """

    """
    with rasterio.open(path_settlements) as src:

        affine = src.transform
        array = src.read(1)
        array[array <= 0] = 0

        population = [d['sum'] for d in zonal_stats(
            region['geometry'], array, stats=['sum'], affine=affine)][0]

    return population


def get_area(modeling_region_geom):
    """
    Return the area in square km.

    """
    project = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True).transform
    new_geom = transform(project, modeling_region_geom)
    area_km = new_geom.area / 1e6

    return area_km

def cal_evevation_angle(sat,centroid):
    sat_altitude = 600000 #km
    # #print(float(sat.latitude),float(sat.longitude),centroid.y,centroid.x)
    # [sat_x,sat_y,sat_z] = lla2ecef(float(sat.latitude), float(sat.longitude), sat_altitude)
    # [gtx,gty,gtz] = lla2ecef(centroid.y, centroid.x, 0)
    # norm_a = math.sqrt(sat_x**2+sat_y**2+sat_z**2)
    # a = [sat_x/norm_a,sat_y/norm_a,sat_z/norm_a]
    # norm_b = math.sqrt(gtx**2+gty**2+gtz**2)
    # b = [gtx/norm_b,gty/norm_b,gtz/norm_b]
    # elevation = 90 -  math.degrees(math.acos(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]))
    # #print(elevation)

    [az,el,range] = pm.geodetic2aer(float(sat.latitude), float(sat.longitude), sat_altitude , centroid.y, centroid.x, 0, deg=True)
    #print(el)
    
    return el

def lla2ecef(lat, lon, alt):
	a = 6378137
	a_sq = a**2
	e = 8.181919084261345e-2
	e_sq = e**2
	b_sq = a_sq*(1 - e_sq)

	lat = np.array([lat]).reshape(np.array([lat]).shape[-1], 1)*np.pi/180
	lon = np.array([lon]).reshape(np.array([lon]).shape[-1], 1)*np.pi/180
	alt = np.array([alt]).reshape(np.array([alt]).shape[-1], 1)

	N = a/np.sqrt(1 - e_sq*np.sin(lat)**2)
	x = (N+alt)*np.cos(lat)*np.cos(lon)
	y = (N+alt)*np.cos(lat)*np.sin(lon)
	z = ((b_sq/a_sq)*N+alt)*np.sin(lat)

	return x, y, z

if __name__ == '__main__':

    online_user_for_sat_sys1 = np.zeros((1000,1)) #max of 1000 satellite, record online user for each satellite
    online_user_for_sat_sys2 = np.zeros((1000,1))
    countries = find_country_list([])#[:2] #['Africa']

    output = []

    for country in tqdm(countries):

        print('-Working on {}: {}'.format(country['country_name'], country['iso3']))

        #process_country_shapes(country)

        #process_regions(country)

        #process_settlement_layer(country)

        [results,online_user_for_sat_sys1,online_user_for_sat_sys2] = create_pop_regional_lookup(country,online_user_for_sat_sys1,online_user_for_sat_sys2,0)

        output = output + results

    path_output = os.path.join(DATA_INTERMEDIATE, 'global_regional_population_lookup.csv')
    output = pd.DataFrame(output)
    output.to_csv(path_output, index=False)

    path_online_user_for_sat_sys1 = os.path.join(DATA_INTERMEDIATE, 'online_user_for_sat_sys1.csv')
    online_user_for_sat_sys1 = pd.DataFrame(online_user_for_sat_sys1)
    online_user_for_sat_sys1.to_csv(path_online_user_for_sat_sys1, index=False)

    path_online_user_for_sat_sys2 = os.path.join(DATA_INTERMEDIATE, 'online_user_for_sat_sys2.csv')
    online_user_for_sat_sys2 = pd.DataFrame(online_user_for_sat_sys2)
    online_user_for_sat_sys2.to_csv(path_online_user_for_sat_sys2, index=False)

    print('Preprocessing1 complete')


    online_user_for_sat_sys1 = np.zeros((1000,1)) #max of 1000 satellite, record online user for each satellite
    online_user_for_sat_sys2 = np.zeros((1000,1))
    countries = find_country_list([])#[:2] #['Africa']

    output = []

    for country in tqdm(countries):

        print('-Working on {}: {}'.format(country['country_name'], country['iso3']))

        #process_country_shapes(country)

        #process_regions(country)

        #process_settlement_layer(country)

        [results,online_user_for_sat_sys1,online_user_for_sat_sys2] = create_pop_regional_lookup(country,online_user_for_sat_sys1,online_user_for_sat_sys2,1)

        output = output + results

    path_output = os.path.join(DATA_INTERMEDIATE, 'global_regional_population_lookup_col.csv')
    output = pd.DataFrame(output)
    output.to_csv(path_output, index=False)

    path_online_user_for_sat_sys1 = os.path.join(DATA_INTERMEDIATE, 'online_user_for_sat_sys1_col.csv')
    online_user_for_sat_sys1 = pd.DataFrame(online_user_for_sat_sys1)
    online_user_for_sat_sys1.to_csv(path_online_user_for_sat_sys1, index=False)

    path_online_user_for_sat_sys2 = os.path.join(DATA_INTERMEDIATE, 'online_user_for_sat_sys2_col.csv')
    online_user_for_sat_sys2 = pd.DataFrame(online_user_for_sat_sys2)
    online_user_for_sat_sys2.to_csv(path_online_user_for_sat_sys2, index=False)

    print('Preprocessing2 complete') 