import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import requests
from time import sleep
import re
import json
import math
PROJECT_ROOT = Path('/Users/apch/Pycharmproekti/nn_real_estate')
df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'dataset_before_fe.csv')
print(df.columns)
df['is_last_floor'] = (df['current_floor'] == df['max_floor']).astype(int) # Создание признака последнего этажа
df['is_first_floor'] = (df['current_floor'] == 1).astype(int) # Создание признака первого этажа
df['living_ratio'] = df['living_area'] / df['total_area'] # Создание признака отношения жилой площади
df['kitchen_ratio'] = df['kitchen_area'] / df['total_area'] # Соаздине признака отношения кухонной площади
df['floor_ratio'] = df['current_floor'] / df['max_floor'] # На каком этаже относительно общего количества
unique_addresses = df['adress'].dropna().unique()
print(f"Уникальных адресов: {len(unique_addresses)}")
coords_dict = {}
df['area_floor_interaction'] = df['total_area'] * df['current_floor']
df['district_ready'] = df['district'] + '_' + df['is_ready'].astype(str)
df['material_age'] = df['material'] + '_' + (df['year'] > 50).astype(str)
district_mean_area = df.groupby('district')['total_area'].transform('mean')
df['area_ratio_to_district'] = df['total_area'] / district_mean_area
df['floor_category'] = pd.cut(
    df['current_floor'] / df['max_floor'],
    bins=[0, 0.33, 0.66, 1],
    labels=['low', 'mid', 'high']
)
coords_path = Path('/Users/apch/Pycharmproekti/nn_real_estate/data/external/address_coords.json')

with open(coords_path, 'r') as f:
    coords_dict = json.load(f)

print(f"Загружено {len(coords_dict)} адресов")
df['lat'] = df['adress'].map(lambda x: coords_dict.get(x, (None, None))[0])
df['lon'] = df['adress'].map(lambda x: coords_dict.get(x, (None, None))[1])
found = df['lat'].notna().sum()
print(f"Найдено координат для {found} / {len(df)} адресов ({found/len(df)*100:.1f}%)")
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c
CENTER = (56.328674, 44.002102)
df['distance_to_center'] = df.apply(
    lambda row: haversine(row['lat'], row['lon'], CENTER[0], CENTER[1])
    if pd.notna(row['lat']) else None, axis=1
)
metro_stations = {
    'Горьковская': (56.3180, 44.0040),
    'Московская': (56.3225, 43.9461),
    'Ленинская': (56.3000, 43.9450),
    'Пролетарская': (56.2800, 43.9150),
    'Двигатель Революции': (56.2635, 43.9090),
    'Заречная': (56.2979, 43.9235),
    'Канавинская': (56.3170, 43.9505),
    'Стрелка': (56.3350, 43.9760),
}
def min_metro_distance(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    return min(haversine(lat, lon, s_lat, s_lon) for s_lat, s_lon in metro_stations.values())

df['distance_to_metro'] = df.apply(
    lambda row: min_metro_distance(row['lat'], row['lon']), axis=1
)
print((df.columns))
df = df.drop(columns=['adress', 'lat', 'lon'])
df.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'dataset_with_geo', index=False)