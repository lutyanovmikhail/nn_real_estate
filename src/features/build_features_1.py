# src/features/build_features.py
"""
Feature engineering pipeline for real estate price prediction.
"""
import pandas as pd
import numpy as np
import json
import math
from pathlib import Path

# Определяем корень проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Константы
CENTER = (56.328674, 44.002102)
METRO_STATIONS = {
    'Горьковская': (56.3180, 44.0040),
    'Московская': (56.3225, 43.9461),
    'Ленинская': (56.3000, 43.9450),
    'Пролетарская': (56.2800, 43.9150),
    'Двигатель Революции': (56.2635, 43.9090),
    'Заречная': (56.2979, 43.9235),
    'Канавинская': (56.3170, 43.9505),
    'Стрелка': (56.3350, 43.9760),
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками на сфере (км)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def min_metro_distance(lat: float, lon: float) -> float:
    """Минимальное расстояние до любой станции метро"""
    if pd.isna(lat) or pd.isna(lon):
        return None
    return min(haversine(lat, lon, s_lat, s_lon) for s_lat, s_lon in METRO_STATIONS.values())


def create_features(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Создаёт все признаки для модели"""
    df = df.copy()

    # Этажные признаки
    df['is_last_floor'] = (df['current_floor'] == df['max_floor']).astype(int)
    df['is_first_floor'] = (df['current_floor'] == 1).astype(int)
    df['floor_ratio'] = df['current_floor'] / df['max_floor']

    # Площадные признаки
    df['living_ratio'] = df['living_area'] / df['total_area']
    df['kitchen_ratio'] = df['kitchen_area'] / df['total_area']
    df['area_floor_interaction'] = df['total_area'] * df['current_floor']

    # Районные и категориальные
    df['district_ready'] = df['district'] + '_' + df['is_ready'].astype(str)
    df['material_age'] = df['material'] + '_' + (df['year'] > 50).astype(str)

    # Относительная площадь по району
    district_mean_area = df.groupby('district')['total_area'].transform('mean')
    df['area_ratio_to_district'] = df['total_area'] / district_mean_area

    # Категория этажа
    df['floor_category'] = pd.cut(
        df['current_floor'] / df['max_floor'],
        bins=[0, 0.33, 0.66, 1],
        labels=['low', 'mid', 'high']
    )

    # Гео-признаки (загрузка координат)
    coords_path = PROJECT_ROOT / 'data' / 'external' / 'address_coords.json'
    with open(coords_path, 'r') as f:
        coords_dict = json.load(f)

    df['lat'] = df['address'].map(lambda x: coords_dict.get(x, (None, None))[0])
    df['lon'] = df['address'].map(lambda x: coords_dict.get(x, (None, None))[1])

    # Расчёт расстояний
    df['distance_to_center'] = df.apply(
        lambda row: haversine(row['lat'], row['lon'], CENTER[0], CENTER[1])
        if pd.notna(row['lat']) else None, axis=1
    )
    df['distance_to_metro'] = df.apply(
        lambda row: min_metro_distance(row['lat'], row['lon']), axis=1
    )

    # Заполняем пропуски в гео-признаках (для адресов без координат)
    geo_cols = ['distance_to_center', 'distance_to_metro']
    for col in geo_cols:
        if df[col].isna().any():
            print(f"⚠️ {df[col].isna().sum()} пропусков в {col}, заполняем медианой")
            df[col] = df[col].fillna(df[col].median())

    # Удаляем временные колонки
    df = df.drop(columns=['adress', 'lat', 'lon'])

    if save:
        output_path = PROJECT_ROOT / 'data' / 'processed' / 'dataset_with_geo.csv'
        df.to_csv(output_path, index=False)
        print(f"✅ Сохранено в {output_path}")

    return df


if __name__ == "__main__":
    input_path = PROJECT_ROOT / 'data' / 'processed' / 'dataset_before_fe.csv'
    df = pd.read_csv(input_path)
    print(f"Загружено {len(df)} строк")
    df_fe = create_features(df)
    print(f"Создано {len(df_fe.columns)} признаков")