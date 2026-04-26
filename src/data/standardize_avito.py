import pandas as pd
import numpy as np
import re
import json
import math
import time
from pathlib import Path
from geopy.geocoders import Nominatim

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ─── КООРДИНАТЫ СТАНЦИЙ МЕТРО НН ────────────────────────────────────────────
METRO_COORDS = {
    'Горьковская':           (56.3180, 44.0040),
    'Московская':            (56.3225, 43.9461),
    'Ленинская':             (56.3000, 43.9450),
    'Пролетарская':          (56.2800, 43.9150),
    'Двигатель Революции':   (56.2635, 43.9090),
    'Заречная':              (56.2979, 43.9235),
    'Канавинская':           (56.3170, 43.9505),
    'Стрелка':               (56.3350, 43.9760),
    'Парк Культуры':         (56.2950, 43.9000),
    'Буревестник':           (56.2720, 43.8850),
    'Автозаводская':         (56.2480, 43.8700),
    'Бурнаковская':          (56.3310, 43.8990),
    'Чкаловская':            (56.3290, 44.0180),
    'Кировская':             (56.3110, 43.9250),
    'Комсомольская':         (56.3050, 43.9350),
}

CENTER = (56.328674, 44.002102)

DISTRICT_KEYWORDS = {
    'Нижегородский': 'Нижегородский район',
    'Советский':     'Советский район',
    'Канавинский':   'Канавинский район',
    'Приокский':     'Приокский район',
    'Ленинский':     'Ленинский район',
    'Сормовский':    'Сормовский район',
    'Автозаводский': 'Автозаводский район',
    'Московский':    'Московский район',
}

# ─── 1. ЗАГРУЗКА ─────────────────────────────────────────────────────────────
def load_avito(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines='skip')
    df.columns = ['price_raw', 'main_info', 'address_part1',
                  'tag1', 'tag2', 'street', 'house_num',
                  'metro_station', 'date']
    print(f"Загружено: {len(df)} строк")
    return df

# ─── 2. СТАНДАРТИЗАЦИЯ ───────────────────────────────────────────────────────
def standardize_avito(df: pd.DataFrame) -> pd.DataFrame:

    # ЦЕНА
    def parse_price(text):
        if pd.isna(text): return None
        cleaned = re.sub(r'[^\d]', '', str(text))
        return int(cleaned) if cleaned else None

    # КОМНАТЫ, ПЛОЩАДЬ, ЭТАЖ из колонки main_info
    def parse_main(text):
        if pd.isna(text): return None, None, None, None
        text = str(text)

        # Комнаты: "2-к. квартира" или "Студия"
        rooms = None
        if 'студия' in text.lower():
            rooms = 0
        else:
            m = re.search(r'(\d+)-к\.?', text)
            if m: rooms = int(m.group(1))

        # Площадь: "41,1 м²"
        total_area = None
        m = re.search(r'(\d+[\.,]\d*)\s*м²', text)
        if m: total_area = float(m.group(1).replace(',', '.'))

        # Этаж: "1/4 эт."
        current_floor, max_floor = None, None
        m = re.search(r'(\d+)/(\d+)\s*эт', text)
        if m:
            current_floor = int(m.group(1))
            max_floor = int(m.group(2))

        return rooms, total_area, current_floor, max_floor

    # АДРЕС: col2 если не ",", иначе col5 + col6
    def parse_address(row):
        a = str(row['address_part1']).strip()
        if a in (',', 'nan', ''):
            street  = str(row['street']).strip()   if pd.notna(row['street'])   else ''
            house   = str(row['house_num']).strip() if pd.notna(row['house_num']) else ''
            combined = f"{street} {house}".strip()
            return combined if combined else None
        return a

    # Применяем
    df['price'] = df['price_raw'].apply(parse_price)

    parsed = df['main_info'].apply(
        lambda x: pd.Series(
            parse_main(x),
            index=['rooms', 'total_area', 'current_floor', 'max_floor']
        )
    )
    df = pd.concat([df, parsed], axis=1)
    df['address']       = df.apply(parse_address, axis=1)
    df['metro_name']    = df['metro_station'].where(df['metro_station'].notna())

    # Признаки которых нет в авито — заполняем unknown/nan
    df['material']      = 'unknown'
    df['year']          = np.nan
    df['living_area']   = np.nan
    df['kitchen_area']  = np.nan

    # Убираем сырые колонки
    df = df.drop(columns=['price_raw', 'main_info', 'address_part1',
                           'tag1', 'tag2', 'street', 'house_num',
                           'metro_station', 'date'])

    # Фильтр — только строки с ценой и площадью
    before = len(df)
    df = df.dropna(subset=['price', 'total_area']).reset_index(drop=True)
    print(f"После фильтрации: {len(df)} строк (убрано {before - len(df)} без цены/площади)")

    return df

# ─── 3. ГЕОКОДИНГ ────────────────────────────────────────────────────────────
def geocode_addresses(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:

    if cache_path.exists():
        with open(cache_path, 'r') as f:
            coords_dict = json.load(f)
    else:
        coords_dict = {}

    geolocator   = Nominatim(user_agent="nn_real_estate_avito")
    new_addresses = [
        addr for addr in df['address'].dropna().unique()
        if addr not in coords_dict
    ]
    print(f"В кэше: {len(coords_dict)}, новых для геокодинга: {len(new_addresses)}")
    print(f"Примерное время: {len(new_addresses) * 1.1 / 60:.0f} минут")

    for i, addr in enumerate(new_addresses):
        try:
            location = geolocator.geocode(
                f"{addr}, Нижний Новгород", timeout=10
            )
            coords_dict[addr] = (
                (location.latitude, location.longitude) if location
                else (None, None)
            )
        except Exception:
            coords_dict[addr] = (None, None)

        # Сохраняем кэш каждые 100 адресов
        if i % 100 == 0:
            print(f"  {i}/{len(new_addresses)} адресов обработано...")
            with open(cache_path, 'w') as f:
                json.dump(coords_dict, f, ensure_ascii=False)

        time.sleep(1.1)

    with open(cache_path, 'w') as f:
        json.dump(coords_dict, f, ensure_ascii=False)

    df['lat'] = df['address'].map(
        lambda x: coords_dict.get(x, (None, None))[0]
    )
    df['lon'] = df['address'].map(
        lambda x: coords_dict.get(x, (None, None))[1]
    )

    found = df['lat'].notna().sum()
    print(f"Координаты найдены: {found}/{len(df)} ({found/len(df)*100:.1f}%)")
    return df

# ─── 4. GEO-ПРИЗНАКИ ─────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2-lat1)/2)**2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2-lon1)/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:

    df['distance_to_center'] = df.apply(
        lambda r: haversine(r['lat'], r['lon'], *CENTER)
        if pd.notna(r['lat']) else None, axis=1
    )

    # distance_to_metro: если есть название станции — берём расстояние до неё
    # иначе — минимум до всех станций
    def metro_distance(row):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            if pd.notna(row['metro_name']) and row['metro_name'] in METRO_COORDS:
                s = METRO_COORDS[row['metro_name']]
                return haversine(row['lat'], row['lon'], s[0], s[1])
            else:
                return min(
                    haversine(row['lat'], row['lon'], s[0], s[1])
                    for s in METRO_COORDS.values()
                )
        return None

    df['distance_to_metro'] = df.apply(metro_distance, axis=1)
    df = df.drop(columns=['lat', 'lon', 'metro_name'])
    return df

# ─── 5. РАЙОН ────────────────────────────────────────────────────────────────
def add_district(df: pd.DataFrame) -> pd.DataFrame:

    def parse_district(text):
        if pd.isna(text): return 'unknown'
        for key, val in DISTRICT_KEYWORDS.items():
            if key in str(text):
                return val
        return 'unknown'

    df['district'] = df['address'].apply(parse_district)

    known = (df['district'] != 'unknown').sum()
    print(f"Район из адреса: {known}/{len(df)} ({known/len(df)*100:.1f}%)")
    print(f"Останется 'unknown': {len(df) - known} — определятся через геокодинг")

    return df

# ─── 6. FEATURE ENGINEERING (как в основном пайплайне) ───────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df['is_ready']     = 1  # авито — вторичка и новостройки без даты сдачи
    df['mini_disctrict'] = 'unknown'

    df['kitchen_ratio'] = df['kitchen_area'] / df['total_area']

    df['area_floor_interaction'] = df['total_area'] * df['current_floor']

    district_mean = df.groupby('district')['total_area'].transform('mean')
    df['area_ratio_to_district'] = df['total_area'] / district_mean

    df['district_ready'] = df['district'] + '_' + df['is_ready'].astype(str)
    df['material_age']   = df['material'] + '_False'  # год неизвестен

    df['living_ratio'] = df['living_area'] / df['total_area']
    df['floor_ratio']  = df['current_floor'] / df['max_floor']

    df['is_last_floor']  = (df['current_floor'] == df['max_floor']).astype(int)
    df['is_first_floor'] = (df['current_floor'] == 1).astype(int)

    df['floor_category'] = pd.cut(
        df['current_floor'] / df['max_floor'],
        bins=[0, 0.33, 0.66, 1],
        labels=['low', 'mid', 'high']
    )

    return df

# ─── 7. ЗАПУСК ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_path    = PROJECT_ROOT / 'data' / 'raw'       / 'avito.csv'
    cache_path  = PROJECT_ROOT / 'data' / 'external'  / 'address_coords.json'
    output_path = PROJECT_ROOT / 'data' / 'processed' / 'avito_processed.csv'

    df = load_avito(raw_path)
    df = standardize_avito(df)
    df = add_district(df)
    df = geocode_addresses(df, cache_path)
    df = add_geo_features(df)
    df = add_features(df)

    # Финальная статистика
    print("\n" + "="*50)
    print("ИТОГ")
    print("="*50)
    print(f"Строк: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")
    for col in ['price', 'total_area', 'rooms', 'current_floor',
                'distance_to_center', 'distance_to_metro', 'district']:
        filled = df[col].notna().sum()
        print(f"  {col}: {filled}/{len(df)} ({filled/len(df)*100:.1f}%)")

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nСохранено: {output_path}")