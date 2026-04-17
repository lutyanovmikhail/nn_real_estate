from unittest import skip

import pandas as pd
from pathlib import Path
from pandas import isna
from src.utils.logger import setup_logger
logger = setup_logger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
def standardize_gipernn(df: pd.DataFrame) -> pd.DataFrame:
    def parse_rooms(text):
        if pd.isna(text):
            return None
        text = str(text).lower()
        if 'студия' in text:
            return 0
        return int(text[0])
    df['rooms'] = df['rooms'].apply(parse_rooms)
    # Форматирование признака "rooms"
    def parse_squares(text):
        if pd.isna(text):
            return None, None, None
        text = str(text).strip()
        parts = text.split('/')
        if len(parts) < 3:
            return None, None, None
        try:
            def to_float(s): # форматирование значения
                return float(s.strip().replace(',', '.'))
            total_area = to_float(parts[0])
            living_area = to_float(parts[1])
            kitchen_area = to_float(parts[2])
            return total_area, living_area, kitchen_area
        except:
            return None, None, None

    df[['total_area', 'living_area', 'kitchen_area']] = df['squares'].apply(
        lambda x: pd.Series(parse_squares(x))
    ) # Создание признаков "total_area", "living_area", "kitchen_area"
    def parse_floors(text):
        if pd.isna(text):
            return None, None
        text = str(text).split()
        cur = int(text[0].replace('/', ''))
        maxim = int(text[2].replace('эт.', ''))
        return cur, maxim
    df[['current_floor', 'max_floor']] = df['floor'].apply(
        lambda x: pd.Series(parse_floors(x))
    ) # Создание признаков "current_floor", "max_floor"
    def parse_year(text):
        if pd.isna(text):
            return None
        text = int(text)
        if text > 2026:
            return 0
        return 2026 - text
    # Создание признака "years"(возраст здания)
    df['year'] = df['year'].apply(parse_year)
    def parse_price(text):
        text = text.replace('руб.', '').replace(' ', '').replace('\xa0', '')
        return int(text)
    df['price'] = df['price'].apply(parse_price)
    # Форматирование целевого признака цены
    def parse_district(text):
        if pd.isna(text):
            return None
        base = {
            'Нижегородский': 'Нижегородский район',
            'Советский': 'Советский район',
            'Канавинский': 'Канавинский район',
            'Приокский': 'Приокский район',
            'Ленинский': 'Ленинский район',
            'Сормовский': 'Сормовский район',
            'Автозаводский': 'Автозаводский район',
            'Московский': 'Московский район',
        }
        for key in base:
            if key in text:
                return base[key]
    df['district'] = df['district'].apply(parse_district)
    # Форматирование признака районов
    def parse_is_ready(year):
        if year > 2026:
            return 0
        return 1
    df['is_ready'] = df['year'].apply(parse_is_ready)
    # Создание признака готовности здания
    df = df.drop(columns=['squares', 'floor'])
    # Удаление избыточных признаков
    return df
def standardize_cian(df: pd.DataFrame) -> pd.DataFrame:
    def parse_rooms(row):
        for col in ['general1', 'general2']:
            val = row[col]
            if pd.isna(val):
                continue
            text = str(val)
            if 'студия' in text.lower():
                return 0
            if text and text[0].isdigit():
                return int(text[0])
        return None
    df['rooms'] = df[['general1', 'general2']].apply(parse_rooms, axis=1)
    # Форматирование и создание признака rooms
    def parse_total_area(row):
        for col in ['general1', 'general2']:
            text = row[col]
            if pd.isna(text):
                continue
            text = str(text).strip()
            import re
            match = re.search(r'(\d+[\.,]?\d*)\s*м²', text)
            if match:
                area_str = match.group(1).replace(',', '.')
                try:
                    return float(area_str)
                except:
                    continue
        return None
    df['total_area'] = df[['general1', 'general2']].apply(parse_total_area, axis=1)
    # Форматирование и создание признака total_area
    def parse_floors(row):
        for col in ['general1', 'general2']:
            text = row[col]
            if pd.isna(text):
                continue
            text = str(text).strip()
            import re
            match = re.search(r'(\d+)/(\d+)\s*этаж', text)
            if match:
                cur_floor = match.group(1)
                max_floor = match.group(2)
                try:
                    return int(cur_floor), int(max_floor)
                except:
                    continue
        return None, None
    result = df[['general1', 'general2']].apply(parse_floors, axis=1)
    df[['current_floor', 'max_floor']] = pd.DataFrame(result.tolist(), columns=['current_floor', 'max_floor'])
    # Форматирование и создание признаков max_floor и current_floor
    def parse_price(text):
        if pd.isna(text):
            return None
        text = str(text).replace(',', ' ').replace('₽', ' ').replace(' ', '').replace('\xa0', '')
        text = int(text[:-2])
        return text
    df['price'] = df['price'].apply(parse_price)
    # Форматирование и создание признака цены
    def parse_district(text):
        if pd.isna(text):
            return None
        base = {
            'Нижегородский': 'Нижегородский район',
            'Советский': 'Советский район',
            'Канавинский': 'Канавинский район',
            'Приокский': 'Приокский район',
            'Ленинский': 'Ленинский район',
            'Сормовский': 'Сормовский район',
            'Автозаводский': 'Автозаводский район',
            'Московский': 'Московский район',
        }
        for key in base:
            if key in text:
                return base[key]
    df['district'] = df['district'].apply(parse_district)
    # Форматирование признака района
    def parse_metro_time(text):
        if pd.isna(text):
            return None
        import re
        text = str(text).lower()
        match = re.search(r'(\d+)', text)
        if not match:
            return None
        minutes = int(match.group(1))
        if 'пешком' in text:
            return minutes * 1.0
        elif 'транспорт' in text:
            return minutes * 2.5
        else:
            return minutes * 1.0
    df['metro'] = df['metro'].apply(parse_metro_time)
    # Признак дистанции до метро
    def parse_is_ready(text):
        if pd.isna(text):
            return None
        text = str(text).lower()
        if 'дом сдан' in text:
            return 1
        import re
        match = re.search(r'(\d+)\s*кв\.\s*(\d{4})', text)
        if match:
            quarter = int(match.group(1))
            year = int(match.group(2))
            if year > 2026:
                return 0
            elif year == 2026 and quarter >= 2:
                return 0
            else:
                return 1
        return None
    df['is_ready'] = df['if_new'].apply(parse_is_ready)
    # Признак готовности здания
    df = df.drop(columns=['with_something2', 'with_something1', 'general1', 'general2', 'if_new'])
    # удалние избыточных признаков
    return df
giper_nn_df = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'gipernn_02_04.csv',
                          sep=';',
                          on_bad_lines='skip'
                          )
giper_nn_df = giper_nn_df.rename(columns={'disctrict': 'district'})
print("Колонки в файле:", giper_nn_df.columns.tolist())
cian_df = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'cian_02_04.csv',
                      sep=';',
                      on_bad_lines='skip'
                      )
cian_df = cian_df.rename(columns={'disctrict': 'district'})
giper_nn_df = standardize_gipernn(giper_nn_df)
cian_df = standardize_cian(cian_df)
merged_df = pd.concat([giper_nn_df, cian_df], ignore_index=True)
before = len(merged_df)
merged_df = merged_df.drop_duplicates(subset=['total_area', 'current_floor', 'max_floor', 'district', 'price'])
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Удалено дубликатов: {before - len(merged_df)}")
merged_df.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'dataset_before_fe.csv', index=False, encoding='utf-8')
print(f"Итоговы датасет {len(merged_df)} строк, {len(merged_df.columns)} колонок")