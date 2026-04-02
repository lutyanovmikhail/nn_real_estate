import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logger
logger = setup_logger(__name__)
def standardize_gipernn(df: pd.DataFrame) -> pd.DataFrame:
    def room_featuring(text):
        if pd.isna(text):
            return None
        text = str(text).lower()
        if 'студия' in text:
            return 0
        return int(text[0])
    df['rooms'] = df['rooms'].apply(room_featuring) # Форматирование признака "rooms"
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
