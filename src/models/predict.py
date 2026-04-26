import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_model(model_path: Path) -> dict:
    """Загружает модель и вспомогательные объекты."""
    bundle = joblib.load(model_path)
    print(f"Модель загружена: {model_path}")
    return bundle


def prepare_features(input_data: dict, target_map: dict, global_mean_m2: float) -> pd.DataFrame:
    """
    Подготавливает признаки для предсказания.
    Честно применяет target encoding через target_map из трейна.
    """
    df = pd.DataFrame([input_data])

    # house_segment нужен для target encoding
    df['house_segment'] = pd.cut(
        df['max_floor'],
        bins=[0, 5, 10, 19, 100],
        labels=['low', 'standard', 'modern', 'high']
    )

    # Target encoding: средняя цена м² по district + house_segment
    df['avg_price_m2_segmented'] = (
        df.set_index(['district', 'house_segment'])
        .index.map(target_map.get)
    )
    df['avg_price_m2_segmented'] = df['avg_price_m2_segmented'].fillna(global_mean_m2)

    # Категориальные признаки
    cat_features = ['district', 'material', 'district_ready', 'mini_disctrict', 'material_age']
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('unknown').replace('nan', 'unknown')

    return df


def predict_price(bundle: dict, input_data: dict) -> float:
    """
    Предсказывает цену квартиры.

    bundle: словарь из joblib.load (model, target_map, global_mean_m2, features)
    input_data: словарь с признаками квартиры
    """
    model         = bundle['model']
    target_map    = bundle['target_map']
    global_mean   = bundle['global_mean_m2']
    features      = bundle['features']

    df = prepare_features(input_data, target_map, global_mean)

    pred_log   = model.predict(df[features])[0]
    pred_price = np.exp(pred_log)
    return pred_price


if __name__ == "__main__":
    model_path = PROJECT_ROOT / 'models' / 'catboost_model.pkl'

    bundle = load_model(model_path)

    # Пример: 2-комнатная квартира в Автозаводском районе
    example_flat = {
        'rooms':              2,
        'total_area':         47.0,
        'year':               1.0,        # лет с постройки
        'kitchen_area':       11.6,
        'current_floor':      7,
        'max_floor':          17,
        'distance_to_center': 9.2,
        'distance_to_metro':  3.5,
        'district':           'Автозаводский район',
        'material':           'панель',
        'mini_disctrict':     'unknown',
        'district_ready':     'Автозаводский район_1.0',
        'material_age':       'панель_False',
        # Производные признаки
        'area_floor_interaction':  47.0 * 7,
        'area_ratio_to_district':  1.0,   # приблизительно среднее по району
    }

    price = predict_price(bundle, example_flat)
    print(f"\nПредсказанная цена: {price:,.0f} ₽  ({price / 1_000_000:.2f} млн ₽)")