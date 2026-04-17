import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_model(model_path: Path):
    """Загружает сохранённую модель"""
    model = joblib.load(model_path)
    print(f" Модель загружена: {model_path}")
    return model


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Подготавливает признаки для предсказания (как в train_model)"""
    df = df.copy()

    # Создаём признаки, если их нет
    if 'is_center' not in df.columns and 'distance_to_center' in df.columns:
        df['is_center'] = (df['distance_to_center'] < 1.5).astype(int)

    if 'is_near_metro' not in df.columns and 'distance_to_metro' in df.columns:
        df['is_near_metro'] = (df['distance_to_metro'] < 1).astype(int)

    features = [
        'rooms', 'total_area', 'year', 'living_area', 'kitchen_area',
        'current_floor', 'max_floor', 'is_center', 'is_near_metro',
        'kitchen_ratio', 'area_floor_interaction', 'area_ratio_to_district',
        'distance_to_center', 'distance_to_metro',
        'district', 'material', 'mini_disctrict', 'district_ready', 'material_age', 'metro'
    ]

    categorical_features = [
        'district', 'material', 'mini_disctrict', 'district_ready', 'material_age', 'metro'
    ]

    X = df[features].copy()

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna('unknown')

    return X


def predict_price(model, input_data: dict) -> float:
    """
    Предсказывает цену квартиры.

    input_data: словарь с признаками квартиры
    """
    df_input = pd.DataFrame([input_data])
    X = prepare_features_for_prediction(df_input)
    pred_log = model.predict(X)[0]
    pred_price = np.exp(pred_log)
    return pred_price


if __name__ == "__main__":
    # Пути
    model_path = PROJECT_ROOT / 'models' / 'catboost_model.pkl'

    # Загружаем модель
    model = load_model(model_path)

    # Пример входных данных
    example_flat = {
        'rooms': 2,
        'total_area': 47.0,
        'year': 2026,
        'living_area': 28.0,
        'kitchen_area': 11.5,
        'current_floor': 7,
        'max_floor': 17,
        'distance_to_center': 19.2,
        'distance_to_metro': 8,
        'district': 'Автозаводский район',
        'material': 'панель',
        'mini_disctrict': 'unknown',
        'district_ready': 'Автозаводский район_0',
        'material_age': 'кирпич_0',
        'metro': 8,
        'kitchen_ratio': 11.5 / 47.0,
        'area_floor_interaction': 47.0 * 7,
        'area_ratio_to_district': 1.0  # приблизительно
    }

    price = predict_price(model, example_flat)
    print(f"\n Предсказанная цена: {price:,.0f} ₽ ({price / 1_000_000:.2f} млн ₽)")