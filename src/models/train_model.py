import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# Определяем корень проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f" Загружено {len(df)} строк, {len(df.columns)} колонок")
    return df


def prepare_features(df: pd.DataFrame):
    """Подготавливает X, y и категориальные признаки"""

    if 'is_center' not in df.columns:
        df['is_center'] = (df['distance_to_center'] < 1.5).astype(int)
        print("➕ Создан признак 'is_center'")

    if 'is_near_metro' not in df.columns:
        df['is_near_metro'] = (df['distance_to_metro'] < 1).astype(int)
        print("➕ Создан признак 'is_near_metro'")

    # Логарифмируем цену (для нормализации)
    df['price_log'] = np.log(df['price'])

    # Список признаков
    features = [
        'rooms', 'total_area', 'year', 'living_area', 'kitchen_area',
        'current_floor', 'max_floor', 'is_center', 'is_near_metro',
        'kitchen_ratio', 'area_floor_interaction', 'area_ratio_to_district',
        'distance_to_center', 'distance_to_metro',
        'district', 'material', 'mini_disctrict', 'district_ready', 'material_age', 'metro'
    ]

    # Категориальные признаки
    categorical_features = [
        'district', 'material', 'mini_disctrict', 'district_ready', 'material_age', 'metro'
    ]

    # Проверяем, какие категориальные признаки реально существуют
    categorical_features = [col for col in categorical_features if col in df.columns]

    X = df[features].copy()
    y = df['price_log']

    #
    for col in categorical_features:
        X[col] = X[col].astype(str).fillna('unknown')

    print(f" Признаки: {len(features)}")
    print(f" Категориальные: {categorical_features}")

    return X, y, categorical_features


def train_model(X, y, categorical_features, model_path: Path):
    """Обучает CatBoostRegressor и сохраняет модель"""

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Веса для дорогих квартир
    y_train_price = np.exp(y_train)
    weights = np.where(y_train_price > 20_000_000, 2.0, 1.0)
    weights = np.where(y_train_price < 7_000_000, 1.2, weights)

    # Модель
    model = CatBoostRegressor(
        cat_features=categorical_features,
        depth=5,
        iterations=600,
        learning_rate=0.06,
        l2_leaf_reg=5,
        random_seed=42,
        loss_function='Quantile',
        verbose=100
    )

    # Обучение
    print("\n Начинаем обучение...")
    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=(X_test, y_test),
        verbose=100,
        plot=False
    )

    # Предсказания
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)

    y_train_pred = np.exp(y_train_pred_log)
    y_test_pred = np.exp(y_test_pred_log)
    y_train_true = np.exp(y_train)
    y_test_true = np.exp(y_test)

    # Метрики
    r2_train = r2_score(y_train_true, y_train_pred)
    r2_test = r2_score(y_test_true, y_test_pred)
    mae_test = mean_absolute_error(y_test_true, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test_true, y_test_pred) * 100

    print("\n" + "=" * 50)
    print(" РЕЗУЛЬТАТЫ МОДЕЛИ")
    print("=" * 50)
    print(f"R² train: {r2_train:.4f}")
    print(f"R² test:  {r2_test:.4f}")
    print(f"Разница:  {r2_train - r2_test:.4f}")
    print(f"MAE test: {mae_test:,.0f} ₽")
    print(f"MAPE test: {mape_test:.2f}%")

    # Сохраняем модель
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n Модель сохранена: {model_path}")

    return model, r2_test, mape_test


if __name__ == "__main__":
    # Пути
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'dataset_before_fe.csv'
    model_path = PROJECT_ROOT / 'models' / 'catboost_model.pkl'

    # Запуск
    df = load_data(data_path)
    X, y, cat_features = prepare_features(df)
    model, r2, mape = train_model(X, y, cat_features, model_path)

    print("\n Готово! Модель можно использовать в predict_model.py")