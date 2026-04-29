import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

PROJECT_ROOT = Path(__file__).parent.parent.parent

CAT_FEATURES = ['district', 'material', 'district_ready', 'mini_disctrict', 'material_age']

BASE_FEATURES = [
    'rooms', 'total_area', 'year', 'kitchen_area',
    'current_floor', 'max_floor',
    'area_floor_interaction', 'area_ratio_to_district',
    'distance_to_center', 'distance_to_metro'
] + CAT_FEATURES

FINAL_FEATURES = BASE_FEATURES + ['avg_price_m2_segmented']


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка, фильтрация выбросов, базовый feature engineering."""

    # Убираем unknown районы
    df = df[df['district'] != 'unknown'].copy().reset_index(drop=True)

    # Сегмент дома по этажности (нужен для target encoding)
    df['house_segment'] = pd.cut(
        df['max_floor'],
        bins=[0, 5, 10, 19, 100],
        labels=['low', 'standard', 'modern', 'high']
    )

    # Чистим категориальные признаки
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str).fillna('unknown').replace('nan', 'unknown')

    # Убираем выбросы (> 99-го перцентиля)
    price_upper = df['price'].quantile(0.99)
    before = len(df)
    df = df[df['price'] <= price_upper].copy().reset_index(drop=True)
    print(f"Убрано выбросов: {before - len(df)} (>{price_upper/1e6:.1f} млн ₽)")

    return df


def add_target_encoding(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train_log: pd.Series) -> tuple:
    """Честный target encoding: среднее цена/м^2 по district+house_segment.
    Считается только на трейне, применяется к обеим выборкам.
    """
    train_temp = X_train.copy()
    train_temp['real_price'] = np.exp(y_train_log)
    train_temp['m2_price'] = train_temp['real_price'] / train_temp['total_area']

    target_map = (
        train_temp
        .groupby(['district', 'house_segment'], observed=True)['m2_price']
        .mean()
        .to_dict()
    )
    global_mean_m2 = train_temp['m2_price'].mean()

    for df_part in [X_train, X_test]:
        df_part['avg_price_m2_segmented'] = (
            df_part
            .set_index(['district', 'house_segment'])
            .index.map(target_map.get)
        )
        df_part['avg_price_m2_segmented'] = (
            df_part['avg_price_m2_segmented'].fillna(global_mean_m2)
        )

    return X_train, X_test, target_map, global_mean_m2


def train_model(df: pd.DataFrame, model_path: Path):
    """Обучает финальную модель и сохраняет её вместе с target_map."""

    # Разделение ДО target encoding — исключаем утечку
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        df.drop('price_log', axis=1),
        df['price_log'],
        test_size=0.2,
        random_state=42
    )

    # Target encoding
    X_train, X_test, target_map, global_mean_m2 = add_target_encoding(
        X_train, X_test, y_train_log
    )

    # Обучение
    model = CatBoostRegressor(
        iterations=1135,
        depth=8,
        learning_rate=0.06,
        l2_leaf_reg=27,
        random_seed=36,
        cat_features=CAT_FEATURES,
        loss_function='MAPE',
        eval_metric='MAPE',
        verbose=100
    )

    print("\nНачинаем обучение...")
    model.fit(X_train[FINAL_FEATURES], y_train_log)

    # Метрики
    preds_real = np.exp(model.predict(X_test[FINAL_FEATURES]))
    y_test_real = np.exp(y_test_log)
    preds_train = np.exp(model.predict(X_train[FINAL_FEATURES]))
    y_train_real = np.exp(y_train_log)

    r2_train = r2_score(y_train_real, preds_train)
    r2_test  = r2_score(y_test_real, preds_real)
    mae      = mean_absolute_error(y_test_real, preds_real)
    mape     = mean_absolute_percentage_error(y_test_real, preds_real) * 100

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ МОДЕЛИ")
    print("=" * 50)
    print(f"R² train: {r2_train:.4f}")
    print(f"R² test:  {r2_test:.4f}")
    print(f"Gap:      {r2_train - r2_test:.4f}")
    print(f"MAE:      {mae:,.0f} ₽")
    print(f"MAPE:     {mape:.2f}%")

    # Feature importance
    fi_df = pd.DataFrame({
        'feature': FINAL_FEATURES,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    print("\nTop-10 признаков:")
    print(fi_df.head(10).to_string(index=False))

    # Сохраняем модель + target_map (нужен для predict_model.py)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': model,
        'target_map': target_map,
        'global_mean_m2': global_mean_m2,
        'features': FINAL_FEATURES,
        'cat_features': CAT_FEATURES
    }, model_path)
    print(f"\nМодель сохранена: {model_path}")

    return model, target_map, global_mean_m2


def cross_validate(df: pd.DataFrame):
    """5-fold Stratified CV для честной оценки качества."""

    y_binned = pd.qcut(df['price_log'], q=10, labels=False)
    strata   = y_binned.astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores, mae_scores, mape_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, strata)):
        train_df = df.iloc[train_idx].copy()
        val_df   = df.iloc[val_idx].copy()

        y_train_cv = train_df['price_log']
        y_val_cv   = val_df['price_log']

        # Target encoding внутри фолда
        train_df['m2_price'] = np.exp(y_train_cv) / train_df['total_area']
        target_map_cv = (
            train_df
            .groupby(['district', 'house_segment'], observed=True)['m2_price']
            .mean()
            .to_dict()
        )
        global_m2_cv = train_df['m2_price'].mean()

        for part in [train_df, val_df]:
            part['avg_price_m2_segmented'] = (
                part.set_index(['district', 'house_segment'])
                .index.map(target_map_cv.get)
            )
            part['avg_price_m2_segmented'] = (
                part['avg_price_m2_segmented'].fillna(global_m2_cv)
            )

        model_cv = CatBoostRegressor(
            iterations=1135, depth=8, learning_rate=0.06,
            l2_leaf_reg=27, random_seed=42,
            cat_features=CAT_FEATURES,
            loss_function='MAPE', eval_metric='MAPE',
            verbose=False
        )
        model_cv.fit(train_df[FINAL_FEATURES], y_train_cv)

        preds_val   = np.exp(model_cv.predict(val_df[FINAL_FEATURES]))
        preds_train = np.exp(model_cv.predict(train_df[FINAL_FEATURES]))
        y_true_val  = np.exp(y_val_cv)
        y_true_tr   = np.exp(y_train_cv)

        r2_tr = r2_score(y_true_tr, preds_train)
        r2_v  = r2_score(y_true_val, preds_val)
        print(f"Fold {fold}: train {r2_tr:.3f} / val {r2_v:.3f} / gap {r2_tr-r2_v:.3f}")

        r2_scores.append(r2_v)
        mae_scores.append(mean_absolute_error(y_true_val, preds_val))
        mape_scores.append(mean_absolute_percentage_error(y_true_val, preds_val) * 100)

    print("-" * 40)
    print(f"R²:   {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"MAE:  {np.mean(mae_scores)/1e6:.2f} ± {np.std(mae_scores)/1e6:.2f} млн ₽")
    print(f"MAPE: {np.mean(mape_scores):.2f} ± {np.std(mape_scores):.2f}%")


if __name__ == "__main__":
    data_path  = PROJECT_ROOT / 'data' / 'processed' / 'dataset_full.csv'
    model_path = PROJECT_ROOT / 'models' / 'catboost_model.pkl'

    df = load_data(data_path)
    df = prepare_data(df)

    print("\n--- Cross-validation ---")
    cross_validate(df)

    print("\n--- Обучение финальной модели ---")
    train_model(df, model_path)