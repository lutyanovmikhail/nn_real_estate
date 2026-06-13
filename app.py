import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# ─── КОНФИГ ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Оценка квартир — Нижний Новгород",
    page_icon="🏠",
    layout="centered"
)

DISTRICTS = [
    'Нижегородский район',
    'Советский район',
    'Канавинский район',
    'Приокский район',
    'Ленинский район',
    'Сормовский район',
    'Автозаводский район',
    'Московский район',
]

MATERIALS = [
    'кирпич',
    'панель',
    'монолитный железобетон',
    'блок+утеплитель',
    'шлакоблок',
    'дерево',
    'unknown',
]

DISTRICT_MEAN_AREA = {
    'Нижегородский район': 72.0,
    'Советский район':     58.0,
    'Канавинский район':   52.0,
    'Приокский район':     55.0,
    'Ленинский район':     50.0,
    'Сормовский район':    48.0,
    'Автозаводский район': 50.0,
    'Московский район':    49.0,
}

CAT_FEATURES = ['district', 'material', 'district_ready', 'mini_disctrict', 'material_age']

FINAL_FEATURES = [
    'rooms', 'total_area', 'year', 'kitchen_area',
    'current_floor', 'max_floor',
    'area_floor_interaction', 'area_ratio_to_district',
    'distance_to_center', 'distance_to_metro',
    'district', 'material', 'district_ready',
    'mini_disctrict', 'material_age',
    'avg_price_m2_segmented'
]

# ─── ЗАГРУЗКА МОДЕЛИ ─────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load(PROJECT_ROOT / 'models' / 'catboost_model.pkl')
    return bundle

# ─── ПОДГОТОВКА ПРИЗНАКОВ ────────────────────────────────────
def build_features(
    district, material, rooms, total_area, kitchen_area,
    current_floor, max_floor, year, distance_to_center,
    distance_to_metro, target_map, global_mean_m2
):
    is_ready = 1

    house_segment = pd.cut(
        [max_floor],
        bins=[0, 5, 10, 19, 100],
        labels=['low', 'standard', 'modern', 'high']
    )[0]

    district_ready  = f"{district}_{float(is_ready)}"
    material_age    = f"{material}_False"
    mini_disctrict  = 'unknown'

    area_ratio = total_area / DISTRICT_MEAN_AREA.get(district, 55.0)
    area_floor = total_area * current_floor

    avg_m2 = target_map.get((district, house_segment), global_mean_m2)

    row = {
        'rooms':                  rooms,
        'total_area':             total_area,
        'year':                   float(year),
        'kitchen_area':           kitchen_area,
        'current_floor':          current_floor,
        'max_floor':              max_floor,
        'area_floor_interaction': area_floor,
        'area_ratio_to_district': area_ratio,
        'distance_to_center':     distance_to_center,
        'distance_to_metro':      distance_to_metro,
        'district':               district,
        'material':               material,
        'district_ready':         district_ready,
        'mini_disctrict':         mini_disctrict,
        'material_age':           material_age,
        'avg_price_m2_segmented': avg_m2,
    }

    return pd.DataFrame([row])[FINAL_FEATURES]

# ─── SHAP WATERFALL ──────────────────────────────────────────
def plot_shap(model, X_input):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input.iloc[0],
            feature_names=FINAL_FEATURES
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

# ─── UI ──────────────────────────────────────────────────────
st.title(" Оценка квартиры в Нижнем Новгороде")
st.caption("Модель: CatBoost + LightGBM ensemble · R² = 0.895 · MAE = 1.21 млн ₽")

st.divider()

col1, col2 = st.columns(2)

with col1:
    district = st.selectbox("Район", DISTRICTS)
    material = st.selectbox("Материал стен", MATERIALS)
    rooms    = st.number_input("Количество комнат", min_value=0, max_value=10, value=2,
                                help="0 = студия")
    total_area   = st.number_input("Общая площадь, м²", min_value=10.0, max_value=300.0,
                                    value=50.0, step=0.5)
    kitchen_area = st.number_input("Площадь кухни, м²", min_value=3.0, max_value=80.0,
                                    value=10.0, step=0.5)

with col2:
    current_floor = st.number_input("Этаж", min_value=1, max_value=50, value=5)
    max_floor     = st.number_input("Этажей в доме", min_value=1, max_value=50, value=9)
    year          = st.number_input("Возраст здания (лет)", min_value=0, max_value=100,
                                     value=10, help="0 = новостройка")
    distance_to_center = st.number_input("Расстояние до центра, км",
                                          min_value=0.1, max_value=30.0, value=5.0, step=0.1)
    distance_to_metro  = st.number_input("Расстояние до метро, км",
                                          min_value=0.1, max_value=20.0, value=3.0, step=0.1)

st.divider()

if current_floor > max_floor:
    st.warning("Этаж не может быть больше этажей в доме")
else:
    if st.button("🔍 Оценить квартиру", use_container_width=True, type="primary"):
        bundle       = load_model()
        model        = bundle['model']
        target_map   = bundle['target_map']
        global_mean  = bundle['global_mean_m2']

        X_input = build_features(
            district, material, rooms, total_area, kitchen_area,
            current_floor, max_floor, year,
            distance_to_center, distance_to_metro,
            target_map, global_mean
        )

        pred_log   = model.predict(X_input)[0]
        pred_price = np.exp(pred_log)
        mae        = 1_210_000

        st.divider()
        st.subheader("Результат")

        c1, c2, c3 = st.columns(3)
        c1.metric("Предсказанная цена",  f"{pred_price/1e6:.2f} млн ₽")
        c2.metric("Нижняя граница",      f"{(pred_price - mae)/1e6:.2f} млн ₽")
        c3.metric("Верхняя граница",     f"{(pred_price + mae)/1e6:.2f} млн ₽")

        st.caption(f"Диапазон ± {mae/1e6:.2f} млн ₽ (средняя ошибка модели на тестовых данных)")

        # Сегмент рынка
        if pred_price < 7_000_000:
            segment = " Эконом (< 7 млн ₽)"
        elif pred_price < 20_000_000:
            segment = " Средний (7–20 млн ₽)"
        else:
            segment = " Премиум (> 20 млн ₽)"
        st.info(f"Ценовой сегмент: {segment}")

        st.divider()
        st.subheader("Почему такая цена? (SHAP)")
        st.caption("Каждый признак показывает свой вклад в итоговую цену относительно средней по датасету")
        plot_shap(model, X_input)

        st.divider()
        st.subheader("Параметры квартиры")
        summary = {
            'Район':               district,
            'Материал':            material,
            'Комнат':              rooms,
            'Площадь':             f"{total_area} м²",
            'Кухня':               f"{kitchen_area} м²",
            'Этаж':                f"{current_floor}/{max_floor}",
            'Возраст здания':      f"{year} лет",
            'До центра':           f"{distance_to_center} км",
            'До метро':            f"{distance_to_metro} км",
            'Цена/м²':             f"{pred_price/total_area:,.0f} ₽",
        }
        st.table(pd.DataFrame(summary.items(), columns=['Параметр', 'Значение']))

st.divider()
st.caption("Модель обучена на 5 930 объявлениях с gipernn.ru, cian.ru, avito.ru · Апрель 2025")