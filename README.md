# Real Estate Price Prediction

Модель CatBoost для предсказания цен на квартиры в Нижнем Новгороде.

## Метрики
- R²: 0.876
- MAPE: 12.49%
- **MAE:** 1.71 млн ₽


## Как запустить
1. Клоинровать репозиторий
2. Установить зависимости: `pip install -r requirements.txt`
3. Обучить модель: `python src/models/train_model.py`
4. Сделать предсказание: `python src/models/predict_model.py`

## Файлы
- `train_model.py` — обучение модели
- `predict_model.py` — предсказание для новых данных
- `models/catboost_model.pkl` — сохранённая модель

## Структура
- `src/models/` — код для обучения и предсказаний
- `models/` — сохранённая модель CatBoost

## Результаты
Модель стабильна, переобучение отсутствует (R² train 0.886, R² test 0.876).