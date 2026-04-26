import pandas as pd

df_main  = pd.read_csv('/Users/apch/Pycharmproekti/nn_real_estate/data/processed/dataset_with_geo.csv')
df_avito = pd.read_csv('/Users/apch/Pycharmproekti/nn_real_estate/data/processed/avito_processed.csv')

print(f"Основной датасет: {len(df_main)} строк")
print(f"Авито: {len(df_avito)} строк")
print(f"Колонки основного: {sorted(df_main.columns.tolist())}")
print(f"Колонки авито:     {sorted(df_avito.columns.tolist())}")

# Колонки которые есть в основном но нет в авито
missing = set(df_main.columns) - set(df_avito.columns)
print(f"\nЕсть в основном, нет в авито: {missing}")

# Колонки которые есть в авито но нет в основном
extra = set(df_avito.columns) - set(df_main.columns)
print(f"Есть в авито, нет в основном: {extra}")

import numpy as np

# Добавляем недостающие колонки в авито
df_avito['price_log'] = np.log(df_avito['price'])
df_avito['price_mln'] = df_avito['price'] / 1_000_000
df_avito['metro']     = np.nan  # в авито только название станции, не время

# Убираем address — в основном датасете её нет
df_avito = df_avito.drop(columns=['address'])

# Мерж
merged = pd.concat([df_main, df_avito], ignore_index=True)
print(f"До дедупликации: {len(merged)} строк")

# Дедупликация
before = len(merged)
merged = merged.drop_duplicates(
    subset=['total_area', 'current_floor', 'max_floor', 'district', 'price']
)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"После дедупликации: {len(merged)} строк")
print(f"Дубликатов удалено: {before - len(merged)}")

# Сохраняем
merged.to_csv('/Users/apch/Pycharmproekti/nn_real_estate/data/processed/dataset_full.csv', index=False, encoding='utf-8')
print(f"\nГотово: {len(merged)} строк, {len(merged.columns)} колонок")