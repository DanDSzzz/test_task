import pandas as pd

# Загрузка данных
df = pd.read_csv('ndqw.csv')

# Преобразование даты покупки в формат даты
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
print(df)
print(df.info())
# Условная дата начала акции
start_date = '2023-09-01'

# Разделение на "до акции" и "после акции"
df_before = df[df['purchase_date'] < start_date]
df_after = df[df['purchase_date'] >= start_date]

# Расчет среднего чека и числа покупок до и после акции
before_avg_check = df_before['purchase_amount'].mean()
after_avg_check = df_after['purchase_amount'].mean()

before_total_sales = df_before['purchase_amount'].sum()
after_total_sales = df_after['purchase_amount'].sum()

before_num_purchases = df_before['purchase_amount'].count()
after_num_purchases = df_after['purchase_amount'].count()

print("Средний чек до акции:", before_avg_check)
print("Средний чек после акции:", after_avg_check)
print("Общие продажи до акции:", before_total_sales)
print("Общие продажи после акции:", after_total_sales)

# Фильтр клиентов, которые не покупали более 5 дней до даты рассылки
df_target_audience = df[df['days_since_last_purchase'] > 5]

# Считаем количество покупок среди целевой аудитории
num_purchases_before = df_target_audience[df_target_audience['purchase_date'] < start_date]['customer_id'].nunique()
num_purchases_after = df_target_audience[df_target_audience['purchase_date'] >= start_date]['customer_id'].nunique()

print("Количество клиентов до рассылки:", num_purchases_before)
print("Количество клиентов после рассылки:", num_purchases_after)
print(num_purchases_before)
# Эффект рассылки
effect = (num_purchases_after - num_purchases_before) / num_purchases_before * 100
print("Эффект рассылки (увеличение клиентов):", effect, "%")
import matplotlib.pyplot as plt

# График продаж до и после акции
labels = ['До акции', 'После акции']
avg_check = [before_avg_check, after_avg_check]
total_sales = [before_total_sales, after_total_sales]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Средний чек
ax[0].bar(labels, avg_check)
ax[0].set_title('Средний чек до и после акции')
ax[0].set_ylabel('Средний чек')

# Общие продажи
ax[1].bar(labels, total_sales, color='orange')
ax[1].set_title('Общие продажи до и после акции')
ax[1].set_ylabel('Объем продаж')

plt.show()