class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    def display(self):
        current = self.head
        while current:
            print(current.data, end=' ')
            current = current.next

# Пример использования связного списка
llist = LinkedList()
llist.append(1)
llist.append(2)
llist.append(3)
llist.display()  # Вывод: 1 2 3

# Пример нахождения суммы двух чисел, равной целевому значению
def two_sum(nums, target):
    nums.sort()
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [nums[left], nums[right]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Пример использования
print(two_sum([1, 2, 3, 4, 5], 5))  # Вывод: [4, 5]

# Пример проверки палиндрома
def is_palindrome(s):
    return s == s[::-1]

# Пример использования
print(is_palindrome("racecar"))  # Вывод: True
print(is_palindrome("hello"))    # Вывод: False

# Пример вычисления факториала числа
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Пример использования
print(factorial(5))  # Вывод: 120

import numpy as np
from scipy import stats


# Генерация примера данных
np.random.seed(42)
control_group = np.random.binomial(1, 0.1, 1000)  # Контрольная группа с конверсией 10%
test_group = np.random.binomial(1, 0.12, 1000)    # Тестовая группа с конверсией 12%

# Расчет конверсии
control_cr = np.mean(control_group)
test_cr = np.mean(test_group)

print(f"Conversion Rate (Control): {control_cr:.2%}")
print(f"Conversion Rate (Test): {test_cr:.2%}")

# T-test для сравнения конверсий
t_stat, p_value = stats.ttest_ind(control_group, test_group)
print(f"T-test p-value: {p_value:.5f}")

# Проверка статистической значимости
alpha = 0.05
if p_value < alpha:
    print("Результат статистически значим.")
else:
    print("Результат не является статистически значимым.")


import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Создание таблицы сопряженности
data = {'Gender': ['Male', 'Male', 'Female', 'Female'],
        'Product Preference': ['Yes', 'No', 'Yes', 'No'],
        'Count': [20, 10, 30, 40]}

df = pd.DataFrame(data)

# Создание кросс-таблицы
contingency_table = pd.pivot_table(df, values='Count', index='Gender', columns='Product Preference', aggfunc=np.sum)

# Выполнение хи-квадрат теста
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Test Statistic: {chi2}")
print(f"P-Value: {p}")

# Проверка статистической значимости
alpha = 0.05
if p < alpha:
    print("Результат статистически значим. Существует зависимость между полом и предпочтением продукта.")
else:
    print("Результат не является статистически значимым. Нет зависимости между полом и предпочтением продукта.")

import random

# Пример данных пользователей
users = pd.DataFrame({
    'user_id': range(1, 1001),
    'group': [''] * 1000,
    'converted': np.random.binomial(1, 0.11, 1000)
})

# Случайное распределение пользователей по группам
users['group'] = users['user_id'].apply(lambda x: 'test' if random.random() < 0.5 else 'control')

# Разделение на тестовую и контрольную группы
control_users = users[users['group'] == 'control']
test_users = users[users['group'] == 'test']

# Вычисление конверсии
control_cr = control_users['converted'].mean()
test_cr = test_users['converted'].mean()

print(f"Conversion Rate (Control): {control_cr:.2%}")
print(f"Conversion Rate (Test): {test_cr:.2%}")

# T-test для сравнения конверсий
t_stat, p_value = stats.ttest_ind(control_users['converted'], test_users['converted'])
print(f"T-test p-value: {p_value:.5f}")

# Проверка статистической значимости
if p_value < alpha:
    print("Результат статистически значим.")
else:
    print("Результат не является статистически значимым.")