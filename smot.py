import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Создание искусственного набора данных с неравномерным распределением классов
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=42)

# Визуализация исходных данных
plt.figure(figsize=(10, 5))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Класс 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Класс 1')
plt.title('Исходное неравномерное распределение классов')
plt.legend()
plt.show()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Метод 1: Корректировка весов классов
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)
print("Корректировка весов классов")
print("Accuracy: ", accuracy_score(y_test, y_pred_weighted))
print("Classification Report: \n", classification_report(y_test, y_pred_weighted))

# Метод 2: SMOTE (увеличение выборки меньшего класса)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Визуализация данных после SMOTE
plt.figure(figsize=(10, 5))
plt.scatter(X_smote[y_smote == 0][:, 0], X_smote[y_smote == 0][:, 1], label='Класс 0')
plt.scatter(X_smote[y_smote == 1][:, 0], X_smote[y_smote == 1][:, 1], label='Класс 1')
plt.title('Данные после применения SMOTE')
plt.legend()
plt.show()

# Обучение модели на данных после SMOTE
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_smote, y_smote)
y_pred_smote = model_smote.predict(X_test)
print("SMOTE (увеличение выборки меньшего класса)")
print("Accuracy: ", accuracy_score(y_test, y_pred_smote))
print("Classification Report: \n", classification_report(y_test, y_pred_smote))

# Метод 3: Уменьшение выборки большего класса
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X_train, y_train)

# Визуализация данных после уменьшения выборки
plt.figure(figsize=(10, 5))
plt.scatter(X_under[y_under == 0][:, 0], X_under[y_under == 0][:, 1], label='Класс 0')
plt.scatter(X_under[y_under == 1][:, 0], X_under[y_under == 1][:, 1], label='Класс 1')
plt.title('Данные после уменьшения выборки большего класса')
plt.legend()
plt.show()

# Обучение модели на данных после уменьшения выборки
model_under = LogisticRegression(random_state=42)
model_under.fit(X_under, y_under)
y_pred_under = model_under.predict(X_test)
print("Уменьшение выборки большего класса")
print("Accuracy: ", accuracy_score(y_test, y_pred_under))
print("Classification Report: \n", classification_report(y_test, y_pred_under))