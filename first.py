import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error
import pydotplus
from IPython.display import Image
import graphviz

# Пример данных: площадь дома (м²) и количество комнат, цена дома (тыс. $)
X = np.array([[50, 2], [60, 3], [70, 3], [80, 3], [90, 4], [100, 4], [110, 4], [120, 5], [130, 5], [140, 5]])
y = np.array([150, 160, 170, 180, 190, 200, 210, 220, 230, 240])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели дерева решений для регрессии
model = DecisionTreeRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Визуализация дерева решений
dot_data = export_graphviz(model, out_file=None,
                           feature_names=['Площадь', 'Комнаты'],
                           filled=True, rounded=True,
                           special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())