from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Загрузка набора данных Iris
iris = load_iris()
data = iris.data

# Метрика для нахождения оптимального количества кластеров - метод локтgit add README.mdя
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

# Построение графика метода локтя
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), sse, marker='o') # Строим график зависимости количества кластеров от SSE (сумму квадратов ошибок)
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('SSE')
plt.show()