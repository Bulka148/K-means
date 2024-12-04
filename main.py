from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from fcmeans import FCM

import pygame


def draw(dataset, clusters):
    fig, ax = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            ax[i][j].scatter(dataset[:, i], dataset[:, j], c=clusters)
    plt.show()


def main():
    iris = load_iris()
    data = iris.data
    target = iris.target
    # draw(data, target)

    kmeans = KMeans(3, n_init='auto')
    kmeans.fit(data)
    labels = kmeans.labels_
    # draw(data, labels)

    fcm = FCM(n_clusters=3)
    fcm.fit(data)
    labelc = fcm.predict(data)

    print(accuracy_score(labelc, target))
    print(confusion_matrix(labelc, target))
    matrix = confusion_matrix(labelc, target)
    sum = 0
    for i in range(3):
        sum += np.max(matrix[i])
    print(sum / 150)


if name == 'main':
    main()


    # Загрузка датасета
    iris = load_iris()
    data = iris.data

    # Запуск алгоритма K-means с оптимальным количеством кластеров
    optimal_clusters = 3  # Заданное оптимальное значение после метода локтя
    k_means(data, optimal_clusters)
