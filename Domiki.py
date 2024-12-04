import matplotlib.pyplot as plt


def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference

    plt.title("Difference (norm: %.2f)" % np.sqrt(np.sum(difference**2)))
    plt.imshow(
        difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr, interpolation="nearest"
    )
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


show_with_diff(distorted, raccoon_face, "Distorted image")


# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
#
# import numpy as np
#
# if __name__ == "__main__":
#     x = []
#
#     for i in range(100):
#         x.append([np.random.randint(0, 50), np.random.randint(0, 50)])
#
#         x.append([100 + np.random.randint(0, 50), np.random.randint(0, 50)])
#
#         x.append([np.random.randint(0, 50), 100 + np.random.randint(0, 50)])
#
#
#     x = np.array(x)
#
#
#
#     kmeans = KMeans(n_clusters=3)
#     kmeans.fit(x)
#     labels = kmeans.labels_
#
#     plt.scatter(x[:, :1], x[:, 1:], c=labels)
#
#     plt.show()
