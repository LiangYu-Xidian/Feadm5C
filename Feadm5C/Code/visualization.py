import matplotlib.pyplot as plt
from sklearn import manifold, datasets


def tsne_visualization(X1, y1, X2, y2):
    tsne = manifold.TSNE(n_components=2, init='random', random_state=517)
    fig1 = plt.figure(figsize=(18, 8), dpi=300)
    ax1 = fig1.add_subplot(1, 2, 1)
    plt.title("(c)", font={'size': 30}, pad=10)
    X1_tsne = tsne.fit_transform(X1)

    x1_min, x1_max = X1_tsne.min(0), X1_tsne.max(0)
    X1_norm = (X1_tsne - x1_min) / (x1_max - x1_min)

    for i in range(X1_norm.shape[0]):
        plt.text(X1_norm[i, 0], X1_norm[i, 1], str(y1[i]), color=plt.cm.Set1(1 - y1[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])

    ax2 = fig1.add_subplot(1, 2, 2)
    plt.title("(d)", font={'size': 30}, pad=10)
    X2_tsne = tsne.fit_transform(X2)
    '''嵌入空间可视化'''
    x2_min, x2_max = X2_tsne.min(0), X2_tsne.max(0)
    X2_norm = (X2_tsne - x2_min) / (x2_max - x2_min)

    for i in range(X2_norm.shape[0]):
        plt.text(X2_norm[i, 0], X2_norm[i, 1], str(y2[i]), color=plt.cm.Set1(1 - y2[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])

    plt.show()
