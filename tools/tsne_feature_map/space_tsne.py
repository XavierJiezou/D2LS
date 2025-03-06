# 标准库
import os
import math
import warnings
from glob import glob
import argparse
from typing import Any, List, Optional, Tuple, Type

# 第三方库
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torchvision.models import (
    convnext_base,
    convnext_small,
    convnext_tiny,
    swin_b,
    swin_v2_b,
    swin_v2_s,
    swin_v2_t,
    mobilenet_v3_large,
    efficientnet_v2_m,
)

# 警告过滤
warnings.filterwarnings("ignore")

from scipy.spatial.distance import pdist, cdist


def plot_tsne(
    data,
    labels=None,
    perplexity=30,
    learning_rate=200,
    n_iter=2000,
    random_state=42,
    title="t-SNE Visualization",
    filename="tsne.png",
):
    """
    使用 t-SNE 对高维数据进行降维并可视化。

    参数：
    - data: np.array 或者 pd.DataFrame, 需要降维的数据，形状为 (N, D)
    - labels: 可选，类别标签，若提供则颜色区分
    - perplexity: t-SNE 的困惑度，影响点之间的分布关系
    - learning_rate: t-SNE 的学习率
    - n_iter: t-SNE 的迭代次数
    - random_state: 随机种子，保证结果可复现
    """
    dim = data.shape[-1]
    data = data.reshape(-1, dim)
    bs = data.shape[0]
    labels = np.array([i for i in range(5)])
    labels = np.tile(labels, (bs + len(labels) - 1) // len(labels))[:bs]

    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
    )

    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="jet", alpha=0.7, s=2
    )
    # plt.colorbar(scatter)
    # plt.title(title)
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.grid(True)
    plt.axis("off")
    filename = os.path.join(filename)
    plt.savefig(filename,bbox_inches='tight',pad_inches=0)
    plt.close()


    # 计算类内距离和类间距离
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # 类内距离：每个类别内部样本的平均距离
    intra_distances = []
    for label in unique_labels:
        class_data = reduced_data[labels == label]
        if len(class_data) < 2:
            continue  # 单个样本无法计算距离
        distances = pdist(class_data, "euclidean")
        intra_distances.append(np.mean(distances))
    intra_distance = np.mean(intra_distances) if intra_distances else 0.0

    # 类间距离：基于类中心计算
    if n_classes >= 2:
        centroids = [
            np.mean(reduced_data[labels == label], axis=0) for label in unique_labels
        ]
        centroid_dists = pdist(centroids, "euclidean")
        inter_distance = np.mean(centroid_dists)
    else:
        inter_distance = 0.0  # 仅一个类别

    print(f"类内平均距离: {intra_distance:.2f}", end=" ")
    print(f"类间平均距离 (基于类中心): {inter_distance:.2f}")

data = []
for _ in range(1151):
    data.append(nn.Embedding(7,512).weight.detach().numpy())

data = np.array(data)
plot_tsne(data)
