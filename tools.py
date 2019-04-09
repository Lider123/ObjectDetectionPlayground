import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_bbox(img, bboxes):
    colors = {
        0: "red",
        1: "green",
        2: "blue"
    }
    plt.imshow(img.T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img.shape[0], 0, img.shape[1]])
    for bbox in bboxes:
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, ec=colors[bbox[4]], fc='none'))

def draw_rgbbbox(img, bboxes, classes):
    colors = {
        0: "red",
        1: "green",
        2: "blue"
    }
    sns.set(style="dark")
    plt.imshow(np.transpose(img, (1, 0, 2)), interpolation='none', origin='lower', extent=[0, img.shape[0], 0, img.shape[1]])
    for bb, cl in zip(bboxes, classes):
        if not cl in colors:
            raise ValueError("Unknown class")
        plt.gca().add_patch(matplotlib.patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, ec=colors[cl], fc='none'))

def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U
