import cv2
import numpy as np
import math
import os

def heatmap_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha = alpha.reshape([-1, alpha.shape[3], alpha.shape[4], 1])
    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break
        att_map = cv2.resize(att_map, (W, H))
        att_max = att_map.max()
        att_map /= att_max
        att_map *= 255
        att_map = att_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)

        show_attention = img.copy()
        show_attention = cv2.addWeighted(heatmap, 0.5, show_attention, 0.5, 0)
        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), show_attention)

    return True