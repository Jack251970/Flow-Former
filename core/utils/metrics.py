from typing import Sequence

import numpy as np


def warp_image_with_flow(image_array: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp image using the given flow.

    Args:
        image_array (nparray): nparray of the image with the shape (H, W, 3).
        flow (nparray): The predicted optical flow with the shape (H, W, 2).

    Returns:
        image_ndarray (nparray): nparray of the image with the shape (H , W).
    """
    h, w = flow.shape[:2]
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1) + flow
    flow_map = np.clip(flow_map, 0, np.array([w - 1, h - 1]))
    warped_image = np.zeros_like(image_array)
    for y in range(h):
        for x in range(w):
            new_x, new_y = flow_map[y, x].astype(int)
            warped_image[y, x] = image_array[new_y, new_x]
    return warped_image


def Interpolation_error(warped_image_array: Sequence[np.ndarray],
                        image2_array: Sequence[np.ndarray]) -> np.ndarray:
    """Calculate Interpolation errors between prediction and ground truth images.

    Args:
        warped_image_array (nparray): warped array of image shape(H, W, 3).
        image2_array (nparray): ground truth array of image map shape(H, W, 3).

    Returns:
        np.ndarray: Interpolation error for output.
    """
    image2_array_norm = np.sqrt(np.sum(image2_array.astype(np.int32) ** 2, axis=2))
    warped_image_array_norm = np.sqrt(np.sum(warped_image_array.astype(np.int32) ** 2, axis=2))
    shape = np.shape(image2_array_norm - warped_image_array_norm)
    temp = (image2_array_norm - warped_image_array_norm) ** 2
    squared_diff = np.sum(temp) / (shape[0] * shape[1])
    sqrt_diff = np.sqrt(squared_diff)  # float
    return np.array([sqrt_diff])


def angular_error_map(flow_pred: np.ndarray, flow_gt: np.ndarray) -> np.ndarray:
    """Calculate angular error map.

    Args:
        flow_pred (ndarray): The predicted optical flow with the shape (H, W, 2).
        flow_gt (ndarray): The ground truth of optical flow with the shape (H, W, 2).

    Returns:
        ndarray: Angular error map with the shape (H , W).
    """
    dot_product = np.sum(flow_pred * flow_gt, axis=-1) + 1
    pred_norm_product = np.sqrt(np.sum(flow_pred * flow_pred, axis=-1) + 1)
    gt_norm_product = np.sqrt(np.sum(flow_gt * flow_gt, axis=-1) + 1)
    ae = np.arccos(np.clip(dot_product / (pred_norm_product * gt_norm_product), -1.0, 1.0))
    return ae


def angular_error(flow_pred: Sequence[np.ndarray], flow_gt: Sequence[np.ndarray],
                  valid_gt: Sequence[np.ndarray]) -> np.ndarray:
    """Calculate angular errors between prediction and ground truth.

    Args:
        flow_pred (list): output list of flow map from flow_estimator shape(H, W, 2).
        flow_gt (list): ground truth list of flow map shape(H, W, 2).
        valid_gt (list): the list of valid mask for ground truth with the shape (H, W).

    Returns:
        np.ndarray: angular error for output.
    """
    ae_list = []
    assert len(flow_pred) == len(flow_gt)
    for _flow_pred, _flow_gt, _valid_gt in zip(flow_pred, flow_gt, valid_gt):
        ae_map = angular_error_map(_flow_pred, _flow_gt)
        val = _valid_gt.reshape(-1) >= 0.5
        ae_list.append(ae_map.reshape(-1)[val])

    ae_all = np.concatenate(ae_list)
    ae = np.mean(ae_all)  # float

    return np.array([ae])
