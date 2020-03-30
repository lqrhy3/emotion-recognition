import torch
import numpy as np
import os
from itertools import product


def xywh2xyxy(coords):
    """Convert face recrtangle coordinates
    [bbox_x_center, y_center, width, height] -> [x_lt, y_lt, x_rb, y_rb]
    Args:
        coords: (numpy.array)
    Return:
        (numpy.array)
    """

    if len(coords.shape) == 1:
        new_coords = np.empty((4,), dtype=np.int)
        new_coords[0] = coords[0] - coords[2] // 2
        new_coords[1] = coords[1] - coords[3] // 2
        new_coords[2] = coords[0] + coords[2] // 2
        new_coords[3] = coords[1] + coords[3] // 2

    elif len(coords.shape) == 2:
        new_coords = np.empty((coords.shape[1],), dtype=np.int)
        new_coords[0] = coords[0][0] - coords[0][2] // 2
        new_coords[1] = coords[0][1] - coords[0][3] // 2
        new_coords[2] = coords[0][0] + coords[0][2] // 2
        new_coords[3] = coords[0][1] + coords[0][3] // 2
        for coord in coords[1:]:
            new_coord = np.empty((coords.shape[1],), dtype=np.int)
            new_coord[0] = coord[0] - coord[2] // 2
            new_coord[1] = coord[1] - coord[3] // 2
            new_coord[2] = coord[0] + coord[2] // 2
            new_coord[3] = coord[1] + coord[3] // 2
            new_coords = np.vstack((new_coords, new_coord))
    return new_coords


def xyxy2xywh(coord):
    """Convert face recrtangle coordinates
        [x_lt, y_lt, x_rb, y_rb] -> [bbox_x_center, y_center, width, height]
        Args:
            coord: (numpy.array)
        Return:
            (numpy.array)
        """
    new_coord = np.empty((4,), dtype=np.int)
    new_coord[0] = (coord[2] + coord[0]) // 2
    new_coord[1] = (coord[3] + coord[1]) // 2
    new_coord[2] = coord[2] - coord[0]
    new_coord[3] = coord[3] - coord[1]
    return new_coord


# def to_yolo_target(bbox, image_w, grid_size, num_bboxes=2):
#     """Convert image and bounding box to YOLO target format
#     Args:
#         bbox: (np.ndarray) [x, y, w, h]
#         image_w: (int) image width
#         grid_size: (int) number of cells in the grid
#         num_bboxes: (int) number of predicted bounding boxes per cell
#     Return:
#         (Tensor) sized [5*B + C x S x S]
#     """
#     target_bboxes = torch.zeros((5, grid_size, grid_size))
#
#     cell_size = image_w / grid_size
#     x_cell_idx = int(bbox[0].item() // cell_size)
#     y_cell_idx = int(bbox[1].item() // cell_size)
#     relative_x = bbox[0].item() % cell_size
#     relative_y = bbox[1].item() % cell_size
#
#     # x_cell_idx, relative_x = divmod(bbox[0].item(), cell_size)
#     # y_cell_idx, relative_y = divmod(bbox[1].item(), cell_size)
#
#     target_bboxes[0, x_cell_idx, y_cell_idx] = relative_x / cell_size
#     target_bboxes[1, x_cell_idx, y_cell_idx] = relative_y / cell_size
#     target_bboxes[2, x_cell_idx, y_cell_idx] = bbox[2].item() / image_w
#     target_bboxes[3, x_cell_idx, y_cell_idx] = bbox[3].item() / image_w
#     target_bboxes[4, x_cell_idx, y_cell_idx] = 1
#
#     target_probabilities = torch.zeros((1, grid_size, grid_size))
#     target_probabilities[:, x_cell_idx, y_cell_idx] = 1
#     target = torch.cat([*[target_bboxes] * num_bboxes, target_probabilities], dim=0)
#
#     return target

def to_yolo_target(bbox, image_w, grid_size, num_bboxes=2):
    """Convert image and bounding box to YOLO target format
    Args:
        bbox: (np.ndarray) [x, y, w, h]
        image_w: (int) image width
        grid_size: (int) number of cells in the grid
        num_bboxes: (int) number of predicted bounding boxes per cell
    Return:
        (Tensor) sized [5*B + C x S x S]
    """
    target_bboxes = np.zeros((5, grid_size, grid_size))

    cell_size = image_w / grid_size
    x_cell_idx = int(bbox[0].item() // cell_size)
    y_cell_idx = int(bbox[1].item() // cell_size)
    relative_x = bbox[0].item() % cell_size
    relative_y = bbox[1].item() % cell_size

    # x_cell_idx, relative_x = divmod(bbox[0].item(), cell_size)
    # y_cell_idx, relative_y = divmod(bbox[1].item(), cell_size)

    target_bboxes[0, x_cell_idx, y_cell_idx] = relative_x / cell_size
    target_bboxes[1, x_cell_idx, y_cell_idx] = relative_y / cell_size
    target_bboxes[2, x_cell_idx, y_cell_idx] = bbox[2].item() / image_w
    target_bboxes[3, x_cell_idx, y_cell_idx] = bbox[3].item() / image_w
    target_bboxes[4, x_cell_idx, y_cell_idx] = 1

    target_probabilities = np.zeros((1, grid_size, grid_size))
    target_probabilities[:, x_cell_idx, y_cell_idx] = 1
    target = np.concatenate([*[target_bboxes] * num_bboxes, target_probabilities], axis=0)

    return target


def from_yolo_target(target, image_w, grid_size, num_bboxes):
    batch_size = target.size(0)
    listed_target = target[:, :5 * num_bboxes, :, :]
    listed_target = listed_target.view(-1, 5 * num_bboxes, grid_size * grid_size)
    listed_target = listed_target.transpose(1, 2).contiguous().view(-1, num_bboxes * grid_size * grid_size, 5)

    for batch in range(batch_size):
        pos = 0
        cell_size = int(image_w / grid_size)
        for cell_idx in product(list(range(grid_size)), repeat=2):
            x_bias, y_bias = tuple(map(lambda idx: idx * cell_size, cell_idx))
            has_object = int(listed_target[batch, pos, 4] > 0)
            listed_target[batch, pos:pos + num_bboxes, 0] = \
                (listed_target[batch, pos:pos + num_bboxes, 0] * cell_size + x_bias) * has_object
            listed_target[batch, pos:pos + num_bboxes, 1] = \
                (listed_target[batch, pos:pos + num_bboxes, 1] * cell_size + y_bias) * has_object
            listed_target[batch, pos:pos + num_bboxes, 2:4] = \
                listed_target[batch, pos:pos + num_bboxes, 2:4] * image_w * has_object

            pos += num_bboxes

    res = listed_target.detach().cpu().numpy()
    return res


def get_object_cell(target):
    res = (target[:, 4, :, :] == 1).nonzero().squeeze(0)
    return res[1:].detach().numpy()


def compute_iou(bbox_1, bbox_2, num_bboxes):
    """Compute Intersection over Union
    Compute IoU between ground truth bounding box corresponded to the cell
    and B predicted bonding boxes
    Args:
         bbox_1: (Tensor) bounding box, sized (1, 4) [x_lt, y_lt, x_rb, y_rb]
         bbox_2: (Tensor) bounding boxes, sized (B, 4)
    Returns:
        (Tensor) IoU with target bbox for every predicted bbox, sized (1, B)
    """
    xy_lt = torch.max(
        bbox_1[:, :2],
        bbox_2[:, :2]
    )

    xy_rb = torch.min(
        bbox_1[:, 2:],
        bbox_2[:, 2:]
    )

    intersection_wh = xy_rb - xy_lt
    intersection_wh[intersection_wh < 0] = 0  # (1, B, 2)

    intersection = intersection_wh[:, 1] * intersection_wh[:, 0]  # (1, B)

    target_area = (bbox_1[:, 2] - bbox_1[:, 0]) * \
                  (bbox_1[:, 3] - bbox_1[:, 1])
    pred_area = (bbox_2[:, 2] - bbox_2[:, 0]) * \
                (bbox_2[:, 3] - bbox_2[:, 1])
    target_area = target_area
    pred_area = pred_area
    assert target_area.size() == pred_area.size()

    union = target_area + pred_area - intersection

    iou = intersection / union

    return iou


def bbox_resize(coords, from_shape, to_shape):
    """Resize bounding box coordinates correspondingly to to_shape
    :param coords: numpy array. bounding box coordinates
    :param from_shape: tuple. current image shape
    :param to_shape: tuple. new image shape
    """
    new_coords = list()
    new_coords.append(int(coords[0] * (to_shape[0] / from_shape[0])))
    new_coords.append(int(coords[1] * (to_shape[1] / from_shape[1])))
    new_coords.append(int(coords[2] * (to_shape[0] / from_shape[0])))
    new_coords.append(int(coords[3] * (to_shape[1] / from_shape[1])))
    return new_coords


def get_model_size(model):
    """"Compute the size of the model
    Returns:
        (float) size of the model in MB"""
    torch.save(model, "temp.pt")
    size = os.path.getsize("temp.pt") / 1e6
    os.remove("temp.pt")
    return size
