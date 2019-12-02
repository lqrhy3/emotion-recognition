import torch
import numpy as np
from itertools import product


def xywh2xyxy(coords):
    """Convert face recrtangle coordinates
    [x_center, y_center, width, height] -> [x_lt, y_lt, x_rb, y_rb]
    Args:
        coord: (numpy.array)
    Return:
        (numpy.array)
    """
    try:
        a = coords[0, 0]
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

    except:
        new_coords = np.empty((4,), dtype=np.int)
        new_coords[0] = coords[0] - coords[2] // 2
        new_coords[1] = coords[1] - coords[3] // 2
        new_coords[2] = coords[0] + coords[2] // 2
        new_coords[3] = coords[1] + coords[3] // 2
        return new_coords


def xyxy2xywh(coord):
    """Convert face recrtangle coordinates
        [x_lt, y_lt, x_rb, y_rb] -> [x_center, y_center, width, height]
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


def to_yolo_target(bbox, image_w, grid_size=6, num_bboxes=2):
    """Convert image and bounding box to YOLO target format
    Args:
        bbox: (Tensor) [x, y, w, h]
        image_w: (int) image width
        grid_size: (int) number of cells in the grid
        num_bboxes: (int) number of predicted bounding boxes per cell
    Return:
        (Tensor) sized [5*B + C x S x S]
    """

    target_bboxes = torch.zeros((5, grid_size, grid_size))

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

    target_probabilities = torch.zeros((1, grid_size, grid_size))
    target_probabilities[:, x_cell_idx, y_cell_idx] = 1
    target = torch.cat([*[target_bboxes]*num_bboxes, target_probabilities], dim=0)

    return target


def from_yolo_target(target, image_w, grid_size=6, num_bboxes=2):
    listed_target = target[:, :5*num_bboxes, :, :].squeeze(0).view(1, 5*num_bboxes, grid_size*grid_size).\
                                                    transpose(1, 2).contiguous().view(num_bboxes*grid_size*grid_size, 5)

    cell_size = int(image_w / grid_size)
    k = 0
    for cell_idx in product(list(range(grid_size)), repeat=2):
        x_bias, y_bias = tuple(map(lambda idx: idx * cell_size, cell_idx))
        has_object = int(listed_target[k, 4] > 0)
        listed_target[k:k+2, 0] = (listed_target[k:k+2, 0] * cell_size + x_bias) * has_object
        listed_target[k:k+2, 1] = (listed_target[k:k+2, 1] * cell_size + y_bias) * has_object
        listed_target[k:k+2, 2:4] = listed_target[k:k+2, 2:4] * image_w * has_object

        k += 2

    res = listed_target.detach().numpy()
    return res


def get_object_cell(target):
    res = (target[:, 4, :, :] == 1).nonzero().squeeze(0)
    return res[1:].detach().numpy()
