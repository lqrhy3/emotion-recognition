import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    def __init__(self, grid_size, num_bboxes, num_classes=1, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def _compute_iou(self, target_bbox, pred_bboxes):
        """Compute Intersection over Union
        Compute IoU between ground truth bounding box corresponded to the cell
        and B predicted bonding boxes
        Args:
             target_bbox: (Tensor) target bounding box, sized [1, 4]
             pred_bboxes: (Tensor) predicted bounding boxes, sized [B, 4]
        Returns:
            (Tensor) IoU with target bbox for every predicted bbox, sized [1, B]
        """
        xy_lt = torch.max(
            target_bbox[:, :2].unsqueeze(1).expand(1, self.B, 2),
            pred_bboxes[:, :2].unsqueeze(0)
        )                                                          # [1, B, 2]

        xy_rb = torch.min(
            target_bbox[:, 2:].unsqueeze(1).expand(1, self.B, 2),
            pred_bboxes[:, 2:].unsqueeze(0)
        )                                                          # [1, B, 2]

        intersection_wh = xy_rb - xy_lt
        intersection_wh[intersection_wh < 0] = 0                   # [1, B, 2]

        intersection = intersection_wh[:, :, 1] * intersection_wh[:, :, 0]  # [1, B]

        target_area = (target_bbox[:, 2] - target_bbox[:, 0]) *\
                      (target_bbox[:, 3] - target_bbox[:, 1])           # [1, ]
        pred_area = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) *\
                    (pred_bboxes[:, 3] - pred_bboxes[:, 1])             # [B, ]
        target_area = target_area.unsqueeze(1).expand_as(intersection)  # [1, B]
        pred_area = pred_area.unsqueeze(0)                              # [1, B]
        assert target_area.size() == pred_area.size()

        union = target_area + pred_area - intersection                  # [1, B]

        iou = intersection / union                                      # [1, B]

        return iou

    def forward(self, pred, target):
        """Compute YOLO loss
        Args:
            pred: (Tensor) yolo output, sized [n_batch, Bx5 + C, S, S], 5=len([x, y, w, h, conf])
            target: (Tensor) targets, sized [n_batch, Bx5 + C, S, S]
        Returns:
            (Tensor) loss value, sized [1,]
        """

        self._N = pred.size(0)  # bath size

        listed_target = target.view(self._N, 5*self.B+self.C, self.S*self.S).transpose(1, 2)  # [N, 5xB+C, S, S] -> [N, SxS, 5xB+C]
        assert listed_target.size() == torch.Size([self._N, self.S*self.S, 5*self.B+self.C])

        obj_mask = listed_target[:, :, 4] > 0  # cells which contain object
        noobj_mask = listed_target[:, :, 4] == 0  # cells which have no objects
        obj_mask = obj_mask.unsqueeze(-1).expand_as(listed_target)      # [N, SxS, 5xB+C]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(listed_target)  # [N, SxS, 5xB+C]

        listed_pred = pred.view(self._N, 5 * self.B + self.C, self.S * self.S).transpose(1, 2)  # [N, SxS, 5xB+C]
        assert (listed_pred.size(1), listed_pred.size(2)) == (self.S*self.S, 5*self.B+self.C)

        obj_pred = listed_pred[obj_mask].view(-1, 5*self.B+self.C)        # [n_obj_cells, 5xB+C]
        assert obj_pred.size(1) == 5 * self.B + self.C

        noobj_pred = listed_pred[noobj_mask].view(-1, 5*self.B+self.C)    # [n_noobj_cells, 5xB+C]
        assert noobj_pred.size(1) == 5 * self.B + self.C

        obj_target = listed_target[obj_mask].view(-1, 5*self.B+self.C)      # [n_obj_cells, 5xB+C]
        noobj_target = listed_target[noobj_mask].view(-1, 5*self.B+self.C)  # [n_noobj_cells, 5xB+C]

        # Compute loss for cells which have no objects
        conf_mask = torch.zeros(noobj_target.size(), dtype=torch.bool)  # [n_noobj_cells, 5xB+C]
        for b in range(self.B):
            conf_mask[:, 4 + 5 * b] = 1
        noobj_pred_conf = noobj_pred[conf_mask]      # [n_noobj_cells x B, ] confidence for each box
        noobj_target_conf = noobj_target[conf_mask]  # [n_noobj_cells x B, ]
        assert noobj_target_conf.size(0) == noobj_target.size(0) * 2
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for cells which contain objects
        bbox_target = obj_target[:, :5*self.B].contiguous().view(-1, 5)  # [n_obj_cells x B, 5]
        assert bbox_target.size() == torch.Size([obj_target.size(0)*self.B, 5])
        bbox_pred = obj_pred[:, :5*self.B].contiguous().view(-1, 5)      # [n_obj_cells x B, 5]

        obj_response_mask = torch.zeros(bbox_target.size(), dtype=torch.bool)     # [n_obj_cells x B, 5]
        # obj_response_target_mask = torch.zeros(bbox_target.size(), dtype=torch.uint8)     # [n_obj_cells x B, 5]
        obj_not_response_mask = torch.ones(bbox_target.size(), dtype=torch.bool)  # [n_obj_cells x B, 5]

        bbox_target_iou = torch.zeros(bbox_target.size())  # [n_obj_cells x B, 5]

        for i in range(0, bbox_target.size(0), self.B):
            cur_bbox_pred = bbox_pred[i:i + self.B]  # [B, 5]

            pred_xyxy = torch.empty(cur_bbox_pred.size(), dtype=torch.float, requires_grad=True)  # [B, 5]
            pred_xyxy[:, :2] = cur_bbox_pred[:, :2] / float(self.S) - 0.5 * cur_bbox_pred[:, 2:4]
            pred_xyxy[:, 2:4] = cur_bbox_pred[:, :2] / float(self.S) + 0.5 * cur_bbox_pred[:, 2:4]

            cur_bbox_target = bbox_target[i].view(-1, 5)  # [1, 5]
            assert cur_bbox_target.size() == torch.Size([1, 5])

            target_xyxy = torch.empty(cur_bbox_target.size(), dtype=torch.float, requires_grad=True)  # [1, 5]
            target_xyxy[:, :2] = cur_bbox_target[:, :2] / float(self.S) - 0.5 * cur_bbox_target[:, 2:4]
            target_xyxy[:, 2:4] = cur_bbox_target[:, :2] / float(self.S) + 0.5 * cur_bbox_target[:, 2:4]

            iou = self._compute_iou(target_xyxy[:, :4], pred_xyxy[:, :4])  # [1, B]
            max_iou, max_iou_index = iou.max(1)         # [1,], [1,]
            max_iou_index = max_iou_index.data[0]       # []

            obj_response_mask[i+max_iou_index] = 1
            # obj_response_target_mask[i] = 1
            obj_not_response_mask[i+max_iou_index] = 0

            bbox_target_iou[i+max_iou_index, 4] = max_iou.data[0]

        bbox_target_iou = bbox_target_iou.clone().detach().requires_grad_(True)
        # Bounding box coordinates, sizes and confidence loss for the responsible boxes
        bbox_pred_response = bbox_pred[obj_response_mask].view(-1, 5)
        bbox_target_response = bbox_target[obj_response_mask].view(-1, 5)
        # bbox_target_response = bbox_target[obj_response_target_mask].view(-1, 5)
        target_iou = bbox_target_iou[obj_response_mask].view(-1, 5)

        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(bbox_pred_response[:, 2:4], bbox_target_response[:, 2:4], reduction='sum')
        loss_conf = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the responsible boxes
        class_pred = obj_pred[:, 5*self.B:]
        class_target = obj_target[:, 5*self.B:]
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Final loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_conf + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / self._N

        logger_loss = {'Coordinates loss': loss_xy.item() / self._N, 'Width/Height loss': loss_wh.item() / self._N,
                       'Confidence loss': loss_conf.item() / self._N, 'No object loss': loss_noobj.item() / self._N,
                        'Class probabilities loss': loss_class.item() / self._N, 'Total loss': loss.item()}
        return loss, logger_loss
