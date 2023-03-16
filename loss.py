import torch
import torch.nn.functional as F


def edge_loss(logits, gt):
    sigmoid_p = torch.sigmoid(logits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filter_x = torch.tensor([-1., 0., 1.]).view(1, 1, 1, 3)
    filter_x = filter_x.to(device)
    xgrad_gt = torch.nn.functional.conv2d(gt, filter_x, padding=0)
    filter_y = torch.tensor([-1., 0., 1.]).view(1, 1, 1, 3)
    filter_y = filter_y.to(device)
    y_gread_gt = torch.nn.functional.conv2d(gt, filter_y, padding=0)
    x_grad_sal = torch.nn.functional.conv2d(sigmoid_p, filter_y, padding=0)
    y_grad_sal = torch.nn.functional.conv2d(sigmoid_p, filter_y, padding=0)
    loss = F.mse_loss(xgrad_gt, x_grad_sal) + F.mse_loss(y_gread_gt, y_grad_sal)
    return loss
