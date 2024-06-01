import torch
import torch.nn.functional as F


def warp_image_with_flow(image, flow):
    """
    Use the flow field to warp the image
    :param image: Input image, [batch_size, channels, height, width]
    :param flow: Flow tensor, [batch_size, 2, height, width]
    :return: Input image warped by the flow field
    """
    batch_size, channels, height, width = image.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(image.device)  # generate grid

    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    grid = grid + flow.permute(0, 2, 3, 1)  # apply flow

    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (width - 1) - 1.0
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (height - 1) - 1.0

    warped_image = F.grid_sample(image, grid, align_corners=True)
    return warped_image