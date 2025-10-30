import torch

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return imgs, targets


# def collate_fn(batch):
#     return tuple(zip(*batch))


import torch

def yolo_collate(batch):
    imgs, targets = [], []
    for img, target in batch:
        imgs.append(img)
        # target이 dict인 경우 처리
        if isinstance(target, dict):
            boxes = target.get("boxes")
            labels = target.get("labels")
            if boxes is not None and labels is not None:
                label_tensor = torch.cat([labels.unsqueeze(1), boxes], dim=1)
                targets.append(label_tensor)
            else:
                targets.append(torch.zeros((0, 5)))
        else:
            # target이 이미 tensor인 경우
            if target.numel() > 0:
                targets.append(target)
            else:
                targets.append(torch.zeros((0, 5)))
    
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets
