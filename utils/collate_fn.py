def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return imgs, targets


# def collate_fn(batch):
#     return tuple(zip(*batch))
