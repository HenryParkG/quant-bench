import torch

def apply_minmax_quant(model, bit_width=8):
    q_model = model
    scale = 2 ** bit_width - 1
    for name, param in q_model.named_parameters():
        min_val = param.data.min()
        max_val = param.data.max()
        param.data = torch.round((param.data - min_val) / (max_val - min_val) * scale) / scale * (max_val - min_val) + min_val
    return q_model
