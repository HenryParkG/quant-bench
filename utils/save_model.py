# save model 
import os
import torch

# -----------------------------
# PyTorch state_dict 저장
# -----------------------------
def save_pth(model, save_dir, file_name="model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), path)
    print(f"[INFO] state_dict 모델 저장 완료: {path}")
    return path

# -----------------------------
# PyTorch 전체 모델 저장
# -----------------------------
def save_pt(model, save_dir, file_name="model.pt"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    torch.save(model, path)
    print(f"[INFO] 전체 모델 저장 완료: {path}")
    return path

# -----------------------------
# TorchScript(JIT) 저장
# -----------------------------
def save_torchscript(model, save_dir, file_name="model_ts.pt", example_input=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    
    if example_input is None:
        raise ValueError("TorchScript export requires example_input for tracing.")
    
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(path)
    print(f"[INFO] TorchScript 모델 저장 완료: {path}")
    return path

# -----------------------------
# ONNX 저장
# -----------------------------
def save_onnx(model, save_dir, file_name="model.onnx", example_input=None, opset_version=11):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    
    if example_input is None:
        raise ValueError("ONNX export requires example_input.")
    
    model.eval()
    torch.onnx.export(
        model,
        example_input,
        path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"[INFO] ONNX 모델 저장 완료: {path}")
    return path

# -----------------------------
# 예시 통합 함수
# -----------------------------
def save_model_all_formats(model, save_dir, file_prefix="model", example_input=None):
    paths = {}
    paths['pth'] = save_pth(model, save_dir, f"{file_prefix}.pth")
    paths['pt'] = save_pt(model, save_dir, f"{file_prefix}.pt")
    
    if example_input is not None:
        paths['torchscript'] = save_torchscript(model, save_dir, f"{file_prefix}_ts.pt", example_input)
        paths['onnx'] = save_onnx(model, save_dir, f"{file_prefix}.onnx", example_input)
    
    return paths


# -----------------------------
# 예시 통합 함수
# -----------------------------
# def save_model_all_formats(model, save_dir, file_prefix="model", example_input=None):
#     paths = {}
#     paths['pth'] = save_pth(model, save_dir, f"{file_prefix}.pth")
#     paths['pt'] = save_pt(model, save_dir, f"{file_prefix}.pt")
    
#     if example_input is not None:
#         paths['torchscript'] = save_torchscript(model, save_dir, f"{file_prefix}_ts.pt", example_input)
#         paths['onnx'] = save_onnx(model, save_dir, f"{file_prefix}.onnx", example_input)
    
#     return paths
