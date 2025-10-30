from ultralytics import YOLO
import torch
import torch.nn as nn

def enable_qat(trainer):
    m = trainer.model  # use trainer.model, not a separate wrapper
    m.train()  
    # Optional: fuse common blocks if your modules expose conv/bn
    for mod in m.modules():
        if hasattr(mod, 'conv') and hasattr(mod, 'bn'):
            try:
                torch.ao.quantization.fuse_modules(mod, ['conv', 'bn'], inplace=True)
            except Exception:
                pass
    # Use a per-tensor weight qconfig to avoid per_channel_affine issues
    m.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    torch.ao.quantization.prepare_qat(m, inplace=True)
    m.train()

def convert_qat(trainer):
    trainer.model.eval()
    int8_model = torch.ao.quantization.convert(trainer.model)
    torch.save(int8_model.state_dict(), 'yolo11_qat_int8.pth')

if __name__ == "__main__":
    # 모델, 콜백, 학습 코드 모두 여기 안으로
    model = YOLO('yolo11n.pt')

    def enable_qat(trainer):
        m = trainer.model
        m.train()  # 반드시 train 모드
        m.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        torch.ao.quantization.prepare_qat(m, inplace=True)

    def convert_qat(trainer):
        # 1. eval 모드로 전환
        trainer.model.eval()
        
        # 2. QAT -> INT8 변환 (FakeQuant 모듈 제거)
        int8_model = torch.ao.quantization.convert(trainer.model)
        
        # 3. INT8 모델의 state_dict만 저장 (pickle 안전)
        torch.save(int8_model.state_dict(), 'yolo11_qat_int8.pth')
        
        # 4. 원하면 trainer.model 자체를 INT8으로 업데이트
        trainer.model = int8_model

    model.add_callback('on_pretrain_routine_start', enable_qat)
    model.add_callback('on_fit_end', convert_qat)
    model.train(data='./datasets/data.yaml', epochs=100, amp=False)