import os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -----------------------------
# Dataset (YOLO txt format)
# -----------------------------
class SingleClassDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=320):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.*')))
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                             transforms.ToTensor()])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base+".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path,'r') as f:
                for l in f.read().strip().splitlines():
                    if l == '': continue
                    cls, xc, yc, w, h = map(float, l.split())
                    x1 = (xc - w/2) * self.img_size
                    y1 = (yc - h/2) * self.img_size
                    x2 = (xc + w/2) * self.img_size
                    y2 = (yc + h/2) * self.img_size
                    boxes.append([x1,y1,x2,y2])
        boxes = torch.tensor(boxes,dtype=torch.float32)
        return img, boxes

def collate_fn(batch):
    imgs, boxes = list(zip(*batch))
    imgs = torch.stack(imgs)
    return imgs, boxes

# -----------------------------
# LiteDetector (same as before)
# -----------------------------
class DWConv(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,s=1,p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch,in_ch,k,s,p,groups=in_ch,bias=False)
        self.pw = nn.Conv2d(in_ch,out_ch,1,1,0,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self,x):
        return self.act(self.bn(self.pw(self.dw(x))))

class Residual(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.conv1 = DWConv(ch,ch)
        self.conv2 = DWConv(ch,ch)
    def forward(self,x):
        return x + self.conv2(self.conv1(x))

class SPP_Lite(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool1 = nn.MaxPool2d(3,1,1)
        self.pool2 = nn.MaxPool2d(5,1,2)
        self.pool3 = nn.MaxPool2d(7,1,3)
        self.conv = nn.Conv2d(in_ch*4, out_ch, 1,1,0)  # concat 후 채널 줄이기
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self,x):
        x = torch.cat([x, self.pool1(x), self.pool2(x), self.pool3(x)], dim=1)
        return self.act(self.bn(self.conv(x)))

class LiteBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = DWConv(3,32,3,2,1)
        self.stage1 = nn.Sequential(DWConv(32,64,3,2,1), Residual(64))
        self.stage2 = nn.Sequential(DWConv(64,128,3,2,1), Residual(128), Residual(128))
        self.stage3 = nn.Sequential(
            DWConv(128,256,3,2,1),
            Residual(256),
            SPP_Lite(256,256)  # SPP output 채널 256으로 맞춤
        )
    def forward(self,x):
        p1 = self.stage1(self.stem(x))
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        return [p1,p2,p3]

class LiteFPN(nn.Module):
    def __init__(self,in_chs=[64,128,256],out_ch=128):
        super().__init__()
        self.lateral = nn.ModuleList([DWConv(c,out_ch,1,1,0) for c in in_chs])
        self.smooth = nn.ModuleList([DWConv(out_ch,out_ch,3,1,1) for _ in in_chs])
    def forward(self,feats):
        feats = [l(f) for l,f in zip(self.lateral,feats)]
        td2 = feats[2]
        td1 = feats[1] + F.interpolate(td2,size=feats[1].shape[2:],mode='nearest')
        td0 = feats[0] + F.interpolate(td1,size=feats[0].shape[2:],mode='nearest')
        return [s(td) for s,td in zip(self.smooth,[td0,td1,td2])]

class Head(nn.Module):
    def __init__(self,in_ch=128):
        super().__init__()
        self.obj = nn.Conv2d(in_ch,1,1)
        self.reg = nn.Conv2d(in_ch,4,1)
    def forward(self,feats):
        return [self.obj(f) for f in feats],[self.reg(f) for f in feats]

class LiteDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LiteBackbone()
        self.neck = LiteFPN()
        self.head = Head()
    def forward(self,x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)

# -----------------------------
# Anchor-free target generation + loss
# -----------------------------
def generate_targets(feats, boxes, img_size):
    # 간단: 모든 feature map에 stride 고려해서 center point gt
    obj_targets = []
    reg_targets = []
    strides = [4,8,16]
    for i,f in enumerate(feats):
        B,C,H,W = f.shape
        obj_t = torch.zeros((B,1,H,W), device=f.device)
        reg_t = torch.zeros((B,4,H,W), device=f.device)
        for b in range(B):
            for box in boxes[b]:
                xc = (box[0]+box[2])/2 / strides[i]
                yc = (box[1]+box[3])/2 / strides[i]
                w = (box[2]-box[0])/strides[i]
                h = (box[3]-box[1])/strides[i]
                xidx = int(xc)
                yidx = int(yc)
                if 0<=xidx<W and 0<=yidx<H:
                    obj_t[b,0,yidx,xidx] = 1.0
                    reg_t[b,:,yidx,xidx] = torch.tensor([xc,yc,w,h], device=f.device)
        obj_targets.append(obj_t)
        reg_targets.append(reg_t)
    return obj_targets, reg_targets

def loss_fn(pred_obj, pred_reg, obj_targets, reg_targets):
    loss_obj = 0
    loss_reg = 0
    for po, pr, to, tr in zip(pred_obj,pred_reg,obj_targets,reg_targets):
        loss_obj += F.binary_cross_entropy_with_logits(po,to)
        mask = to>0
        if mask.sum()>0:
            loss_reg += F.l1_loss(pr[mask.expand_as(pr)], tr[mask.expand_as(tr)])
    return loss_obj + loss_reg

# -----------------------------
# Training Loop
# -----------------------------
def train():
    image_dir = "data/sausage/images"
    label_dir = "data/sausage/labels"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = SingleClassDetectionDataset(image_dir,label_dir)
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4,pin_memory=True, collate_fn=collate_fn)

    model = LiteDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

    epochs = 10
    for ep in range(epochs):
        model.train()
        for imgs, boxes in dataloader:
            imgs = imgs.to(device)
            obj_targets, reg_targets = generate_targets(model(imgs)[0], boxes, img_size=320)
            optimizer.zero_grad()
            pred_obj, pred_reg = model(imgs)
            loss = loss_fn(pred_obj,pred_reg,obj_targets,reg_targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {ep+1}/{epochs} done, loss={loss.item():.4f}")

    # TorchScript 저장
    os.makedirs("./output",exist_ok=True)
    dummy_input = torch.randn(1,3,320,320).to(device)
    scripted_model = torch.jit.trace(model,dummy_input)
    scripted_model.save("./output/lite_detector_scripted.pt")
    print("학습 완료 & TorchScript 저장!")

if __name__=="__main__":
    train()
