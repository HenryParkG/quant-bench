from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def visualize(img_path, boxes):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def visualize_normalized_coords(img_path, boxes, figsize=(6,6)):
    """
    boxes: [[x_center, y_center, w, h], ...] - 0~1 normalized
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size
    for xc,yc,w,h in boxes:
        x1 = (xc - w/2) * w_img
        y1 = (yc - h/2) * h_img
        x2 = (xc + w/2) * w_img
        y2 = (yc + h/2) * h_img
        draw.rectangle([x1,y1,x2,y2], outline="blue", width=2)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    