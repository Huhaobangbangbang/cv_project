import torch, glob, cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt


def preict_one_img(img_path, model_path):
    resnet152 = models.resnet152(pretrained=False).eval()
    resnet152.load_state_dict(torch.load(model_path))

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    # 1.将numpy数据变成tensor
    tran = transforms.ToTensor()
    img = tran(img)
    img = img.to(device)
    # 2.将数据变成网络需要的shape
    img = img.view(1, 3, 224, 224)

    out1 = net(img)
    # out1 = F.softmax(out1, dim=1)
    # proba, class_ind = torch.max(out1, 1)
    #
    # proba = float(proba)
    # class_ind = int(class_ind)
    # # print(proba, class_ind)
    # img = img.cpu().numpy().squeeze(0)
    # # print(img.shape)
    # new_img = np.transpose(img, (1,2,0))
    # # plt.imshow(new_img)
    # plt.title("the predict is %s . prob is %s" % (classes[class_ind], round(proba, 3)))   # round(proba, 3)保留三位小数
    # # plt.show()
    # cv2.imwrite('test_2007_000027.jpg', new_img)


if __name__ == '__main__':
    classes = ["ants", "bees"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = r"/Users/huhao/Documents/2007_000027.jpg"
    model_path = r"/Users/huhao/Documents/fcn16-013-0000.pth"
    preict_one_img(img_path, model_path)
