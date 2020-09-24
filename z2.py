from __future__ import print_function, division
import torch
import torchvision.transforms as transforms
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import argparse
import json


PATH = 'D:/mf.pth'
threshold = 0.55
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

net = Net().to(device)
net.load_state_dict(torch.load(PATH))
net.eval()
print(net)

transform = transforms.Compose([transforms.ToTensor()])
with open('D:/imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

def check_photos(dir):
    res_dic = {}
    for f in os.listdir(dir):
        if "jpg" in f:
            try:
                ppt = dir + f
                img = cv2.imread(ppt, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (50, 50))
                img_t = transform(img)
                batch_t = torch.unsqueeze(img_t, 0)
                net.eval()
                out = net(batch_t)
                _, index = torch.max(out, 1)
                percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
                #print(labels[index[0]], percentage[index[0]].item())
                res_dic[f] = labels[index[0]]
            except Exception as e:
                pass
    return res_dic


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('path', type=dir_path)
    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def main():
    parsed_args = parse_arguments()
    data = check_photos(parsed_args.path)
    with open('process_results.json', 'w') as f:
        json.dump(data, f)
        print('Result in process_results.json')


if __name__ == "__main__":
    main()

