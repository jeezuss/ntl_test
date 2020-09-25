import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False  # Установить True для первого обучения, дальше False


class MandF():
    # класс для обработки изображений
    IMG_SIZE = 50  # размер изображения, к которому будет производиться нормализация
    MALE = "D:/internship_data/male"  # директория с фото мужчин
    FEMALE = "D:/internship_data/female"  # директория с фото женщин
    LABELS = {MALE: 0, FEMALE: 1}  # метки
    training_data = []

    malecount = 0
    femalecount = 0

    def make_training_data(self):
        # перебираем все jpg файлы в директориях и нормализуем их
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[
                            self.LABELS[label]]])
                        if label == self.MALE:
                            self.malecount += 1
                        elif label == self.FEMALE:
                            self.femalecount += 1

                    except Exception as e:
                        pass
                        # print(label, f, str(e))

        np.random.shuffle(self.training_data)  # перемешивание данных
        np.save("training_data.npy", self.training_data)
        print('Male:', MandF.malecount)
        print('Female:', MandF.femalecount)


class Net(nn.Module):
    # класс описывающий модель
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
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


if torch.cuda.is_available():
    # Проверяем, если есть возможность запустить на gpu
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

net = Net().to(device)

if REBUILD_DATA:
    # Если True нормализуем и перемешиваем данные
    mandf = MandF()
    mandf.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# преобразовываем данные в тензоры и разбиваем на тренировочные и тестрируемые
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))


def train(net):
    # функция тренировки модели
    BATCH_SIZE = 100
    EPOCHS = 6  # выбрано 6 путем перебора разных вариантов
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i + BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(loss)  # вывод функции потерь


train(net)


def test(net):
    # функция проверки
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy:", round(correct / total, 3))  # выводит точность


test(net)

# сохраняем модель
PATH = 'D:\mf.pth'
torch.save(net.state_dict(), PATH)
