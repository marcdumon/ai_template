import torch
from time import time
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import time
import random
import numpy as np

# TODO: SEE https://towardsdatascience.com/https-medium-com-dinber19-take-a-deeper-look-at-your-pytorch-model-with-the-new-tensorboard-built-in-513969cf6a72

class MNISTClass(nn.Module):
    def __init__(self):
        super(MNISTClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1080, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # conv1(kernel=3, filters=15) 28x28x1 -> 26x26x15
        x = F.relu(self.conv1(x))

        # conv2(kernel=3, filters=20) 26x26x15 -> 13x13x30
        # max_pool(kernel=2) 13x13x30 -> 6x6x30
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))

        # flatten 6x6x30 = 1080
        x = x.view(-1, 1080)

        # 1080 -> 100
        x = F.relu(self.fc1(x))

        # 100 -> 10
        x = self.fc2(x)

        # transform to logits
        return F.log_softmax(x, dim=1)


def test_data_recorder(i, pred, writer, target, data, output, epoch):
    global step
    labels_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                   7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    # Undo normalization to show the images on Tensorboard
    denormalize = transforms.Normalize((-1,), (1 / 0.5,))

    # Show some misclassified images in Tensorboard
    if i < 10 and target.data[pred != target.data].nelement() > 0:
        for inx, d in enumerate(data[pred != target.data]):
            img_name = 'Test-misclassified/Prediction-{}/Label-{}_Epoch-{}_{}/'.format(
                labels_dict[pred[pred != target.data].tolist()[inx]],
                labels_dict[target.data[pred != target.data].tolist()[inx]], epoch, i)
            writer.add_image(img_name, denormalize(d), epoch)
            i += 1

    # Record histograms:
    # Randomly pick batches to record (test dataset size = 10000, batch size 32)
    if epoch == 0 and random.randint(1, 100) < 4 or epoch > 0 and \
            random.randint(1, 100) < 2 or epoch == 0 and i < 2:

        image_max, label_conf = [[], [[] for x in range(32)]]
        for t in range(output.size(0)):  # go over all tensors
            prob_out = F.softmax(output[t], dim=0)
            image_max.append(prob_out.max().item())
            for l in range(output.size(1)):  # go over all labels
                label_conf[l].append(prob_out[l].item())

        writer.add_histogram('Max confidence per image', np.array(image_max), step, bins='auto')
        for l in range(output.size(1)):
            writer.add_histogram('Confidence per label, label {}'.format(labels_dict[l]),
                                 np.array(label_conf[l]), step)
        writer.flush()
        step += 1
        print('.', end='')


def train(model, device, train_loader, opt, epoch, writer):
    model.train()

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        preds = model(data)
        loss = F.nll_loss(preds, target)
        loss.backward()
        opt.step()

        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset),
                       100. * batch_id / len(train_loader), loss.item()))
    # Record loss into the writer
    writer.add_scalar('Train/Loss', loss.item(), epoch)
    writer.flush()


def test(model, device, test_loader, epoch, writer):
    model.eval()  # SWITCH TO TEST MODE
    i, test_loss, correct, n = [0, 0, 0, 0]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

            # Record images and data into the writer:
            test_data_recorder(i, pred, writer, target, data, output, epoch)

    test_loss /= len(test_loader)  # loss function already averages over batch size
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    # Record loss and accuracy into the writer
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)
    writer.flush()


def main():
    mnist_path = '/storage/data/Fashion_MNIST/'
    PATH_to_log_dir = '/storage/Blog/TensorBoard/logdir/'

    # Declare Tensorboard writer
    timestr = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(PATH_to_log_dir + timestr)
    print('Tensorboard is recording into folder: ' + PATH_to_log_dir + timestr)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    # Download data and create datasets
    trainset = datasets.FashionMNIST(mnist_path, download=True, train=True, transform=transform)
    valset = datasets.FashionMNIST(mnist_path, download=True, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # To inspect the input dataset visualize the grid
    grid = utils.make_grid(images)
    writer.add_image('Dataset/Inspect input grid', grid, 0)
    writer.close()

    # Create model and optimizer
    model = MNISTClass()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    '''Run!!'''
    device = "cpu"
    global step  # for histogram stack recording
    step = 0
    for epoch in range(0, 10):
        print("Epoch %d" % epoch)
        train(model, device, trainloader, opt, epoch, writer)
        test(model, device, valloader, epoch, writer)
        writer.close()
    print('Tensorboard is recording into folder: ' + PATH_to_log_dir + timestr)


if __name__ == '__main__':
    main()