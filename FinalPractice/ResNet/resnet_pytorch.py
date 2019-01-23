import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
from torchvision import transforms

start_time = time.time()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)

        # Convolution block 1
        self.cb1_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cb1_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cb1_3 = nn.Conv2d(128, 128, 1, stride=1)
        self.norm_cb1_1 = nn.BatchNorm2d(128)
        self.norm_cb1_2 = nn.BatchNorm2d(128)
        self.norm_cb1_3 = nn.BatchNorm2d(128)
        self.drop_cb1 = nn.Dropout(0.25)

        # Straight block 1
        self.sb1_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.sb1_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.norm_sb1_1 = nn.BatchNorm2d(128)
        self.norm_sb1_2 = nn.BatchNorm2d(128)
        self.drop_sb1 = nn.Dropout(0.25)

        # Convolution block 2
        self.cb2_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cb2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cb2_3 = nn.Conv2d(128, 128, 1, stride=1)
        self.norm_cb2_1 = nn.BatchNorm2d(128)
        self.norm_cb2_2 = nn.BatchNorm2d(128)
        self.norm_cb2_3 = nn.BatchNorm2d(128)
        self.drop_cb2 = nn.Dropout(0.25)

        # Straight block 2
        self.sb2_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.sb2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.norm_sb2_1 = nn.BatchNorm2d(128)
        self.norm_sb2_2 = nn.BatchNorm2d(128)
        self.drop_sb2 = nn.Dropout(0.25)

        # Convolution block 3
        self.cb3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cb3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cb3_3 = nn.Conv2d(128, 128, 1, stride=1)
        self.norm_cb3_1 = nn.BatchNorm2d(128)
        self.norm_cb3_2 = nn.BatchNorm2d(128)
        self.norm_cb3_3 = nn.BatchNorm2d(128)
        self.drop_cb3 = nn.Dropout(0.25)

        # Straight block 3
        self.sb3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.sb3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.norm_sb3_1 = nn.BatchNorm2d(128)
        self.norm_sb3_2 = nn.BatchNorm2d(128)
        self.drop_sb3 = nn.Dropout(0.25)

        # Final part
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 10)
        self.drop = nn.Dropout(0.25)

    # Convolution block 1
    # Has convolution in shortcut
    def convolve_block_1(self, inputs):
        norm = self.norm_cb1_1(inputs)
        convolve1 = self.cb1_1(norm)
        drop = self.drop_cb1(convolve1)
        convolve2 = self.cb1_2(drop)

        # Sum
        norm = self.norm_cb1_2(convolve2)

        shortcut = self.cb1_3(inputs)
        shortcut = self.norm_cb1_3(shortcut)

        add = torch.add(shortcut, norm)
        residuals = F.relu(add)

        return residuals

    # Straight block 1
    # Has not convolution in shortcut
    def straight_block_1(self, inputs):
        norm = self.norm_sb1_1(inputs)
        convolve1 = self.sb1_1(norm)
        drop = self.drop_sb1(convolve1)
        convolve2 = self.sb1_2(drop)

        # Sum
        norm = self.norm_sb1_2(convolve2)
        add = torch.add(norm, inputs)
        residuals = F.relu(add)

        return residuals

    # Convolution block 2
    # Has convolution in shortcut
    def convolve_block_2(self, inputs):
        norm = self.norm_cb2_1(inputs)
        convolve1 = self.cb2_1(norm)
        drop = self.drop_cb2(convolve1)
        convolve2 = self.cb2_2(drop)

        # Sum
        norm = self.norm_cb2_2(convolve2)

        shortcut = self.cb2_3(inputs)
        shortcut = self.norm_cb2_3(shortcut)

        add = torch.add(shortcut, norm)
        residuals = F.relu(add)

        return residuals

    # Straight block 2
    # Has not convolution in shortcut
    def straight_block_2(self, inputs):
        norm = self.norm_sb2_1(inputs)
        convolve1 = self.sb2_1(norm)
        drop = self.drop_sb2(convolve1)
        convolve2 = self.sb2_2(drop)

        # Sum
        norm = self.norm_sb2_2(convolve2)
        add = torch.add(norm, inputs)
        residuals = F.relu(add)

        return residuals

    # Convolution block 3
    # Has convolution in shortcut
    def convolve_block_3(self, inputs):
        norm = self.norm_cb3_1(inputs)
        convolve1 = self.cb3_1(norm)
        drop = self.drop_cb3(convolve1)
        convolve2 = self.cb3_2(drop)

        # Sum
        norm = self.norm_cb3_2(convolve2)

        shortcut = self.cb3_3(inputs)
        shortcut = self.norm_cb3_3(shortcut)

        add = torch.add(shortcut, norm)
        residuals = F.relu(add)

        return residuals

    # Straight block 3
    # Has not convolution in shortcut
    def straight_block_3(self, inputs):
        norm = self.norm_sb3_1(inputs)
        convolve1 = self.sb3_1(norm)
        drop = self.drop_sb3(convolve1)
        convolve2 = self.sb3_2(drop)

        # Sum
        norm = self.norm_sb3_2(convolve2)
        add = torch.add(norm, inputs)
        residuals = F.relu(add)

        return residuals

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2, padding=0)

        # Residual block 1
        x = self.convolve_block_1(x)
        x = self.straight_block_1(x)
        x = F.max_pool2d(x, 2, 2, padding=0)

        # Residual block 2
        x = self.convolve_block_2(x)
        x = self.straight_block_2(x)
        x = F.max_pool2d(x, 2, 2, padding=0)

        # Residual block 3
        x = self.convolve_block_3(x)
        x = self.straight_block_3(x)
        x = F.max_pool2d(x, 2, 2, padding=0)

        # Final part
        x = F.avg_pool2d(x, 2, 2)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    batch_size = 700
    test_batch_size = 700
    epochs = 10
    save_model = False

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
    print("\n\n\nThe program has been finished for --- %s seconds ---" % (time.time() - start_time))