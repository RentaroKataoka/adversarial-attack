from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import random
from torchvision.transforms import InterpolationMode
import PIL
import os

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

torch_fix_seed()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(57600, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    criterion = torch.nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.type(torch.float).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    criterion = torch.nn.MSELoss()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.type(torch.float).to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
    
    test_loss /= len(test_loader)

    print(test_loss)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    
    return test_loss


class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, alphabet="A"):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = "model/reg_" + alphabet + ".pth"    #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")


    class MyDataset(torch.utils.data.Dataset):

        def __init__(self, label_path, transform=None):
            x = []
            y = []
            
            with open(label_path, 'r') as infh:
                for line in infh:
                    d = line.replace('\n', '').split('\t')
                    x.append(os.path.join(os.path.dirname(label_path), d[0]))
                    y.append(float(d[1]))
            
            self.x = x    
            self.y = torch.from_numpy(np.array(y)).float().view(-1, 1)
            
            self.transform = transform
        
        
        def __len__(self):
            return len(self.x)
        
        
        def __getitem__(self, i):
            img = PIL.Image.open(self.x[i]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            
            return img, self.y[i]

    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
    transforms.Normalize((0.5,), (0.5,))])

    for alphabet in [chr(i + 65) for i in range(26)]:
        train_data_dir = 'attack_result2/org/' + alphabet + '/reg_' + alphabet + '_train.tsv'
        val_data_dir = 'attack_result2/org/' + alphabet + '/reg_' + alphabet + '_val.tsv'

        train_dataset = MyDataset(train_data_dir, transform=transform)
        val_dataset = MyDataset(val_data_dir, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)



        # class ImageTransform():
        #     def __init__(self, mean, std):
        #         self.data_transform = transforms.Compose([
        #         transforms.Grayscale(),
        #         transforms.ToTensor(),
        #         transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
        #         transforms.Normalize(mean, std)
        #         ])

        #     def __call__(self, img):
        #         return self.data_transform(img)
        # mean = (0.5,)
        # std = (0.5,)
        
        # alphabet = "C"
        # images_train = ImageFolder("data/GoogleFonts/train_reg/" + alphabet, transform = ImageTransform(mean, std))
        # images_val = ImageFolder( "data/GoogleFonts/val_reg/", transform = ImageTransform(mean, std))
        # batch_size = 4
        # train_loader = DataLoader(images_train, batch_size = batch_size, shuffle = True, drop_last=True)
        # val_loader = DataLoader(images_val, batch_size = batch_size, shuffle = False, drop_last=True)



        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        #★EarlyStoppingクラスのインスタンス化★
        earlystopping = EarlyStopping(patience=10, verbose=True, alphabet=alphabet)

        # for epoch in range(1, args.epochs + 1):
        for epoch in range(1, 10000):
            train(args, model, device, train_loader, optimizer, epoch)
            test_loss =  test(model, device, val_loader)
            #★毎エポックearlystoppingの判定をさせる★
            earlystopping(test_loss, model) #callメソッド呼び出し
            if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
                print("Early Stopping!")
                break
            scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == '__main__':
    main()