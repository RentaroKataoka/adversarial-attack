import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from CycleGAN_model import Generator,Discriminator,init_weights
from CycleGAN_utils import ImagePool,BasicDataset
import argparse
import time 
import os
import sys
from itertools import chain
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
from memory_profiler import profile
import pdb 
from torchvision.datasets import ImageFolder
import random

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


pretrained_model = "./model/googlefonts.pth" #事前学習済みMNISTモデル(重みパラメータ)
use_cuda = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(57600, 128)
        self.fc2 = nn.Linear(128, 26)

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
        output = F.log_softmax(x, dim=1)
        return output

# 使うデバイス（CPUかGPUか）の定義
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# ネットワークの初期化
model = Net().to(device)
print(model)
# 訓練済みモデルのロード
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
# モデルを評価モードに設定。本チュートリアルの例では、これはドロップアウト層等を評価モードにするのに必要
model.eval()



def attack(data, data_grad, target, dirname_res, dirname_pro, chr, count, epsilon, lim, success):
    os.makedirs(dirname_pro + chr + "/{}".format(count), exist_ok=True)
    # os.makedirs(dirname_pro + chr + "/{}".format(i), exist_ok=True)
    for i in range(1, 10001):
        data.requires_grad = False
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        # perturbed_data += (perturbed_data < torch.Tensor([1 - lim]).to("cuda")) * epsilon + (perturbed_data < torch.Tensor([0]).to("cuda")) * -epsilon + (perturbed_data > torch.Tensor([-1 + lim]).to("cuda")) * -epsilon + (perturbed_data > torch.Tensor([0]).to("cuda")) * epsilon
        perturbed_data = torch.clamp(perturbed_data, -1, 1)
        data = perturbed_data
        data.requires_grad = True
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.max(1, keepdim=True)[1]
        # plt.xticks([], [])
        # plt.yticks([], [])
        # plt.imsave(dirname_pro + chr + "/{}".format(count) + "/" + "{}.png".format(i), data.squeeze().detach().cpu().numpy(), cmap="gray")
        if pred.item() != target.item():
            success += 1
            break
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
    # os.makedirs(dirname_res + chr + "/{}".format(i), exist_ok=True)
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.imsave(dirname_res + chr + "/{}".format(i) + "/" + "{}.png".format(count), data.squeeze().detach().cpu().numpy(), cmap="gray")
    return data, pred, success, i


class loss_scheduler():
    def __init__(self, args):
        self.epoch_decay = args.epoch_decay

    def f(self, epoch):
        #ベースの学習率に対する倍率を返す(pytorch仕様)
        if epoch<=self.epoch_decay:
            return 1
        else:
            scaling = 1 - (epoch-self.epoch_decay)/float(self.epoch_decay)
            return scaling


def set_requires_grad(models, requires=False):
    if not isinstance(models,list):
        models = [models]
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires



def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation: CycleGAN')
    #for train
    parser.add_argument('--image_size', '-i', type=int, default=64, help='input image size')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--epoch_decay', '-ed', type=int, default=100,
                        help='Number of epochs to start decaying learning rate to zero')                    
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--pool_size', type=int, default=50, help='for discriminator: the size of image buffer that stores previously generated images')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Assumptive weight of cycle consistency loss')
    parser.add_argument('--lambda_identity', type=float, default=15.0, help='Assumptive weight of identity mapping loss')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    #for save and load
    parser.add_argument('--sample_frequecy', '-sf', type=int, default=5000,
                        help='Frequency of taking a sample')
    parser.add_argument('--checkpoint_frequecy', '-cf', type=int, default=10,
                        help='Frequency of taking a checkpoint')
    parser.add_argument('--data_name', '-d', default="horse2zebra", help='Dataset name')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--log_dir', '-l', default='logs/',
                        help='Directory to output the log')
    parser.add_argument('--model', '-m', help='Model name')
    args = parser.parse_args()



    #set GPU or CPU
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    #set depth of resnet
    if args.image_size == 64:
        res_block=3
    else:
        res_block=9
    
    #set models
    G_A2B = Generator(1,res_block).to(device)
    G_B2A = Generator(1,res_block).to(device)
    D_A = Discriminator(1).to(device)
    D_B = Discriminator(1).to(device)

    # data pararell
    # if device == 'cuda':
    #     G_A2B = torch.nn.DataParallel(G_A2B)
    #     G_B2A = torch.nn.DataParallel(G_B2A)
    #     D_A = torch.nn.DataParallel(D_A)
    #     D_B = torch.nn.DataParallel(D_B)
    #     torch.backends.cudnn.benchmark=True


    #init weights
    G_A2B.apply(init_weights)
    G_B2A.apply(init_weights)
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    #set loss functions
    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    #set optimizers
    optimizer_G = torch.optim.Adam(chain(G_A2B.parameters(),G_B2A.parameters()),lr=args.lr,betas=(args.beta1,0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    
    scheduler_G = LambdaLR(optimizer_G,lr_lambda=loss_scheduler(args).f)
    scheduler_D_A = LambdaLR(optimizer_D_A,lr_lambda=loss_scheduler(args).f)
    scheduler_D_B = LambdaLR(optimizer_D_B,lr_lambda=loss_scheduler(args).f)

    #dataset loading
    class ImageTransform():
        def __init__(self, mean, std):
            self.data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean, std)
            ])

        def __call__(self, img):
            return self.data_transform(img)
    mean = (0.5,)
    std = (0.5,)
    weak_images = ImageFolder( "./data/GAN_weakest_400", transform = ImageTransform(mean, std))
    strong_images = ImageFolder( "./data/GAN_strongest_400", transform = ImageTransform(mean, std))
    train_weak_loader = torch.utils.data.DataLoader(weak_images, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_strong_loader = torch.utils.data.DataLoader(strong_images, batch_size=args.batch_size, shuffle=True, num_workers=0)

    #######################################################################################

    #train
    total_epoch = args.epoch

    fake_A_buffer = ImagePool()
    fake_B_buffer = ImagePool()

    for epoch in range(total_epoch):
        start = time.time()
        losses = [0 for i in range(6)]

        chr_lambda = lambda a: chr(a + 65)
        dirname_grad = "./GAN_result" + "/grad/"
        dirname_org = "./GAN_result" + "/org/"
        dirname_adv = "./GAN_result" + "/adv/"
        dirname_res = "./GAN_result" + "/resistance/"
        dirname_pro = "./GAN_result" + "/progress/"
        # for c in [chr(i) for i in range(65, 65+26)]:
        #     os.makedirs(dirname_grad + c, exist_ok=True)
        #     os.makedirs(dirname_org + c, exist_ok=True)
        #     os.makedirs(dirname_adv + c, exist_ok=True)
        #     os.makedirs(dirname_res + c, exist_ok=True)
        #     os.makedirs(dirname_pro + c, exist_ok=True)
        #     for d in [chr(i) for i in range(65, 65+26)]:
        #         os.makedirs(dirname_adv + c + "/" + c + "→" + d, exist_ok=True)

        # 精度カウンター
        correct = 0
        success = 0
        count_list = [0] * 26

        for i, ((real_A, label_A), (real_B, label_B)) in enumerate(zip(train_weak_loader, train_strong_loader)):
            #generate image
            label_A = label_A.type(torch.LongTensor).to(device)
            label_B = label_B.type(torch.LongTensor).to(device)
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            fake_A, fake_B = G_B2A(real_B), G_A2B(real_A)
            rec_A, rec_B = G_B2A(fake_B), G_A2B(fake_A)
            if args.lambda_identity>0:
                iden_A, iden_B = G_B2A(real_A), G_A2B(real_B)

            #train generator
            set_requires_grad([D_A,D_B],False)
            optimizer_G.zero_grad()

            pred_fake_A = D_A(fake_A)
            loss_G_B2A = adv_loss(pred_fake_A, torch.tensor(1.0).expand_as(pred_fake_A).to(device))
            
            pred_fake_B = D_B(fake_B)
            loss_G_A2B = adv_loss(pred_fake_B, torch.tensor(1.0).expand_as(pred_fake_B).to(device))

            loss_cycle_A = cycle_loss(rec_A, real_A)
            loss_cycle_B = cycle_loss(rec_B, real_B)
            loss_A_B = cycle_loss(fake_A, real_B)
            loss_B_A = cycle_loss(fake_B, real_A)


            if args.lambda_identity>0:
                loss_identity_A = identity_loss(iden_A,real_A)
                loss_identity_B = identity_loss(iden_B,real_B)
                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A*args.lambda_cycle + loss_cycle_B*args.lambda_cycle + loss_identity_A*args.lambda_cycle*args.lambda_identity + loss_identity_B*args.lambda_cycle*args.lambda_identity + loss_A_B*args.lambda_cycle + loss_B_A*args.lambda_cycle

            else:
                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A*args.lambda_cycle + loss_cycle_B*args.lambda_cycle

            loss_G.backward()
            optimizer_G.step()

            losses[0]+=loss_G_A2B.item()
            losses[1]+=loss_G_B2A.item()
            losses[2]+=loss_cycle_A.item()
            losses[3]+=loss_cycle_B.item()


            #train discriminator
            set_requires_grad([D_A,D_B],True)
            optimizer_D_A.zero_grad()
            pred_real_A = D_A(real_A)
            fake_A_ = fake_A_buffer.get_images(fake_A)
            pred_fake_A = D_A(fake_A_.detach())
            loss_D_A_real = adv_loss(pred_real_A, torch.tensor(1.0).expand_as(pred_real_A).to(device))
            loss_D_A_fake = adv_loss(pred_fake_A, torch.tensor(0.0).expand_as(pred_fake_A).to(device))
            loss_D_A = (loss_D_A_fake + loss_D_A_real)*0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            pred_real_B = D_B(real_B)
            fake_B_ = fake_B_buffer.get_images(fake_B)
            pred_fake_B = D_B(fake_B_.detach())
            loss_D_B_real = adv_loss(pred_real_B, torch.tensor(1.0).expand_as(pred_real_B).to(device))
            loss_D_B_fake = adv_loss(pred_fake_B, torch.tensor(0.0).expand_as(pred_fake_B).to(device))
            loss_D_B = (loss_D_B_fake + loss_D_B_real)*0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            losses[4]+=loss_D_A.item() 
            losses[5]+=loss_D_B.item()

            if epoch + 1 == 100 or epoch + 1 == 200:
                real_A.requires_grad = True
                # データをモデルに順伝播させます
                output_A = model(real_A)
                init_pred_A = output_A.max(1, keepdim=True)[1] # 最大の確率のインデックスを取得します。

                if init_pred_A.item() == label_A.item():
                    data_copy_A = real_A.detach().clone()
                    
                    count_list[init_pred_A.item()] += 1
                
                    # 損失を計算します
                    loss_A = F.nll_loss(output_A, label_A)
                    # 既存の勾配を全てゼロにします
                    model.zero_grad()
                    # 逆伝播させてモデルの勾配を計算します
                    loss_A.backward()
                    # データの勾配を取得します
                    data_grad_A = real_A.grad.data

                    perturbed_data_A, pred_A, success_A, def_A = attack(real_A, data_grad_A, label_A, dirname_res, dirname_pro, chr_lambda(init_pred_A.item()), count_list[init_pred_A.item()], 0.001, 0, success)

                    final_pred_A = pred_A

                    org_A = data_copy_A.squeeze().detach().cpu().numpy()
                    adv_A = perturbed_data_A.squeeze().detach().cpu().numpy()

            
                fake_B = fake_B.detach()
                fake_B.requires_grad = True
                # データをモデルに順伝播させます
                output_B = model(fake_B)

                init_pred_B = output_B.max(1, keepdim=True)[1] # 最大の確率のインデックスを取得します。

                # 最初から予測が間違っている場合、攻撃する必要がないため次のイテレーションに進みます。
                if init_pred_B.item() == label_A.item():
                    data_copy_B = fake_B.detach().clone()
                
                    # 損失を計算します
                    loss_B = F.nll_loss(output_B, label_A)
                    # 既存の勾配を全てゼロにします
                    model.zero_grad()
                    # 逆伝播させてモデルの勾配を計算します
                    loss_B.backward()
                    # データの勾配を取得します
                    data_grad_B = fake_B.grad.data

                    perturbed_data_B, pred_B, success_B, def_B = attack(fake_B, data_grad_B, label_A, dirname_res, dirname_pro, chr_lambda(init_pred_B.item()), count_list[init_pred_B.item()], 0.001, 0, success)

                    final_pred = pred_B

                    org_B = data_copy_B.squeeze().detach().cpu().numpy()
                    adv = perturbed_data_B.squeeze().detach().cpu().numpy()

                # #各条件を満たす画像の保存
                # plt.xticks([], [])
                # plt.yticks([], [])
                # plt.imsave(dirname_org + chr_lambda(init_pred.item()) + "/{}.png".format(count_list[init_pred.item()]), org, cmap="gray")
                
                # os.makedirs(dirname_adv + chr_lambda(init_pred.item()) + "/" + chr_lambda(init_pred.item()) + "→" + chr_lambda(final_pred.item()) + "/", exist_ok=True)
                # plt.xticks([], [])
                # plt.yticks([], [])
                # plt.imsave(dirname_adv + chr_lambda(init_pred.item()) + "/" + chr_lambda(init_pred.item()) + "→" + chr_lambda(final_pred.item()) + "/{}.png".format(count_list[init_pred.item()]), adv, cmap="gray")

                os.makedirs("./sample3/" + str(epoch + 1), exist_ok=True)
                if init_pred_A.item() != label_A.item():
                    def_A = 0
                if init_pred_B.item() != label_A.item():
                    def_B = 0
                real_A = real_A.squeeze().detach().cpu().numpy()
                plt.subplot(1, 2, 1)
                plt.xticks([], [])
                plt.yticks([], [])  
                plt.title("{}".format(def_A))
                plt.imshow(real_A, cmap="gray")
                fake_B = fake_B.squeeze().detach().cpu().numpy()
                plt.subplot(1, 2, 2)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title("{}".format(def_B))
                plt.imshow(fake_B, cmap="gray")
                plt.savefig("./sample3/" + str(epoch + 1) + "/{}.png".format(i))
                      
    
            current_batch = epoch * len(train_weak_loader) + i
            sys.stdout.write(f"\r[Epoch {epoch+1}/200] [Index {i}/{len(train_weak_loader)}] [D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}] [G loss: adv: {loss_G.item():.4f}] [lr: {scheduler_G.get_lr()}]")
            
        
        
        #get tensorboard logs
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        writer.add_scalar('loss_G_A2B', losses[0]/float(len(train_weak_loader)), epoch)
        writer.add_scalar('loss_D_A', losses[4]/float(len(train_weak_loader)), epoch)
        writer.add_scalar('loss_G_B2A', losses[1]/float(len(train_weak_loader)), epoch)
        writer.add_scalar('loss_D_B', losses[5]/float(len(train_weak_loader)), epoch)
        writer.add_scalar('loss_cycle_A', losses[2]/float(len(train_weak_loader)), epoch)
        writer.add_scalar('loss_cycle_B', losses[3]/float(len(train_weak_loader)), epoch)
        writer.add_scalar('learning_rate_G', np.array(scheduler_G.get_lr()), epoch)
        writer.add_scalar('learning_rate_D_A', np.array(scheduler_D_A.get_lr()), epoch)
        writer.add_scalar('learning_rate_D_B', np.array(scheduler_D_B.get_lr()), epoch)
        sys.stdout.write(f"[Epoch {epoch+1}/200] [D_A loss: {losses[4]/float(len(train_weak_loader)):.4f}] [D_B loss: {losses[5]/float(len(train_weak_loader)):.4f}] [G adv loss: adv: {losses[0]/float(len(train_weak_loader))+losses[1]/float(len(train_weak_loader)):.4f}]")
        
        #update learning rate
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        

        os.makedirs("models/G_A2B/", exist_ok=True)
        os.makedirs("models/G_B2A/", exist_ok=True)
        os.makedirs("models/D_A/", exist_ok=True)
        os.makedirs("models/D_B/", exist_ok=True)
        torch.save(G_A2B.state_dict(), "models/G_A2B/"+str(epoch)+".pth")
        torch.save(G_B2A.state_dict(), "models/G_B2A/"+str(epoch)+".pth")
        torch.save(D_A.state_dict(), "models/D_A/"+str(epoch)+".pth")
        torch.save(D_B.state_dict(), "models/D_B/"+str(epoch)+".pth")


if __name__ == "__main__":
    main()