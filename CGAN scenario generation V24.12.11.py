"""
@Date: 2024/12/11

@Author: DREAM_DXC

@Summary: define a condition generation adversarial networks （cGAN） for day-ahead scenario generation

@Paper：Day-ahead Scenario Generation of Renewable Energy Based on Conditional GAN
Proceedings of the Chinese Society of Electrical Engineering, v 40, n 17, p 5527-5535, September 5, 2020
DOI: 10.13334/j.0258-8013.pcsee.190633

@Note: This program is not the original program of the manuscript, but the realization of the function is the consistent
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import grad
import os

parser = argparse.ArgumentParser()
parser.add_argument("--feature_number", type=int, default=1, help="feature number of NWP data")
parser.add_argument("--farm_number", type=int, default=1, help="number of farm")
parser.add_argument("--ratepower", type=int, default=1500, help="Wind farm rated power 1500 MW")
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="RMSprop: learning rate")
parser.add_argument("--interval", type=int, default=1, help="define the interval (The original data is sampled every xx minutes)")
parser.add_argument("--sample_step", type=int, default=96, help="define the sampling step")
parser.add_argument("--input_step", type=int, default=96, help="define the input temporal step")
parser.add_argument("--target_step", type=int, default=96, help="define the target temporal step")
parser.add_argument("--train_path", default='Ireland_Train_Data.xlsx', help="train data path and data file name")
parser.add_argument("--test_path", default='Ireland_Test_Data.xlsx', help="test data path and data file name")
parser.add_argument("--plt_name", default='Wind.pdf', help="plot file name")
parser.add_argument("--test_number", type=int, default=50, help="the number of test sample")
parser.add_argument("--scenario_number", type=int, default=200, help="define the scenario number")
args = parser.parse_args()
print(args)

#gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("train_type:",device.type)

pwd = os.getcwd()
grader_father=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")
father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")

def Train_Test_Data(feature_number,rate_value,interval,step,input_time_step,target_time_step,data_file_path):

    power_data = pd.read_excel(data_file_path,'Sheet1')
    nwp_data = pd.read_excel(data_file_path,'Sheet1')
    Wind_data = power_data[['Wind Generation MW']].values
    NWP_data = nwp_data[['Forecast MW']].values   # day-ahead forecast : NWP

    where_are_nan = np.isnan(Wind_data)
    Wind_data[where_are_nan] = 0            # delete data if data = nan
    Wind_data[Wind_data < 0] = 0

    Len_data = Wind_data.shape[0]  # data length
    x = np.arange(0, Len_data, interval)
    Wind_data = Wind_data[x]  # get the interval data
    NWP_data = NWP_data[x]

    Len_data = Wind_data.shape[0]
    num = (( Len_data - target_time_step) // step) + 1  # the finally data cant use to train ,lack the target data ,so lack 1 num
    print("sample-number:",num)

    # get target data
    node = np.arange(0, target_time_step, 1)
    x = node
    for i in range(1, num):
        x = np.append(x, node + i * step, axis=0)
    Target_data = Wind_data[x]
    Target_data = Target_data / rate_value
    Target_data = torch.from_numpy(Target_data).float()
    Target_data = Target_data.view(num, target_time_step, 1)
    Target_data = Target_data.clamp(min=1e-5,max=1-(1e-5)) # for numerical stability

    # get NWP input data
    input_node = np.arange(0, input_time_step, 1)
    y = input_node
    for i in range(1, num):
        y = np.append(y, input_node + i * step, axis=0)

    Input_data = NWP_data[y] / rate_value  # forecast power norm! NWP data can not use

    Input_data = torch.from_numpy(Input_data).float()
    Input_data = Input_data.view(num, input_time_step, feature_number)

    # reshape size
    Target_data = Target_data.permute(0, 2, 1)
    Input_data = Input_data.permute(0, 2, 1)

    return  Input_data,Target_data  #Input_data:nwp condition,Target_data:real data

def dataload_data(input_data, target_data, BATCH_SIZE):
    torch_dataset = Data.TensorDataset(input_data, target_data)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch Tensor Dataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
    )
    return train_loader

"""---------------------------------------Generator Model---------------------------------------------"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(nn.Linear(24, 96),
                                    nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1,  # input height
                out_channels=64,  # n_filters/filters number/channels
                kernel_size=(3, 3),  # filter kernel size (height,width)
                stride=(1, 1),  # filter movement/step
                padding=(1, 1),
                dilation=(1, 1)
            ),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # input height
                out_channels=64,  # n_filters/filters number/channels
                kernel_size=(3, 3),  # filter kernel size (height,width)
                stride=(1, 1),  # filter movement/step
                padding=(1, 2),  # padding
                dilation=(1, 2)
            ),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # input height
                out_channels=1,  # n_filters/filters number/channels
                kernel_size=(3, 3),  # filter kernel size (height,width)
                stride=(1, 1),  # filter movement/step
                padding=(1, 3),  # padding
                dilation=(1, 3)
            ),
            nn.Sigmoid(),
        )

        self.padding = nn.ModuleList()
        self.padding.append(nn.ReplicationPad2d(padding=(1, 1, 1, 1)))   # Left, right, up, down
        self.padding.append(nn.ReplicationPad2d(padding=(2, 2, 1, 1)))
        self.padding.append(nn.ReplicationPad2d(padding=(3, 3, 1, 1)))

    def forward(self, x, noise):
        noise = self.linear(noise)
        x = torch.cat((x, noise), 2)
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.conv3(x)[:,0,1,:]
        return output

"""-------------------------------------Discriminator Model-------------------------------------------"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters/filters number/channels
                kernel_size=(3, 3),  # filter kernel size (height,width)
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters/filters number/channels
                kernel_size=(3, 3),  # filter kernel size (height,width)
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 2)
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # input height
                out_channels=2,  # n_filters/filters number/channels
                kernel_size=(3, 3),  # filter kernel size (height,width)
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 3)
            ),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
        )

        self.padding = nn.ModuleList()
        self.padding.append(nn.ReplicationPad2d(padding=(1, 1, 1, 1)))   # Left, right, up, down
        self.padding.append(nn.ReplicationPad2d(padding=(2, 2, 1, 1)))
        self.padding.append(nn.ReplicationPad2d(padding=(3, 3, 1, 1)))

        self.Linear1 = nn.Sequential(
            nn.Linear(384, 1),
        )
    def forward(self, x):
        x = self.padding[0](x)
        x = self.conv1(x)
        x = self.padding[1](x)
        x = self.conv2(x)
        x = self.padding[2](x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output_D = self.Linear1(x)
        return output_D

"""----------------------------------------training Model---------------------------------------------"""
def train():
    opt_D = torch.optim.RMSprop(D.parameters(), lr=args.lr, weight_decay=0.002)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=args.lr, weight_decay=0.002)

    # learning rate decay
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(opt_D, milestones=[500, 2000], gamma=0.5)
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(opt_G, milestones=[500, 2000], gamma=0.5)

    for epoch in range(args.epoch):

        for step, (condition, target) in enumerate(train_loader):

            condition = condition.to(device).unsqueeze(dim=1)
            target = target.to(device).unsqueeze(dim=1)

            noise = torch.randn(condition.size(0), 1, 1, 24)
            noise = noise.to(device)

            output = G(condition, noise)  # cnn output
            output = output.view(output.size(0), 1, 1, args.target_step)

            Fake = torch.cat((output, condition), 2)
            Real = torch.cat((target, condition), 2)

            prob_fake = D(Fake)
            G_loss = - torch.mean(prob_fake)

            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            prob_fake = D(Fake.detach())  # D try to reduce this prob
            prob_real = D(Real)  # D try to increase this prob

            # gradient penalty
            alpha = torch.rand((condition.size(0), 1, 1, 1))
            alpha = alpha.to(device)

            x_hat = alpha * Real.data + (1 - alpha) * Fake.data
            x_hat.requires_grad = True

            pred_hat = D(x_hat)
            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            D_loss = torch.mean(prob_fake) - torch.mean(prob_real) + gradient_penalty

            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            scheduler_G.step()
            scheduler_D.step()

            print('epoch:', epoch, 'step:', step, torch.mean(prob_fake).detach().cpu().numpy(),
                  torch.mean(prob_real).detach().cpu().numpy())

    data_file_name = 'G{ep}.pkl'.format(ep=args.epoch)
    torch.save(G.state_dict(), data_file_name)  # save model parameters

"""-----------------------------------------------Testing----------------------------------------------"""
def test(input_data,target_data):
    G = Generator()

    G.load_state_dict(torch.load('G{ep}.pkl'.format(ep=args.epoch), map_location='cpu'))

    test_noise = torch.randn(args.scenario_number*input_data.size(0), 1, 1, 24)

    condition = input_data.unsqueeze(dim=1).repeat(args.scenario_number,1,1,1)

    test_fore = input_data.numpy()
    test_target = target_data.numpy()

    scenario = G(condition, test_noise)
    scenario = scenario.detach().numpy()
    scenario = scenario.reshape((args.scenario_number, test_target.shape[0], 1, args.target_step),order='C').transpose(1,0,2,3)
    scenario = np.clip(scenario, a_min=0, a_max=1)
    return scenario,test_fore,test_target

def plot_test(scenario,test_fore,test_target):

    time = np.arange(1, args.target_step+1, 1)
    time = np.expand_dims(time, axis=0)  # define time point

    plt.title("epoch: %s" % (args.epoch-1))
    for step1 in range(args.scenario_number):
        plt.plot(time[0, :], scenario[args.test_number,step1,0,:], linewidth=0.2)
    plt.plot(time[0, :], test_fore[args.test_number,0,:], linewidth=2.0)
    plt.plot(time[0, :], test_target[args.test_number,0,:], linewidth=2.0)

    plt.savefig(args.plt_name, dpi=600, format='pdf')
    plt.show()

"""---------------------------------------Starte training----------------------------------------------"""
# (๑•㉨•๑)ฅ finish def

if __name__ == '__main__':
    Train_input, Train_target = Train_Test_Data(args.feature_number, args.ratepower, args.interval, args.sample_step,
                                                args.input_step, args.target_step, args.train_path)
    Test_input, Test_target = Train_Test_Data(args.feature_number, args.ratepower, args.interval, args.target_step,
                                              args.input_step, args.target_step, args.test_path)

    train_loader = dataload_data(input_data=Train_input, target_data=Train_target, BATCH_SIZE=args.batch_size)

    D = Discriminator().to(device)
    print(D)
    G = Generator().to(device)
    print(G)

    train()
    scenario,test_fore,test_target = test(input_data=Test_input,target_data=Test_target)
    plot_test(scenario, test_fore, test_target)