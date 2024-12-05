import torch.utils.data as dataf
import argparse
import os
from cans import *
from model import *
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from utils import GradCAM,show_cam_on_image
from main import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')  # diffGrad 1e-3
parser.add_argument('--pth', type=int, default=1, help='n/o training pth 1-Y')
parser.add_argument('--m', type=int, default=1, help='Selected visualization samples')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def yyjok_main():

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    cnn = YYJOK()
    test_Loss_list = []
    test_Accuracy_list = []
    Recall_list = []
    F1_score_list = []
    Precision_list = []
    Kappa_list = []

    for epochs in range(0, args.epoches):
        test_loss, test_acc, pred, target = train_network(cnn,train_loader,test_loader,args.lr,args.pth)
        Recall = recall_score(target, pred, average='macro', zero_division=1)
        Precision = precision_score(target, pred, average='macro',zero_division=0)
        F1_score = f1_score(target, pred, average='macro')
        kappa = cohen_kappa_score(target, pred)
        test_Loss_list.append(test_loss)
        test_Accuracy_list.append(test_acc)
        Recall_list.append(Recall)
        Precision_list.append(Precision)
        F1_score_list.append(F1_score)
        Kappa_list.append(kappa)
    return test_Loss_list,test_Accuracy_list,Recall_list,Precision_list,F1_score_list,Kappa_list

def cam_nose():
    model = YYJOK()
    target_layers = [model.xxx.xx]
    img_tensor = data[args.m, :,:,:]
    target_category = data[args.m].item()
    img_tensor = np.array(img_tensor).astype('float64')
    input_tensor = torch.Tensor(img_tensor).reshape(1,1,10,104)
    dict1 = torch.load("train.pth")
    model.load_state_dict(dict1)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = torch.Tensor(grayscale_cam).view(1, 1, 10, 52)
    resized_image = F.interpolate(grayscale_cam, size=(10, 52), mode='bilinear', align_corners=False)
    plt.imshow(resized_image.reshape(10, 52))
    plt.xticks(ticks=[0, 9, 19, 29, 39, 49, 51], labels=[1, 10, 20, 30, 40, 50, 52], fontsize=10,
               fontweight='bold',
               fontname='Times New Roman')
    plt.yticks(ticks=[0, 4, 9], labels=[1, 5, 10], fontsize=10, fontweight='bold', fontname='Times New Roman')
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(labelsize=10, labelrotation=0, width=2)
    tick_labels = np.round(cbar.ax.get_yticks(), 1).tolist()
    cbar.ax.set_yticks(cbar.ax.get_yticks())  # 这里设置了FixedLocator
    cbar.ax.set_yticklabels(tick_labels, fontname='Times New Roman', fontweight='bold')
    plt.show()

def cam_gao():
    model = YYJOK()
    target_layers = [model.xxx.xx]
    img_tensor = data[args.m, :,:,:]
    target_category = data[args.m].item()
    img_tensor = np.array(img_tensor).astype('float64')
    input_tensor = torch.Tensor(img_tensor).reshape(1,1,10,104)
    dict1 = torch.load("train.pth")
    model.load_state_dict(dict1)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam3 = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam3 = grayscale_cam3[0, :]
    grayscale_cam3 = torch.Tensor(grayscale_cam3).view(1, 1, 10, 52)
    resized_image3 = F.interpolate(grayscale_cam3, size=(10, 52), mode='bilinear', align_corners=False)
    plt.imshow(resized_image3.reshape(1, 520), aspect=30)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(0, 1, num=9))
    cbar.ax.tick_params(labelsize=10, labelcolor='black')  # 设置刻度数字的大小和颜色
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    plt.xticks(ticks=[0, 100, 200, 300, 400, 520], labels=[380, 512, 643, 775, 906, 1038], fontsize=13,
               fontweight='bold', fontname='Times New Roman')  # 横坐标从1到10
    plt.show()

cnn = YYJOK()
print(cnn)
def main_train():
        test_Loss_list, test_Accuracy_list, Recall_list, Precision_list, F1_score_list, Kappa_list=yyjok_main()
        Loss.append(test_Loss_list)
        Accuracy.append(test_Accuracy_list)
        Precision.append(Precision_list)
        Recall.append(Recall_list)
        F1_score_test.append(F1_score_list)
        Kappa_test.append(Kappa_list)
if __name__ == '__main__':

    main_train()
    cam_nose()
    cam_gao()





