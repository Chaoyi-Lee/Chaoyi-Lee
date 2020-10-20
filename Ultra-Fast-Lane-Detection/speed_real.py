# Thanks for the contribution of KopiSoftware https://github.com/KopiSoftware

import torch
import time
import numpy as np
from model.model import parsingNet
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import scipy.special, tqdm

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def resize(x, y):
    global cap
    cap.set(3,x)
    cap.set(4,y)

def test_practical_without_readtime():
    global cap
    for i in range(10):
        _,img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda()+1
        y = net(x)
        
    print("pracrical image input size:",img.shape)
    print("pracrical tensor input size:",x.shape)
    t_all = []
    for i in range(100):
        _,img = cap.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda()+1

        t1 = time.time()
        y = net(x)
        t2 = time.time()
        t_all.append(t2 - t1)
        
    print("practical with out read time:")
    print('\taverage time:', np.mean(t_all) / 1)
    print('\taverage fps:',1 / np.mean(t_all))
    
    # print('fastest time:', min(t_all) / 1)
    # print('fastest fps:',1 / min(t_all))
    
    # print('slowest time:', max(t_all) / 1)
    # print('slowest fps:',1 / max(t_all))
 
    
def test_practical():
    global cap
    while True:
        _,img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda()+1
        out = net(x)


        col_sample = np.linspace(0, 800 - 1, 200)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(200) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        # import pdb; pdb.set_trace()
        for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            cv2.circle(vis,ppp,5,(0,255,0),-1)








        cv2.imshow('My Image', img)

        cv2.waitKey(1)

    
    # print('fastest time:', min(t_all) / 1)
    # print('fastest fps:',1 / min(t_all))
    
    # print('slowest time:', max(t_all) / 1)
    # print('slowest fps:',1 / max(t_all))
    
###x = torch.zeros((1,3,288,800)).cuda() + 1
def test_theoretical():
    x = torch.zeros((1,3,288,800)).cuda() + 1
    for i in range(10):
        y = net(x)
    
    t_all = []
    for i in range(100):
        t1 = time.time()
        y = net(x)
        t2 = time.time()
        t_all.append(t2 - t1)
    print("theortical")
    print('\taverage time:', np.mean(t_all) / 1)
    print('\taverage fps:',1 / np.mean(t_all))
    
    # print('fastest time:', min(t_all) / 1)
    # print('fastest fps:',1 / min(t_all))
    
    # print('slowest time:', max(t_all) / 1)
    # print('slowest fps:',1 / max(t_all))
    



if __name__ == "__main__":
    ###captrue data from camera or video
    #cap = cv2.VideoCapture("video.mp4") #uncommen to activate a video input
    cap = cv2.VideoCapture(0) #uncommen to activate a camera imput
    #resize(480, 640) #ucommen to change input size
    
    
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    net = parsingNet(pretrained = False, backbone='18',cls_dim = (100+1,56,4),use_aux=False).cuda()
    # net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,18,4),use_aux=False).cuda()
    net.eval()
    

    #test_practical_without_readtime()
    test_practical()
    cap.release()
    #test_theoretical()    
