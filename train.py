# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from tools import config_model_dir
import Dataloader
from timm.models.layers import to_2tuple
from net import SwinMFF
from loss import LpLssimLoss
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='SwinMFF')
parser.add_argument('--save_name', default='train',help='name of save dir')
parser.add_argument('--datapath', default='./DUTS_256',help='dataset_path')
parser.add_argument('--epochs', type=int, default=10,help='number of epochs to train')
parser.add_argument('--input_size', type=tuple, default=256,help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,help='32 default')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=5e-4,help='5e-4 default')
parser.add_argument('--model_save_fre', type=int, default=1,help='model save frequence')
parser.add_argument('--resume', default=False, help='continue training the model')
args = parser.parse_args()

#ensure reproduce
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_left_img, train_right_img, test_left_img, test_right_img, label_train_fusion_img,label_val_fusion_img = Dataloader.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         Dataloader.myImageFloder(train_left_img,train_right_img,label_train_fusion_img,augment=False),
         batch_size= args.batch_size, shuffle= True)

TestImgLoader = torch.utils.data.DataLoader(
        Dataloader.myImageFloder(test_left_img,test_right_img,label_val_fusion_img,augment=False),
         batch_size= args.batch_size, shuffle= False)


model_save_path=config_model_dir(resume=args.resume,subdir_name=args.save_name)
writer = SummaryWriter(logdir=model_save_path)
#define models
model = SwinMFF(img_size=to_2tuple(args.input_size),depths=[2, 2, 2, 6, 2, 2],num_heads=[6,6,6,6,6,6])
model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),eps=1e-8,weight_decay=0.0001)#weight_decay=0.0001  lr=5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

if torch.cuda.device_count() > 1:
    model.cuda()

def train(imgL, imgR, img_label):

    if torch.cuda.is_available():
        imgL, imgR, img_label = imgL.cuda(), imgR.cuda(), img_label.cuda()
    critertion1 = nn.MSELoss()
    critertion2 = LpLssimLoss()
    output = model(imgL, imgR)
    loss1 = critertion1(output, img_label)
    loss2 = critertion2(output, img_label)
    loss = 0.2 * loss1 + 0.8 * loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def test(imgL, imgR, img_label):
    if torch.cuda.is_available():
        imgL, imgR, img_label = imgL.cuda(), imgR.cuda(), img_label.cuda()
    critertion1 = nn.MSELoss()
    critertion2 = LpLssimLoss()
    output = model(imgL, imgR)
    loss1 = critertion1(output, img_label)
    loss2 = critertion2(output, img_label)
    loss = 0.2 * loss1 + 0.8 * loss2
    return loss

def main(args):
    start_epoch=0
    for epoch in tqdm(range(start_epoch, args.epochs)):
        print('This is %d-th epoch,' % (epoch),'lr is ',optimizer.param_groups[0]["lr"])
        lr_current=scheduler.get_last_lr()[0]
        writer.add_scalar('lr', lr_current,epoch)
        ## training ##
        model.train()
        total_train_loss = 0
        tqdm_bar_train = tqdm(TrainImgLoader)
        for batch_idx, (imgL, imgR, img_label) in enumerate(tqdm_bar_train):
            train_loss = train(imgL, imgR, img_label)
            torch.cuda.synchronize()
            total_train_loss += train_loss.item()
            # # for check
            grad_sum = 0
            if batch_idx %20 ==0:
                for name, param in model.named_parameters():
                    grad_sum += param.grad.abs().sum()
                grad_sum=round(grad_sum.item(),4)
            writer.add_scalar('Grad sum', grad_sum,(epoch+1)*len(TrainImgLoader)+batch_idx)
            tqdm_bar_train.set_description(
                f'Epoch {epoch}, Step {batch_idx}, Train Loss {train_loss.item():.4f}, Lr {lr_current}, Grad_sum {grad_sum}') if grad_sum!=0 else tqdm_bar_train.set_description(
                f'Epoch {epoch}, Step {batch_idx}, Train Loss {train_loss.item():.4f}, Lr {lr_current}')

        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        #for visilize
        writer.add_scalar('training loss(epoch)', total_train_loss / len(TrainImgLoader),epoch)

        #for SAVE
        if (epoch + 1) % args.model_save_fre == 0:
            checkpoint_data = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler': scheduler.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{}.ckpt".format(model_save_path, str(epoch)))
        ## test ##
        model.eval()
        total_test_loss = 0
        tqdm_bar_val = tqdm(TestImgLoader)
        for batch_idx, (imgL, imgR, img_label) in enumerate(tqdm_bar_val):
            test_loss = test(imgL, imgR, img_label)
            total_test_loss += test_loss.item()
            tqdm_bar_val.set_description(f'Epoch {epoch}, Step {batch_idx}, Test Loss {test_loss.item():.4f}, Lr {lr_current}')
        scheduler.step()
        print('epoch %d total test loss = %.3f' % (epoch,total_test_loss / len(TestImgLoader)))
        # for visilize
        writer.add_scalar('val loss(epoch)', total_test_loss / len(TestImgLoader),epoch)

    writer.close()


if __name__ == '__main__':
    main(args)