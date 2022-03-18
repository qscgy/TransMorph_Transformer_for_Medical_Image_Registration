from torch.utils.tensorboard import SummaryWriter
import os, glob
import utils
import sys
import pickle
from losses import AffineChamferLoss
from torch.utils.data import DataLoader
import numpy as np
import torch, TransMesh_affine
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
from TransMesh_affine import CONFIGS as CONFIGS_TM
from natsort import natsorted
from datasets import MeshDataset

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)

def main():
    batch_size = 1
    with open('affine_splits.pkl','rb') as f:
        splits = pickle.load(f)
    train_paths = splits['train_list']
    val_paths = splits['val_list']
    save_dir = 'TransMesh_Affine/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    lr = 0.00004
    epoch_start = 0
    max_epoch = 500
    cont_training = False

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMesh-Affine']
    model = TransMesh_affine.TransMeshAffine(config)
    model.cuda()

    '''
    Continue training
    '''
    if cont_training:
        epoch_start = 335
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_set = MeshDataset(train_paths)
    val_set = MeshDataset(val_paths)
    train_loader = DataLoader(train_set, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=True)

    # Optimizers
    optimizer = optim.AdamW(model.parameters(), lr=updated_lr, amsgrad=True)

    aff_loss = AffineChamferLoss()

    best_mse = 1e10
    writer = SummaryWriter(log_dir=save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]

            ####################
            # Affine transform
            ####################
            x = data[0]
            y = data[1]
            # x_in = torch.cat((x, y), dim=1)
            mat, inv_mat, trans = model(x, y)
            loss, _ = aff_loss(y, x, mat, trans)
            loss.backward()
            optimizer.step()
            loss_all.update(loss.item(), x.num_verts_per_mesh()[0])
            print('Iter {} of {} loss {:.6f}'.format(idx, len(train_loader), loss.item()))
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {}, loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''
        eval_mse = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                # x_in = torch.cat((x, y), dim=1)
                mat, inv_mat, trans = model(x)
                loss, _ = aff_loss(y, x, mat, trans)
                # eval_mse.update(mse.item(), x.size(0))
                print(eval_mse.avg)
        best_mse = min(eval_mse.avg, best_mse)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='mse{:.4f}.pth.tar'.format(eval_mse.avg))
        writer.add_scalar('Loss_GF/val', eval_mse.avg, epoch)
        # TODO make some plots
        # plt.switch_backend('agg')
        # xcat_fig = comput_fig(phan)
        # ct_fig = comput_fig(x)
        # aff_fig = comput_fig(ct_aff)
        # writer.add_figure('Aff Def', aff_fig, epoch)
        # plt.close(aff_fig)
        # writer.add_figure('phan', xcat_fig, epoch)
        # plt.close(xcat_fig)
        # writer.add_figure('ct', ct_fig, epoch)
        # plt.close(ct_fig)
        loss_all.reset()
        eval_mse.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, 72:88, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[1]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, i, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[-1])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()