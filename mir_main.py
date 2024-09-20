import torch
import torch.nn.functional as F  # torch.tanh
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio
from torch.utils.data import DataLoader
from utils.metric import compress, calculate_top_map#, euclidean_dist,cosine_dist
import utils.datasets_mir as datasets_mir
import time
import os
from utils.utils import *
import logging
from utils.models import *
import argparse
import numpy as np
import os.path as osp


parser = argparse.ArgumentParser(description="MGCH demo")
parser.add_argument('--bits', default='16', type=str,help='binary code length (default: 16)')
parser.add_argument('--gpu', default='0', type=str,help='selected gpu (default: 0)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--LAMBDA1', default=1, type=float, help='hyper-parameter:  (default: 1)')
parser.add_argument('--LAMBDA2', default=0.01, type=float, help='hyper-parameter: (default: 10**-2)')
parser.add_argument('--LAMBDA3', default=1, type=float, help='hyper-parameter:  (default: 10**0)')
parser.add_argument('--LAMBDA4', default=0.01, type=float, help='hyper-parameter: (default: 10**-2)')
parser.add_argument('--LAMBDA5', default=10, type=float, help='hyper-parameter:  (default: 10**1)')
parser.add_argument('--LAMBDA6', default=1, type=float, help='hyper-parameter:  (default: 10**0)')
parser.add_argument('--NUM_EPOCH', default=100, type=int, help='hyper-parameter: EPOCH (default: 100)')
parser.add_argument('--LR_IMG', default=0.001, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR_TXT', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-2)')
parser.add_argument('--LR_J', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-2)')#64bit:0.1,128bit:1
parser.add_argument('--LR_GIMG', default=0.001, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR_GTXT', default=0.001, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR_GLAB', default=0.0001, type=float, help='hyper-parameter: learning rate (default: 10**-4)')
parser.add_argument('--MOMENTUM', default=0.9, type=float, help='hyper-parameter: momentum (default: 0.9)')
parser.add_argument('--WEIGHT_DECAY', default=0.0005, type=float, help='hyper-parameter: weight decay (default: 5*10**-4)')
parser.add_argument('--NUM_WORKERS', default=4, type=int, help='workers (default: 1)')
parser.add_argument('--EVAL', default= False, type=bool,help='')
parser.add_argument('--EPOCH_INTERVAL', default=2, type=int, help='INTERVAL (default: 2)')
parser.add_argument('--EVAL_INTERVAL', default=10, type=int, help='evaluation interval (default: 10)')
parser.add_argument('--MODEL_DIR', default='/home/chen/PycharmProjects/pythonProject/MGCH/checkpoint', type=str, help='')


class Session:
    def __init__(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


        self.train_dataset = datasets_mir.MIRFlickr(train=True)
        self.test_dataset = datasets_mir.MIRFlickr(train=False, database=False)
        self.database_dataset = datasets_mir.MIRFlickr(train=False, database=True)
        # Data Loader (Input Pipeline)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=args.NUM_WORKERS)

        self.database_loader = DataLoader(dataset=self.database_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=args.NUM_WORKERS)

        self.best_it = 0
        self.best_ti = 0


    def define_model(self, code_length):


        self.CodeNet_I = ImgNet(code_len=code_length)
        self.CodeNet_J = JNet(code_len=code_length)
        self.CodeNet_T = TxtNet(code_len=code_length)
        self.gcn_I = GCNLI(code_len=code_length)
        self.gcn_T = GCNLT(code_len=code_length)
        self.gcn_L = GCNL()


        self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=args.LR_IMG, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=args.LR_TXT, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_J = torch.optim.SGD(self.CodeNet_J.parameters(), lr=args.LR_J, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)

        self.opt_GI = torch.optim.Adam(self.gcn_I.parameters(), lr=args.LR_GIMG)
        self.opt_GT = torch.optim.Adam(self.gcn_T.parameters(), lr=args.LR_GTXT)
        self.opt_GL = torch.optim.Adam(self.gcn_L.parameters(), lr=args.LR_GLAB)


    def train(self, epoch, args):

        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()
        self.CodeNet_J.cuda().train()
        self.gcn_I.cuda().train()
        self.gcn_T.cuda().train()
        self.gcn_L.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        self.CodeNet_J.set_alpha(epoch)
        self.gcn_I.set_alpha(epoch)
        self.gcn_T.set_alpha(epoch)

        logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f' % (epoch + 1, args.NUM_EPOCH, self.CodeNet_I.alpha))

        for idx, (F_I, F_T, labels, _) in enumerate(self.train_loader):
            F_I = Variable(torch.FloatTensor(F_I.numpy()).cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            labels = Variable(torch.FloatTensor(labels.numpy()).cuda())

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_J.zero_grad()
            self.opt_GI.zero_grad()
            self.opt_GT.zero_grad()
            self.opt_GL.zero_grad()



            F_I1, code_I = self.CodeNet_I(F_I)
            F_T1, code_T = self.CodeNet_T(F_T)

            J = torch.cat((F_I1, F_T1), 1)
            code_J = self.CodeNet_J(J)

            F_I = F.normalize(F_I)
            F_T = F.normalize(F_T)
            # construct similarity matrix
            #C = torch.cat((2 * F_I, 0.3 * F_T), 1)
            #A_IC = euclidean_dist(C, C)
            #A_IC = torch.exp(-A_IC / 4)
            #A_TC = cosine_dist(C, C)
            #A_TC = torch.exp(-A_TC / 4)
            #C_I = C.mm(C.t()) * A_IC
            #C_T = C.mm(C.t()) * A_TC
            # C= C * 2 - 1
            S_I = euclidean_dist(F_I, F_I)
            S_I = torch.exp(-S_I / 4)
            S_T = cosine_dist(F_T, F_T)
            #S_T = torch.exp(-S_T / 4)

            F_BI, B_GI = self.gcn_I(F_I1, S_I)
            F_BT, B_GT = self.gcn_T(F_T1, S_T)

            S_I = S_I * 2 - 1
            S_T = S_T * 2 - 1

            view1_predict, view2_predict, _ = self.gcn_L(F_I1, F_T1)

            # optimize
            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)
            B_J = F.normalize(code_J)
            B_GI = F.normalize(B_GI)
            B_GT = F.normalize(B_GT)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BJ = B_I.mm(B_J.t())
            BT_BJ = B_T.mm(B_J.t())
            BJ_BJ = B_J.mm(B_J.t())
            B_BGI = B_GI.mm(B_GI.t())
            B_BGT = B_GT.mm(B_GT.t())


            loss1 = F.mse_loss(BI_BI, S_I) + F.mse_loss(BT_BT, S_T) + F.mse_loss(BJ_BJ, S_I) + F.mse_loss(BJ_BJ, S_T)
            loss2 = F.mse_loss(B_BGI, S_I) + F.mse_loss(B_BGT, S_T)
            loss3 = F.mse_loss(BI_BJ, S_I) + F.mse_loss(BT_BJ, S_T)
            loss4 = F.mse_loss(B_I, B_GI) + F.mse_loss(B_T, B_GT)
            loss5 = F.mse_loss(B_I, B_J) + F.mse_loss(B_T, B_J)
            loss6 = calc_loss(view1_predict, view2_predict, labels, labels)
            loss = args.LAMBDA1 * loss1 + args.LAMBDA2 * loss2 + args.LAMBDA3 * loss3 + args.LAMBDA4 * loss4 + args.LAMBDA5 * loss5 + args.LAMBDA6 * loss6

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            self.opt_J.step()
            self.opt_GI.step()
            self.opt_GT.step()
            self.opt_GL.step()

            if (idx + 1) % (len(self.train_dataset) // args.batch_size / args.EPOCH_INTERVAL) == 0: #判断当前迭代次数是否能被间隔迭代次数整除，即是否达到了输出训练损失的时机
                logger.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                            % (epoch + 1, args.NUM_EPOCH, idx + 1, len(self.train_dataset) // args.batch_size,
                                loss.item()))

    def eval(self):
        logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        self.CodeNet_J.eval().cuda()


        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader,  self.CodeNet_I,
                                                              self.CodeNet_T, self.database_dataset, self.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        if (self.best_it + self.best_ti) < (MAP_I2T + MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I

        #self.best_it = max(self.best_it, MAP_I2T)
        #self.best_ti = max(self.best_ti, MAP_T2I)
        logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        logger.info('Best MAP of Image to Text: %.3f, Best MAP of Text to Image: %.3f' % (self.best_it, self.best_ti))
        logger.info('--------------------------------------------------------------------')

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(args.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(args.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def mkdir_multi(path):
    # confirm if the path exists
    isExists = os.path.exists(path)

    if not isExists:
        # if not, create path
        os.makedirs(path)
        print('successfully creat path！')
        return True
    else:
        # if exists, notify
        print('path already exists！')
        return False


def _logging():
    global logger
    # logfile = os.path.join(logdir, 'log.log')
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def main():
    global logdir, args

    args = parser.parse_args()

    sess = Session()

    bits = [int(bit) for bit in args.bits.split(',')]
    for bit in bits:
        logdir = './MGCH-demo/result/mir/'  + str(bit) + '/'
        mkdir_multi(logdir)
        _logging()
        if args.EVAL == True:
            sess.load_checkpoints()
            sess.eval()
        else:
            logger.info('--------------------------train Stage--------------------------')
            sess.define_model(bit)
            for epoch in range(args.NUM_EPOCH):
                # train the Model
                sess.train(epoch, args)
                if (epoch + 1) % args.EVAL_INTERVAL == 0:
                    sess.eval()
                if epoch + 1 == args.NUM_EPOCH:
                    sess.save_checkpoints(step=epoch + 1)



if __name__=="__main__":
    main()



