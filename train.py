import torch
import argparse
import numpy as np
import random
import time
import os
import tqdm
from torch.autograd import Variable
import dataset
import logging
from model import resnet50_woroi_v2
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup_seed(seed):
    print("random seed is set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def metric(pred, targets):
    pred = np.array(pred)
    targets = np.array(targets)
    num, num_class = pred.shape
    gt_num = np.zeros(num_class)
    tp_num = np.zeros(num_class)
    pred_num = np.zeros(num_class)

    for index in range(num_class):
        score = pred[:, index]
        target = targets[:, index]
        gt_num[index] = np.sum(target == 1)
        pred_num[index] = np.sum(score >= 0.5)
        tp_num[index] = np.sum(target * (score >= 0.5))

    pred_num[pred_num == 0] = 1

    OP = np.sum(tp_num) / np.sum(pred_num)
    OR = np.sum(tp_num) / np.sum(gt_num)
    OF1 = (2 * OP * OR) / (OP + OR)

    return OP, OR, OF1

def metric_mAP(pred, targets):
    pred = np.array(pred)
    targets = np.array(targets)
    num, num_class = pred.shape
    aps = np.zeros(num_class, dtype=np.float64)
    for cls_id in range(num_class):
        score = pred[:, cls_id]
        target = targets[:, cls_id]

        tmp = np.argsort(-score)
        target = target[tmp]

        pre, obj = 0, 0
        for i in range(num):
            if target[i] == 1:
                obj += 1.
                pre += obj / (i+1)
        pre /= obj
        aps[cls_id] = pre
    mAP = np.mean(aps)
    return mAP


def metric_imgAcc(pred,targets):
    pred = np.array(pred)
    targets = np.array(targets)
    pred = np.reshape(pred, (-1, 5))
    targets = np.reshape(targets, (-1, 5))
    tp = 0
    fp = 0
    sum = 0
    for indx, one in enumerate(pred):
        one[one<0.5] = 0.
        one[one >= 0.5] = 1.
        if (one == targets[indx]).all():
            tp += 1
        else:
            fp += 1
        sum += 1

    acc = (tp / sum)
    return acc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--Data_format", default='voc', type=str)
    parser.add_argument("--data_root", default="", type=str)
    parser.add_argument("--epoches", default=100, type=int)
    parser.add_argument("--nums_cls", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--resize", default=768, type=int)
    parser.add_argument("--exp_save", default='logs/new_dataset_kaggle_v7', type=str)


    args = parser.parse_args()

    # 初始化随机种子
    setup_seed(args.seed)

    # 开辟训练过程中的一些日志及模型权重文件存储位置
    exp_name = args.exp_save
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    exp_dir = os.path.join("./logs", exp_name + '_' + time_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    results_train_file = open(exp_dir + '/results_train.csv', 'w')
    results_train_file.write('epoch, train_acc, train_loss\n')
    results_train_file.flush()
    results_test_file = open(exp_dir + '/results_test.csv', 'w')
    results_test_file.write('epoch, test_mAP, test_OP, test_OR, test_OF1\n')
    results_test_file.flush()

    log = logging.getLogger("test")
    log.setLevel(logging.INFO)

    #保存best模型
    max_test_acc = 0
    max_test_mAP = 0


    # 数据准备，建立Dataloader

    trainset = dataset.STEEL_ClsDataset('data_luo/last_train_kaggle_5.txt', voc_root=(args.data_root + "train/"), resize=args.resize,num_class=5)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = dataset.STEEL_ClsDataset('data_luo/val_kaggle_5.txt', voc_root=(args.data_root + "val/"),
                                           resize=args.resize, num_class=5)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 构建模型
    log.warning('==> Building model..')
    net = resnet50_woroi_v2(num_classes=5)


    # 导入模型初始化权重
    filename = 'logs/kaggle_v7_withROILoss_03-17-11-22/model_best_acc_37.pth'
    pretrained_path = filename
    if pretrained_path:
        log.warning('load pretrained backbone')
        net_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
    log.warning('==> Successfully Building model..')
    net.train()

    # 建立优化器
    optimizer = optim.SGD([
        {'params': nn.Sequential(*list(net.children())[7:]).parameters(), 'lr': args.lr},
        {'params': nn.Sequential(*list(net.children())[:7]).parameters(), 'lr': args.lr/10}

    ],
        momentum=args.momentum, weight_decay=args.weight_decay)

    # 开始训练迭代
    log.warning("开始训练")
    results_train_file = open(exp_dir + '/results_train.csv', 'a')
    for epoch in range(args.epoches):
        torch.cuda.empty_cache()  # 释放显存
        train_loss = 0
        correct = 0
        correct_target = []
        correct_roi = 0
        correct_res = 0
        total = 0
        idx = 0

        log.warning('Epoch: %d' % epoch)
        tarm = tqdm.tqdm(trainloader, ncols=100)
        for step, data in enumerate(tarm):
            idx = step
            optimizer.zero_grad()
            img = data['img']
            label = data['label']
            if use_cuda:
                img, label = img.cuda(), label.cuda()
            img, label = Variable(img), Variable(label)
            if epoch <= 200:
                acc_ret, loss_ret = net.forward(img, label, TRAIN=True, roi_train=False)
            else:
                acc_ret, loss_ret = net.forward(img, label, TRAIN=True, roi_train=True)
            loss_mean = loss_ret["loss"]
            loss_mean.backward()
            optimizer.step()

            train_loss += loss_mean.data
            total += label.size(0)
            correct += acc_ret['acc']
            correct_roi += acc_ret['acc_cam']
            correct_res += acc_ret['acc_resnet']


        train_acc = 100. * correct / total
        train_acc_resnet = 100. * correct_res / total
        train_acc_roi = 100. * correct_roi / total
        train_loss = train_loss / (idx + 1)

        log.warning('Iteration %d, train_acc = %.4f, train_loss = %.4f, train_acc_resnet = %.4f, train_acc_roi = %.4f'
                     % (epoch, train_acc, train_loss, train_acc_resnet, train_acc_roi))
        results_train_file.write('%d, %.4f,%.4f\n' % (epoch, train_acc, train_loss))
        results_train_file.flush()

        # 测试集上验证
        results_test_file = open(exp_dir + '/results_test.csv', 'a')
        with torch.no_grad():
            net.eval()
            pred_sum = []
            target_sum = []
            val_tarm = tqdm.tqdm(testloader, ncols=100)
            for val_step, val_data in enumerate(val_tarm):
                idx = val_step
                img = val_data['img']
                label = val_data['label']
                if use_cuda:
                    img, label = img.cuda(), label.cuda()
                img, label = Variable(img), Variable(label)
                if epoch <= 100:
                    out_mean, out, out_roi = net.forward(img, label, TRAIN=False, roi_train=False)
                else:
                    out_mean, out, out_roi = net.forward(img, label, TRAIN=False, roi_train=True)
                for idx in range(len(out_mean)):
                    pred_sum.append(out_mean[idx].cpu().detach().numpy())
                    target_sum.append(label[idx].cpu().detach().numpy())

            OP, OR, OF1 = metric(pred_sum, target_sum)
            test_acc = metric_imgAcc(pred_sum, target_sum)
            # print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))
            mAP = metric_mAP(pred_sum, target_sum)
            # print("mAP: {:4f}".format(np.mean(mAP)))

            log.warning(
                'Iteration %d, ,test_Acc = %.4f,test_mAP = %.4f, test_OP = %.4f, test_OR = %.4f, test_OF1 = %.4f' % (
                    epoch, test_acc, mAP, OP, OR, OF1))
            results_test_file.write('%d, %.4f, %.4f,%.4f, %.4f, %.4f\n' % (epoch, test_acc, mAP, OP, OR, OF1))
            results_test_file.flush()
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(net.state_dict(), os.path.join(exp_dir, 'model_best_acc_' + str(epoch) + '.pth'))
            log.warning("mx_test_acc=%.4f, set the epoch : %d" % (max_test_acc, epoch))
        if mAP > max_test_mAP:
            max_test_mAP = mAP
            torch.save(net.state_dict(), os.path.join(exp_dir, 'model_best_mAP_' + str(epoch) + '.pth'))
            log.warning("mx_test_mAP=%.4f, set the epoch : %d" % (max_test_mAP, epoch))


    torch.save(net.state_dict(), os.path.join(exp_dir, 'model_final.pth'))




if __name__ == '__main__':
    main()




