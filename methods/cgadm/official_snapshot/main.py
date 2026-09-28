import torch
from torch.utils.data import DataLoader

import numpy as np

import os
import time
import shutil
from tqdm import tqdm
import time

from utils import load_data
from utils import metric
from utils.early_stop import EarlyStopping
from utils.logger import Logger

from models.diffuse import Model, get_optimizer, get_scheduler

from config.diffuse_config import get_config


def my_collate_train(batch):
    pos_pair = torch.stack([item[0] for item in batch], dim=0)
    idx_batch = [item[1] for item in batch]


    pos_pair = torch.LongTensor(pos_pair)
    idx_batch = torch.LongTensor(idx_batch)

    return [pos_pair, idx_batch]



def directory_name_generate(model, opt):
    directory = os.path.join(opt.checkpoints_dir, model.name, opt.dataset_name, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))
        
    directory = directory + '__' + opt.note
    return directory


def train(Data, opt):
    print(opt)
    print('Building dataloader >>>>>>>>>>>>>>>>>>>')

    train_dataset = Data.train_dataset


    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=my_collate_train)


    device = torch.device("cuda:{0}".format(opt.cuda_device))

    print("building model >>>>>>>>>>>>>>>")
    model = Model(opt, Data, device)
    optimizer = get_optimizer(model, opt)
    scheduler = get_scheduler(optimizer, opt)


    print(device)
    if opt.loadFilename != None:
        checkpoint = torch.load(opt.loadFilename)
        sd = checkpoint['sd']
        optimizer_sd = checkpoint['optimizer_sd']
        scheduler_sd = checkpoint['scheduler_sd']
        model.load_state_dict(sd)
        optimizer.load_state_dict(optimizer_sd)
        scheduler.load_state_dict(scheduler_sd)

    model = model.to(device)


    directory = directory_name_generate(model, opt)
    logger = Logger(opt, directory)
    stop_manager = EarlyStopping(directory, patience=200)


    print("Start training >>>>>>>>>>>>>>>")
    start_epoch = 0
    if opt.loadFilename != None:
        checkpoint = torch.load(opt.loadFilename)
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, opt.epoch):
        model.train()


        total_iters = 0
        iter_start_time = time.time()  # timer for computation per iteration


        loss = model()
        if total_iters % opt.print_freq == 0:
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            logger.print_current_losses(epoch, total_iters, {"Loss": loss.item()}, t_comp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_iters += 1
    
        scheduler.step()
        model.eval()

        val_score, _ = metric.anomaly_evaluate(model, Data, device, opt)

        AUC, PRC, RecK = val_score['AUROC'], val_score['AUPRC'], val_score['RecK']


        logger.print_current_metrics(epoch, total_iters, {"Val_AUC": AUC, "PRC": PRC, "RecK": RecK}, 0)

        stop_manager(PRC, epoch, model, optimizer, scheduler, loss)
        if stop_manager.early_stop:
            print("Early stopping")
            break



    
    best_checkpoint = torch.load(os.path.join(
        directory + '/buffer', '{}_{}.tar'.format(stop_manager.best_score, 'checkpoint')))
    torch.save({
        'sd': best_checkpoint['sd'],
        'opt':opt,
    }, os.path.join(directory, 'best_{}.tar'.format(stop_manager.best_score)))
    
    shutil.rmtree(directory + '/buffer')

    return os.path.join(directory, 'best_{}.tar'.format(stop_manager.best_score))


def test(Data, opt, test_tar):
    # add test_tar into opt, opt = parser.parse_args():
    opt.test_tar = test_tar


    print(opt)
    print('Building dataloader >>>>>>>>>>>>>>>>>>>')

    device = torch.device("cuda:{0}".format(opt.cuda_device))

    print("building model >>>>>>>>>>>>>>>")
    model = Model(opt, Data, device)

    best_checkpoint = torch.load(opt.test_tar)
    model.load_state_dict(best_checkpoint['sd'])
    model = model.to(device)

    model.eval()

    _, test_score = metric.anomaly_evaluate(model, Data, device, opt)
    AUC, PRC, RecK = test_score['AUROC'], test_score['AUPRC'], test_score['RecK']

    print("Test_AUC: {0}, PRC: {1}, RecK: {2}".format(AUC, PRC, RecK))
    # return the results of the test:
    return AUC, PRC, RecK


if __name__ == "__main__":
    opt = get_config()
    AUC_list = []
    PRC_list = []
    RecK_list = []
    for _ in range(10):
        Data = load_data.data_load(opt)

        path = train(Data, opt)
        AUC, PRC, RecK = test(Data, opt, path)
        AUC_list.append(AUC)
        PRC_list.append(PRC)
        RecK_list.append(RecK)
    

    print("Test_AUC_mean: {0}, PRC_mean: {1}, RecK_mean: {2}".format(np.mean(AUC_list), np.mean(PRC_list), np.mean(RecK_list)))
    print("Test_AUC_std: {0}, PRC_std: {1}, RecK_std: {2}".format(np.std(AUC_list), np.std(PRC_list), np.std(RecK_list)))

