import os
import sys
import time
import numpy as np
import math
import random
from os.path import join,exists
import glob
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.optim
from torch.cuda.amp import autocast, GradScaler
from torch import autograd
from util import automkdir, adjust_learning_rate, evaluation, load_checkpoint, save_checkpoint, makelr_fromhr_cuda, test_video,compute_psnr_torch_list,save_distribution,calcu_scope
import dataloader
from models.common import weights_init, cha_loss,cur_loss
import json
#the training code for pretrain period in realvsr
def setup(rank, world_size):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '23456'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, config):
    torch.cuda.set_device(rank)
    config.device = device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # config.seed = random.randint(1, 10000)
    print("Random Seed: ", config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)

    world_size = len(config.gpus)
    setup(rank, world_size)

    model = config.network
    criterion = getattr(sys.modules[__name__], config.train.loss)()
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = criterion.to(device)

    dist.barrier()
    epoch = load_checkpoint(model, config.path.resume, config.path.checkpoint, weights_init = weights_init, rank = rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.init_lr, weight_decay=0)
    
    config.train.epoch = epoch
    iter_per_epoch = config.train.iter_per_epoch
    epoch_decay = config.train.epoch_decay
    step = 0
    scaler = GradScaler()

    train_batch_size = config.train.batch_size // world_size + max(min(config.train.batch_size % world_size - rank, 1), 0)
    train_dataset = dataloader.loader(config.path.train, data_kind=config.data_kind, mode='train', scale=config.model.scale, 
                                    crop_size=config.train.in_size, num_frame=config.train.num_frame,model=config.model.file)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=config.train.num_workers,
                                            pin_memory=True, drop_last=True)
    datalen=train_dataset.length

    eval_dataset = dataloader.loader(config.path.eval, data_kind=config.data_kind, mode='eval', scale=config.model.scale, 
                                    num_frame=config.train.num_frame)
    eval_sampler=torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.eval.batch_size, shuffle=False, num_workers=config.eval.num_workers, 
                                    pin_memory=True)

    while(epoch < config.train.num_epochs):
        if step == 0:
            adjust_learning_rate(config.train.init_lr, config.train.final_lr, epoch, epoch_decay, step % iter_per_epoch, iter_per_epoch, optimizer, True)
            if rank == 0:
                evaluation(model, eval_loader, config)
            time_start = time.time()
        for iteration, (img_lq,img_hq) in enumerate(train_loader):
            adjust_learning_rate(config.train.init_lr, config.train.final_lr, epoch, epoch_decay, step % iter_per_epoch, iter_per_epoch, optimizer, False)
            optimizer.zero_grad()

            with autocast():     #Automatic Mixed Precision Training
                print("hq shape",img_hq.shape)
                img_de=model(img_hq)
                loss = criterion(img_de, img_lq) 
            loss_v = loss.detach()
            if (loss_v > 50 or loss_v < 0 or math.isnan(loss_v)) and epoch > 0:
                print(f'epoch {epoch}, skip iteration {iteration}, loss {loss_v}')
                raise RuntimeWarning(f'epoch {epoch}, iteration {iteration}, loss {loss_v}')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            step += 1

            # print loss per display_iter
            if (step % config.train.display_iter) == 0 and rank == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), "Epoch[{}/{}]({}/{}): Loss: {:.8f}".format(epoch, config.train.num_epochs, 
                        step % iter_per_epoch, iter_per_epoch, loss_v))
                with open("./loss.txt", 'a+') as f:
                    #print(loss_v.cpu().numpy().type)
                    eval_dict = {'Epoch': epoch, 'loss': loss_v.cpu().numpy().tolist()}
                    eval_json = json.dumps(eval_dict)
                    f.write(eval_json)
                    f.write(eval_json)
                    f.write('\n')
                sys.stdout.flush()
            
            if step % iter_per_epoch == 0:
                dist.barrier()
                if rank == 0:
                    time_cost = time.time()-time_start
                    print(f'spent {time_cost} s')

                    epoch += 1
                    config.train.epoch = epoch
                    save_checkpoint(model, epoch, config.path.checkpoint,psnrprogress,wrecord,wde,win,alfas)
                    
                dist.barrier()
                if rank == 0:
                    evaluation(model, eval_loader, config)
                    if epoch == config.train.num_epochs:
                        raise Exception(f'epoch {epoch} >= max epoch {config.train.num_epochs}')
                    time_start = time.time()
                    print(f'Epoch={epoch}, lr={optimizer.param_groups[0]["lr"]}')


def test(rank, config):
    print(config.path.test)#vid4
    torch.cuda.set_device(rank)
    config.device = device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    world_size = len(config.gpus)
    setup(rank, world_size)

    model = config.network
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    start_epoch = load_checkpoint(model, config.path.resume, config.path.checkpoint, weights_init, rank)

    datapath = sorted(glob.glob(join(config.path.test, '*')))#calendar,city, walk,foliage
    datapath = [d for d in datapath if os.path.isdir(d)][rank :: world_size]#[c,c,w,f]的完整路径
    seqname = [os.path.split(d)[-1] for d in datapath]#[c,c,w,f]
    savepath = [join(config.path.test, d, config.test.save_name) for d in seqname]#/mnt/disk2/hsc/data/vid4/city/govblabla
    #print("datapath",datapath)
    for i, d in enumerate(tqdm(datapath)):
        print(d, savepath[i])
        test_video(model, d, savepath[i], config)
