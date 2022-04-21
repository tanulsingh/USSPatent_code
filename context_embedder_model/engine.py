import torch
import numpy as np
from tqdm import tqdm
import config
from train_utils import AverageMeter,PearsonMeter
import torch.nn.functional as F


def train_fn(dataloader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    loss_score = AverageMeter()
    pearson = PearsonMeter()
    
    if config.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi,d in tk0:
        
        batch_size = d[2].shape[0]

        input_encoding = d[0]
        context_encoding = d[1]
        targets = d[2]

        input_encoding = {k:v.to(device) for k, v in input_encoding.items()}
        context_encoding = {k:v.to(device) for k, v in context_encoding.items()}
        targets = targets.to(device)

        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(input_encoding,context_encoding)
                loss = criterion(output,targets.view(-1,1))
                #loss = F.mse_loss(output.squeeze(-1), targets, reduction="mean")
                #loss = criterion(output.squeeze(-1),targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(context_encoding,context_encoding)
            loss = criterion(output,targets.view(-1,1))
            #loss = F.mse_loss(output.squeeze(-1), targets, reduction="mean")
            #loss = criterion(output.squeeze(-1),targets)
        
            loss.backward()
            optimizer.step()
        
        loss_score.update(loss.detach().item(), batch_size)
        
        #pearson.update(targets,output.squeeze(-1))
        pearson.update(targets,output.sigmoid().squeeze(-1))
        tk0.set_postfix(Train_Loss=loss_score.avg,Train_Pearson=pearson.avg,Epoch=epoch,LR=optimizer.param_groups[0]['lr'])
        
        if scheduler is not None:
                scheduler.step()
        
        optimizer.zero_grad()
        
    return loss_score
    

def eval_fn(dataloader,model,criterion,device,epoch):
    model.eval()
    loss_score = AverageMeter()
    pearson = PearsonMeter()

    fin_out = []
    fin_tar = []
    fin_id = []
    
    with torch.no_grad():
        
        tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
        for bi,d in tk0:

            batch_size = d[2].shape[0]

            input_encoding = d[0]
            context_encoding = d[1]
            targets = d[2]

            input_encoding = {k:v.to(device) for k, v in input_encoding.items()}
            context_encoding = {k:v.to(device) for k, v in context_encoding.items()}
            targets = targets.to(device)
            id = d[3]

            output = model(input_encoding,context_encoding)
            loss = criterion(output,targets.view(-1,1))
            #loss = F.mse_loss(output.squeeze(-1), targets, reduction="mean")
            #loss = criterion(output.squeeze(-1),targets)

            loss_score.update(loss.detach().item(), batch_size)
            pearson.update(targets,output.sigmoid().squeeze(-1))
            #pearson.update(targets,output.squeeze(-1))
            
            tk0.set_postfix(Eval_Loss=loss_score.avg,Eval_Pearson_score=pearson.avg,Epoch=epoch)
            
            #output = output.squeeze(-1).detach().cpu().numpy()
            output = output.sigmoid().squeeze(-1).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            fin_out.append(output)
            fin_tar.append(targets)
            fin_id.append(id)
        
    return loss_score.avg,np.concatenate(fin_out),np.concatenate(fin_tar),np.concatenate(fin_id)