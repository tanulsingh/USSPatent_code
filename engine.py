from black import out
import torch
import numpy as np
from tqdm import tqdm
import config
from train_utils import AverageMeter,PearsonMeter


def train_fn(dataloader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    loss_score = AverageMeter()
    pearson = PearsonMeter()
    
    if config.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi,d in tk0:
        
        batch_size = d[0].shape[0]

        input_ids = d[0]
        attention_mask = d[1]
        targets = d[2]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(input_ids,attention_mask)
                loss = criterion(output,targets.view(-1,1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input_ids,attention_mask)
            loss = criterion(output,targets.view(-1,1))
        
            loss.backward()
            optimizer.step()
        
        loss_score.update(loss.detach().item(), batch_size)
        
        pearson.update(targets,output.squeeze(-1))
        tk0.set_postfix(Train_Loss=loss_score.avg,Train_Pearson=pearson.avg,Epoch=epoch,LR=optimizer.param_groups[0]['lr'])

        if scheduler is not None:
                scheduler.step()
        
    return loss_score
    

def eval_fn(dataloader,model,criterion,device,epoch):
    model.eval()
    loss_score = AverageMeter()

    fin_out = []
    fin_tar = []
    fin_id = []
    
    with torch.no_grad():
        
        tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
        for bi,d in tk0:

            batch_size = d[0].shape[0]

            input_ids = d[0]
            attention_mask = d[1]
            targets = d[2]
            id = d[3]

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            output = model(input_ids,attention_mask)
            loss = criterion(output,targets.view(-1,1))

            loss_score.update(loss.detach().item(), batch_size)
            
            tk0.set_postfix(Eval_Loss=loss_score.avg,Epoch=epoch)
            
            output = output.squeeze(-1).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            fin_out.append(output)
            fin_tar.append(targets)
            fin_id.append(id)
        
    return loss_score.avg,np.concatenate(fin_out),np.concatenate(fin_tar),np.concatenate(fin_id)