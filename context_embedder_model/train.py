import pandas as pd

from train_utils import *
from config import *
import torch.nn as nn
from engine import *
from dataset import get_loader
from scipy import stats 

from model import USSModel
from transformers import AdamW


def run(data,fold):
    LOGGER.info('\n')
    LOGGER.info(f"====================== FOLD : {fold} training ========================")

    seed_everything(SEED)
    
    train_loader,valid_loader = get_loader(data,fold)
    
    seed_everything(SEED)
    
    # Defining Model for specific fold
    model = USSModel(
        model_name = transformer_model,
        pooling = pooling
    )
    model.to(device)
    
    #DEfining criterion
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    #criterion = CorrLoss()
    criterion.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
    
    optimizer = AdamW(optimizer_parameters, lr=LR,weight_decay=WEIGHT_DECAY)
    
    #Defining LR SCheduler
    total_steps = len(train_loader)*EPOCHS
    scheduler = fetch_scheduler(optimizer,int(warmup_ratio*total_steps),total_steps)
    
    # THE ENGINE LOOP
    best_pearson = 0 
    oof_tar = 0
    oof_pred = 0
    oof_id = 0

    for epoch in range(EPOCHS):
        train_loss = train_fn(train_loader, model,criterion, optimizer, device,scheduler=scheduler,epoch=epoch)
        valid_loss,valid_out,valid_tar,valid_id = eval_fn(valid_loader, model, criterion,device,epoch=epoch)
        
        pearson = np.corrcoef(valid_out, valid_tar)[0][1]

        LOGGER.info(f'-------  Metrics for Epoch {epoch} --------')
        LOGGER.info(f'Train Loss : {train_loss.avg}')
        LOGGER.info(f'Valid Loss : {valid_loss}')
        LOGGER.info(f'Valid Pearson Score : {pearson}')
        
        if pearson >= best_pearson:
            best_pearson = pearson
            torch.save(model.state_dict(),f"{SAVE_DIR}/model_{fold}.bin")
            LOGGER.info('Best model found for epoch {} Saving Model'.format(epoch))

            oof_tar = valid_tar
            oof_pred = valid_out
            oof_id = valid_id

    return oof_tar,oof_pred,oof_id


if __name__ == "__main__":
    import shutil
    import config
    import os
    os.chdir(config.main_dir)
    
    # mapping = {
    # "A": "Human Necessities",
    # "B": "Operations and Transport",
    # "C": "Chemistry and Metallurgy",
    # "D": "Textiles",
    # "E": "Fixed Constructions",
    # "F": "Mechanical Engineering",
    # "G": "Physics",
    # "H": "Electricity",
    # "Y": "Emerging Cross-Sectional Technologies"}

    make_dir(SAVE_DIR)
    LOGGER = get_logger()
    LOGGER.info('Making Directory')
    
    LOGGER.info('Copying Config')
    shutil.copy('config.py',SAVE_DIR)

    LOGGER.info('Copying Model')
    shutil.copy('model.py',SAVE_DIR)

    data = pd.read_csv(f'data/train_folds.csv')
    # data['context'] = data['context'].apply(lambda x:x[0])
    # data['context_text'] = data['context'].map(mapping)
    #data = data.rename(columns={'context':'context_text'})
    cpc_texts = torch.load("data/cpc_texts.pth")
    
    data['context_text'] = data['context'].map(cpc_texts)
    print(data.head())

    fin_tar = []
    fin_pred = []
    fin_id = []

    for i in range(NUM_FOLDS):
        print(i)
        oof_tar,oof_pred,oof_id = run(data,i)
        fin_tar.append(oof_tar)
        fin_pred.append(oof_pred)
        fin_id.append(oof_id)
        
    if i > 1:
        fin_tar = np.concatenate(fin_tar)
        fin_pred = np.concatenate(fin_pred)
        fin_id = np.concatenate(fin_id)

        LOGGER.info(f"OOF Pearson Score : {np.corrcoef(fin_pred, fin_tar)[0][1]}")

        df_oof = pd.DataFrame(dict(
        id = fin_id, target=fin_tar, pred = fin_pred))
        df_oof.to_csv(f'{SAVE_DIR}/oof.csv',index=False)
        df_oof.head()