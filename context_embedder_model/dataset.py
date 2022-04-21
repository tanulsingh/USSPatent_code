from config import *
from torch.utils.data import Dataset,DataLoader

class STSDataset(Dataset):
    def __init__(self, csv, context_type='a'):
        self.csv = csv.reset_index()
        self.context_type = context_type

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        sent_a = row.anchor
        sent_b = row.target
        context = row.context_text
        
        sep_token = TOKENIZER.sep_token

    
        input_encoding = TOKENIZER(
                sent_a,
                sent_b,
                padding='max_length', 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            )
        input_encoding = {k:v[0] for k, v in input_encoding.items()}
        
        context_encoding = TOKENIZER(
                context,
                padding='max_length', 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            )
        context_encoding = {k:v[0] for k, v in context_encoding.items()}
         
        return input_encoding, context_encoding, torch.tensor(row.score,dtype=torch.float),row.id


def get_loader(data,fold,debug=False):
    train = data[data['kfold']!=fold]
    valid = data[data['kfold']==fold]
    
    train_dataset = STSDataset(csv=train)
    valid_dataset = STSDataset(csv=valid)

    if debug:
        for i in train_dataset:
            print(i[0].shape)
            print(i[1].shape)
            print(i[2].shape)
            print(i[2])
            break
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        num_workers=NUM_WORKERS
    )

    return train_loader,valid_loader


if __name__ == '__main__':
    import os
    os.chdir('/home/tanulsingh/USSPatent_code')
    import pandas as pd

    data = pd.read_csv(f'data/train_folds.csv')
    cpc_texts = torch.load("data/cpc_texts.pth")
    
    data['context_text'] = data['context'].map(cpc_texts)
    train_loader,valid_loader = get_loader(data,fold=0,debug=False)

    for i in train_loader:
        print(i[0]['input_ids'].shape)
        print(i[2].view(-1,1).shape)
        break
