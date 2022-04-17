from config import *
from torch.utils.data import Dataset,DataLoader

class STSBiEncTypeDataset(Dataset):
    def __init__(self, csv):
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

        encodings_1 = TOKENIZER(
            sent_a,
            padding='max_length',
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        )

        encodings_2 = TOKENIZER(
            sent_b,
            context,
            padding='max_length',
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        )
         
        return encodings_1, encodings_2, torch.tensor(row.score,dtype=torch.float),row.id


class STSCrossEncTypeDataset(Dataset):
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

        if self.context_type == 'a':
            text = TOKENIZER(
                context + sep_token + sent_a + sep_token + sent_b,
                padding='max_length', 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            )
        elif self.context_type == 'b':
            text = TOKENIZER(
                sent_a + sep_token + sent_b + sep_token + context,
                padding='max_length', 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            )
        else: 
            text = TOKENIZER(
                f'context : {context} {sep_token} sentence : {sent_a}',
                sent_b,
                padding='max_length', 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            )

        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]  
         
        return input_ids, attention_mask, torch.tensor(row.score,dtype=torch.float),row.id


class QATypeDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv.reset_index()

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        sent_a = row.anchor
        sent_b = row.target
        context = row.context

        question = f'Are the two sentences similar in the context of {context}'
        sep_token = TOKENIZER.sep_token

        text = TOKENIZER(
            question,
            sent_a + sep_token + sent_b,
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        )

        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]  
         
        return input_ids, attention_mask, torch.tensor(row.score,dtype=torch.float),row.id


def get_loader(data,fold,solution_type='STS',debug=False):
    train = data[data['kfold']!=fold]
    valid = data[data['kfold']==fold]
    
    print('Solution_type',solution_type)
    print('context_type',context_type)

    if solution_type == 'STSbienc':
        train_dataset = STSBiEncTypeDataset(csv=train,context_type=context_type)
        valid_dataset = STSBiEncTypeDataset(csv=valid,context_type=context_type)
    elif solution_type == 'STScrossenc':
        train_dataset = STSCrossEncTypeDataset(csv=train,context_type=context_type)
        valid_dataset = STSCrossEncTypeDataset(csv=valid,context_type=context_type)
    else:
        train_dataset = QATypeDataset(csv=train)
        valid_dataset = QATypeDataset(csv=valid)

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
    import pandas as pd

    data = pd.read_csv(f'data/train_folds.csv')
    titles = pd.read_csv('data/titles.csv')
    data = data.merge(titles[['context','context_text']],on='context')
    train_loader,valid_loader = get_loader(data,fold=0,solution_type='STScrossenc',debug=False)

    for i in train_loader:
        print(i[2].view(-1,1).shape)
        break
