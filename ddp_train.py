"""
single node の multi GPUで分散学習
single GPU との違い
- `ddp_setup`関数を使う
- `DataLoader`の引数に`sampler=DistributedSampler(dataset)`を追加
- `Trainer`クラスのコンストラクタに`self.model = DDP(model, device_ids=[gpu_id])`をついか
- `main`関数を直接呼び出すのではなく，`mp.spawn(main, args=(world_size,), nprocs=world_size)`とする．
"""
import os
import yaml

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import dataset as ds
import utils
from model import TransformerWithPE
from scalers import StandardScaler



class Trainer:
    def __init__(self,
                model,
                gpu_id,
                optimizer,
                train_loader,
                valid_loader,
                loss_fn,
                config):
        self.model = DDP(model, device_ids=[gpu_id])
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.patience = config['Train']['patience']
        self.save_dir = config['Train']['save_dir']
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, epochs):
        best_score = float('inf')
        early_stop_cnt = 0
        print('########start training########', flush=True)
        for epoch in range(1, epochs+1):
            train_rmse = self._train() #debug
            valid_rmse = self._eval()
            print(f'{epoch} epoch\nTrain RMSE: {train_rmse:.4f} | Valid RMSE: {valid_rmse:.4f} | ', flush=True)

            # early stopping
            if valid_rmse < best_score:
                best_score = valid_rmse
                self._save_model('checkpoint.pth')
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
                print(f'EarlyStopping counter: {early_stop_cnt} out of {self.patience}', flush=True)
                if early_stop_cnt >= self.patience:
                    break
            print('-'*50, flush=True)
        print('finish training!', flush=True)
        self._save_model('model.pth')

    def _train(self):
        self.model.train()
        train_rmse = []
        for _, (src, tgt, tgt_y) in enumerate(self.train_loader):
            src, tgt, tgt_y = src.unsqueeze(2).to(self.device), tgt.unsqueeze(2).to(self.device), tgt_y.unsqueeze(2).to(self.device)
            scaler = StandardScaler()
            scaler.fit(src)
            src = scaler.transform(src)
            tgt = scaler.transform(tgt)
            tgt_y = scaler.transform(tgt_y)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Make forecasts
            prediction = self.model(
                        src=src, 
                        forecast_window=tgt_y.shape[1],
                        src_key_padding_mask=None)

            # Compute and backprop loss
            loss = self.loss_fn(tgt_y, prediction)
            loss.backward()
            self.optimizer.step()

            train_rmse.append(loss.item()**0.5)

        return sum(train_rmse) /len(train_rmse)

    @ torch.no_grad()
    def _eval(self):
        self.model.eval()
        valid_rmse = []
        for _, (src, _, tgt_y) in enumerate(self.valid_loader):
            src, tgt_y = src.unsqueeze(2).to(self.device), tgt_y.unsqueeze(2).to(self.device)
            
            scaler = StandardScaler()
            scaler.fit(src)
            src = scaler.transform(src)
            tgt_y = scaler.transform(tgt_y)
            
            pred = self.model(
                src=src, 
                forecast_window=tgt_y.shape[1], 
                src_key_padding_mask=None
                )
            loss = self.loss_fn(tgt_y, pred)
            valid_rmse.append(loss.item()**0.5)
        return sum(valid_rmse) / len(valid_rmse)

    def _save_model(self, fname:str):
        # print(f'save the model at {self.save_dir}/{fname}', flush=True)
        torch.save(self.model.module.state_dict(), f'{self.save_dir}/{fname}')
        # torch.save(self.model.state_dict(), f'{self.save_dir}/{fname}')

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def prepare_dataloader(config):
    df = utils.read_data(timestamp_col_name=config['Data']['timestamp_col'], 
                         data_dir='/gs/hs0/tga-nakatalab/home/nagai/infuluenza_transformer/data')

    train_df = df[:-(round(len(df)*config['Data']['test_size']))]
    valid_df = df[-(round(len(df)*config['Data']['test_size'])):]


    window_size = config['Data']['enc_seq_len'] + config['Data']['out_seq_len']
    train_indices = utils.get_indices_entire_sequence(
        data=train_df,
        window_size=window_size,
        step_size=config['Data']['step_size'])

    valid_indices = utils.get_indices_entire_sequence(
        data=valid_df,
        window_size=window_size,
        step_size=config['Data']['step_size'])

    # Making instance of custom dataset class
    train_data = ds.TransformerDataset(
        data=torch.tensor(train_df[config['Data']['target_col']].values).float(),
        indices=train_indices,
        enc_seq_len=config['Data']['enc_seq_len'],
        dec_seq_len=config['Data']['out_seq_len'],
        target_seq_len=config['Data']['out_seq_len']
        )

    # Making instance of custom dataset class
    valid_data = ds.TransformerDataset(
        data=torch.tensor(valid_df[config['Data']['target_col']].values).float(),
        indices=valid_indices,
        enc_seq_len=config['Data']['enc_seq_len'],
        dec_seq_len=config['Data']['out_seq_len'],
        target_seq_len=config['Data']['out_seq_len']
        )

    # Making dataloader
    train_loader = DataLoader(train_data, config['Train']['batch_size'],
                              pin_memory=True,
                              shuffle=False,
                              sampler=DistributedSampler(train_data))
    
    valid_loader = DataLoader(valid_data, config['Train']['test_batch_size'],
                              pin_memory=True,
                              shuffle=False,
                              sampler=DistributedSampler(valid_data))
    return train_loader, valid_loader

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def main(rank, world_size):
    print('process start', flush=True)
    ddp_setup(rank, world_size)
    config = load_config('config.yaml')
    print('loading the dataset', flush=True)
    train_loader, valid_loader = prepare_dataloader(config)

    print('setup', flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerWithPE(
    in_dim=config['Model']['in_dim'], 
    out_dim=config['Model']['out_dim'], 
    embed_dim=config['Model']['embed_dim'],
    num_heads=config['Model']['num_heads'], 
    num_layers=config['Model']['num_layers']
    ).to(device)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['Train']['lr']))
    loss_fn = torch.nn.MSELoss()

    trainer = Trainer(
        model=model,
        gpu_id=rank,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        config=config
    )

    trainer.train(config['Train']['epochs'])


if __name__=='__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
