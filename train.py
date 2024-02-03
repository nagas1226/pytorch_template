class Trainer:
    def __init__(self,
                model,
                optimizer,
                train_loader,
                valid_loader,
                loss_fn,
                config):
        self.model = model
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
            print(f'{epoch} epoch', flush=True)
            train_rmse = self._train()
            print(f'Train RMSE: {train_rmse:.4f}', end=' | ' , flush=True)
            valid_rmse = self._eval()
            print(f'Valid RMSE: {valid_rmse:.4f}', flush=True)

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
        for i, (src, tgt, tgt_y) in enumerate(self.train_loader):
            src, tgt, tgt_y = src.unsqueeze(2).to(self.device), tgt.unsqueeze(2).to(self.device), tgt_y.unsqueeze(2).to(self.device)
            # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
            if self.config['Model']['batch_first'] == False:
                src = src.permute(1, 0, 2)
                tgt = tgt.permute(1, 0, 2)
                tgt_y = tgt_y.permute(1, 0, 2)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Generate masks
            tgt_mask = utils.generate_square_subsequent_mask(
                dim1=self.config['Data']['out_seq_len'],
                dim2=self.config['Data']['out_seq_len'],
                device=next(self.model.parameters()).device
                )
            src_mask = utils.generate_square_subsequent_mask(
                dim1=self.config['Data']['out_seq_len'],
                dim2=self.config['Data']['enc_seq_len'],
                device=next(self.model.parameters()).device
                )


            # tgt_mask = torch.cat([tgt_mask.unsqueeze(0)], dim=0)
            # src_mask = torch.cat([src_mask.unsqueeze(0)], dim=0)
            # Make forecasts
            prediction = self.model(src, tgt, src_mask, tgt_mask)

            # Compute and backprop loss
            loss = self.loss_fn(tgt_y, prediction)
            loss.backward()
            self.optimizer.step()

            train_rmse.append(loss.item()**0.5)

        return sum(train_rmse)/len(train_rmse)

    @ torch.no_grad()
    def _eval(self):
        self.model.eval()
        valid_rmse = []
        for i, (src, _, tgt_y) in enumerate(self.valid_loader):
            src, tgt_y = src.unsqueeze(2).to(self.device), tgt_y.unsqueeze(2).to(self.device)
            # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
            if self.config['Model']['batch_first'] == False:
                src = src.permute(1, 0, 2)
                tgt_y = tgt_y.permute(1, 0, 2)

            prediction = inference.run_encoder_decoder_inference(
                model=self.model,
                src=src,
                forecast_window=self.config['Data']['out_seq_len'],
                device=self.device,
                batch_size=src.shape[1]
                )
            print(f'prediction.shape: {prediction.shape}') #debug

            loss = self.loss_fn(tgt_y, prediction)
            valid_rmse.append(loss.item()**0.5)
        return sum(valid_rmse) / len(valid_rmse)

    def _save_model(self, fname:str):
        print(f'save the model at {self.save_dir}/{fname}', flush=True)
        # torch.save(self.model.module.state_dict(), f'{self.save_dir}/{fname}')
        torch.save(self.model.state_dict(), f'{self.save_dir}/{fname}')

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def prepare_dataloader(config):
    df = utils.read_data(timestamp_col_name=config['Data']['timestamp_col'], data_dir='./data')

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
    train_loader = DataLoader(train_data, config['Train']['batch_size'])
    valid_loader = DataLoader(valid_data, config['Train']['batch_size'])
    return train_loader, valid_loader

def main():
    config = load_config('config.yaml')
    train_loader, valid_loader = prepare_dataloader(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tst.TimeSeriesTransformer(
        input_size=1,
        dec_seq_len=config['Data']['enc_seq_len'],
        batch_first=config['Model']['batch_first'],
        num_predicted_features=1
    ).to(device)

    # # make parallel if cuda is available
    # if device == 'cuda':
    #     model = torch.nn.DataParallel(model)
    #     torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['Train']['lr']))
    loss_fn = torch.nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        config=config
    )

    trainer.train(config['Train']['epochs'])



if __name__=='__main__':
    main()
