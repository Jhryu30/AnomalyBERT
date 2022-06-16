import os, time
import numpy as np
import torch
import torch.nn as nn
import argparse

from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import utils.config as config
from models.anomaly_transformer import get_anomaly_transformer
from utils.datasets import get_data



def main(options):
    # SMAP dataset
    (train_smap, _), (_, _) = get_data('SMAP')
    d_data = len(train_smap[0])
    numerical_column = (0,)
    num_numerical = 1
    window_size = options.window_size

    # Define model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = get_anomaly_transformer(d_data=d_data,
                                    d_embed=512,
                                    hidden_dim_rate=4.,
                                    max_seq_len=window_size,
                                    mask_token_rate=(0,0),
                                    positional_encoding=None,
                                    relative_position_embedding=True,
                                    transformer_n_layer=options.n_layer,
                                    transformer_n_head=8,
                                    dropout=0.1).to(device)
    
    # Load a checkpoint if exists.
    if options.checkpoint != None:
        model.load_state_dict(torch.load(options.checkpoint, map_location='cpu'))

    log_dir = os.path.join(config.LOG_DIR, time.strftime('%y%m%d%H%M%S_', time.localtime(time.time())) + 'smap_test/')
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))
    
    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    # Train model.
    max_iters = options.max_steps
    n_batch = options.batch_size
    valid_index_list = np.arange(len(train_smap) - window_size)

    # Train loss
    lr = options.lr
    if options.loss == 'l1':
        train_loss = nn.L1Loss().to(device)
    elif options.loss == 'mse':
        train_loss = nn.MSELoss().to(device)
    rec_loss = nn.MSELoss().to(device)
    
    
    # Similarity map and constrastive loss
    def similarity_map(features):
        similarity = torch.matmul(features, features.transpose(-1, -2))
        norms = torch.norm(features, dim=-1)
        denom = torch.matmul(norms.unsqueeze(-1), norms.unsqueeze(-2)) + 1e-8
        return similarity / denom
    
    diag_mask = torch.eye(window_size, device=device).bool().unsqueeze(0)

    def contrastive_loss(features, anomaly_label):
        similarity = similarity_map(features)
        similarity_sum = torch.log(torch.exp(similarity).masked_fill(diag_mask, 0).sum(dim=-1))
        similarity.masked_fill_(diag_mask, 0)

        anomaly = anomaly_label.bool()
        normal = anomaly == False
        n_anomaly = anomaly_label.sum(dim=-1, keepdim=True).expand_as(anomaly_label)

        positive_term = similarity
        positive_term[anomaly] = 0
        positive_term = positive_term.transpose(-1, -2)[normal].mean(dim=-1) - similarity_sum[normal]
        positive_term /= (n_anomaly - window_size)[normal]

        negative_term = similarity_sum[anomaly]

        return positive_term.mean() + negative_term.mean()
    
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=max_iters,
                                  lr_min=lr*0.01,
                                  warmup_lr_init=lr*0.001,
                                  warmup_t=max_iters // 10,
                                  cycle_limit=1,
                                  t_in_epochs=False,
                                 )

    
    # Replaced data length table
    replacing_rate = (0, options.replacing_rate_max)
    replacing_table = list(np.random.randint(int(window_size*replacing_rate[0]), int(window_size*replacing_rate[1]), size=10000))
    replacing_table_index = 0
    replacing_table_length = 10000
    
    soft_replacing_prob = 1 - options.soft_replacing
    uniform_replacing_prob = soft_replacing_prob - options.uniform_replacing
    peak_noising_prob = options.peak_noising
    white_noising = options.white_noising

    
    # Start training.
    for i in range(options.initial_iter, max_iters):
        first_index = np.random.choice(valid_index_list, size=n_batch)
        x = []
        for j in first_index:
            x.append(torch.Tensor(train_smap[j:j+window_size].copy()))
        x_true = torch.stack(x).to(device)

        # Replace data.
        current_index = replacing_table_index
        replacing_table_index += n_batch

        replacing_lengths = []
        if replacing_table_index > replacing_table_length:
            replacing_lengths = replacing_table[current_index:]
            replacing_table_index -= replacing_table_length
            replacing_lengths = replacing_lengths + replacing_table[:replacing_table_index]
        else:
            replacing_lengths = replacing_table[current_index:replacing_table_index]
            if replacing_table_index == replacing_table_length:
                replacing_table_index = 0

        replacing_lengths = np.array(replacing_lengths)
        replacing_index = np.random.randint(0, len(train_smap)-replacing_lengths+1)
        target_index = np.random.randint(0, window_size-replacing_lengths+1)

        # Replacing type
        replacing_type = np.random.uniform(0., 1., size=(n_batch,))
        replacing_dim = np.random.uniform(0., 1., size=(n_batch, d_data)) < 0.3

        x_rep = []  # list of replaced intervals
        x_anomaly = torch.zeros(n_batch, window_size, device=device)  # list of anomaly points
        
        # Create anomaly intervals.
        for j, rep, tar, leng, typ, dim in zip(range(n_batch), replacing_index, target_index, replacing_lengths,
                                               replacing_type, replacing_dim):
            if leng > 0:
                x_rep.append(x[j][tar:tar+leng].clone())
                _x = x_rep[-1].clone().transpose(0, 1)
                rep_len = len(dim[dim])
                
                # External interval replacing
                if typ > soft_replacing_prob:
                    _x_temp = _x[numerical_column].clone()
                    _x[dim] = torch.Tensor(train_smap[rep:rep+leng, dim].copy()).transpose(0, 1)
                    _x[numerical_column] = (_x_temp + _x[numerical_column]) * 0.5
                    x_anomaly[j, tar:tar+leng] = 1
                    
                # Uniform replacing
                elif typ > uniform_replacing_prob:
                    _x[numerical_column] = torch.rand(num_numerical, 1)
                    x_anomaly[j, tar:tar+leng] = 1
                    
                # Peak noising
                elif typ < peak_noising_prob:
                    peak_index = np.random.randint(0, leng)
                    peak_value = (_x[dim, peak_index] < 0.5).float()
                    _x[dim, peak_index] = peak_value
                    x_anomaly[j, tar+peak_index] = 1
                
                else:
                    x_rep[-1] = None

                x[j][tar:tar+leng] = _x.transpose(0, 1)
            
            else:
                x_rep.append(None)
            
        # Process data.
        z = torch.stack(x).to(device)
        if white_noising:  # Add white noise.
            z[:, :, numerical_column] = (z[:, :, numerical_column]\
                                         + torch.normal(mean=0., std=0.001,
                                                        size=(n_batch, window_size, num_numerical)).to(device)).clamp(min=0., max=1.)
            
        y, features = model(z)

        # Compute losses.
        loss = options.total_loss * train_loss(x_true, y)
        partial_loss = 0
        
        for pred, gt, tar, leng in zip(y, x_rep, target_index, replacing_lengths):
            if leng > 0 and gt != None:
                partial_loss += train_loss(pred[tar:tar+leng], gt.to(device))
        if not torch.isnan(partial_loss):
            loss += options.partial_loss * partial_loss

        if options.contrastive_loss > 0:
            con_loss = contrastive_loss(features, x_anomaly)
            loss += options.contrastive_loss * con_loss

        # Print training summary.
        if i % options.summary_steps == 0:
            with torch.no_grad():
                origin = rec_loss(z, x_true).item()
                rec = rec_loss(x_true, y).item()
                
                summary_writer.add_scalar('Train Loss', loss.item(), i)
                summary_writer.add_scalar('Original Error', origin, i)
                summary_writer.add_scalar('Reconstruction', rec, i)
                summary_writer.add_scalar('Error rate', rec/origin, i)

                print('iter ', i, ',\tloss : {:.10f}'.format(loss.item()), ',\torigin err : {:.10f}'.format(origin), ',\trec : {:.10f}'.format(rec), sep='')
                print('\t\terr rate : {:.10f}'.format(rec/origin), sep='')
                print()
            torch.save(model.state_dict(), os.path.join(log_dir, 'state/state_dict_step_{}.pt'.format(i)))

        # Update gradients.
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()
        scheduler.step_update(i)

    torch.save(model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--summary_steps", default=500, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--initial_iter", default=0, type=int)
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--window_size", default=512, type=int)
    parser.add_argument("--n_layer", default=12, type=int)
    parser.add_argument("--replacing_rate_max", default=0.25, type=float)
    
    parser.add_argument("--soft_replacing", default=0.5, type=float)
    parser.add_argument("--uniform_replacing", default=0.25, type=float)
    parser.add_argument("--peak_noising", default=0., type=float)
    parser.add_argument("--white_noising", default=False, action='store_true')
    
    parser.add_argument("--loss", default='l1', type=str)
    parser.add_argument("--total_loss", default=0.2, type=float)
    parser.add_argument("--partial_loss", default=1., type=float)
    parser.add_argument("--contrastive_loss", default=0., type=float)
    
    options = parser.parse_args()
    main(options)