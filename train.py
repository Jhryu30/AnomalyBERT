import os, time, json
import numpy as np
import torch
import torch.nn as nn
import argparse

from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import utils.config as config
from models.anomaly_transformer import get_anomaly_transformer

from estimate import estimate
from compute_metrics import f1_score



def main(options):
    # Load data.
    train_data = np.load(config.TRAIN_DATASET[options.dataset]).copy().astype(np.float32)
    replacing_data = train_data if options.replacing_data == None\
                     else np.load(config.TRAIN_DATASET[options.replacing_data]).copy().astype(np.float32)
    test_data = np.load(config.TEST_DATASET[options.dataset]).copy().astype(np.float32)
    test_label = np.load(config.TEST_LABEL[options.dataset]).copy().astype(np.int32)
    
    d_data = len(train_data[0])
    numerical_column = np.array(config.NUMERICAL_COLUMNS[options.dataset])
    num_numerical = len(numerical_column)
    categorical_column = np.array(config.CATEGORICAL_COLUMNS[options.dataset])
    num_categorical = len(categorical_column)
    
    # Ignore the specific columns.
    if options.dataset in config.IGNORED_COLUMNS.keys():
        ignored_column = np.array(config.IGNORED_COLUMNS[options.dataset])
        remaining_column = [col for col in range(d_data) if col not in ignored_column]
        train_data = train_data[:, remaining_column]
        replacing_data = train_data if options.replacing_data == None else replacing_data[:, remaining_column]
        test_data = test_data[:, remaining_column]
        
        d_data = len(remaining_column)
        numerical_column -= (numerical_column[:, None] - ignored_column[None, :] > 0).astype(int).sum(axis=1)
        categorical_column -= (categorical_column[:, None] - ignored_column[None, :] > 0).astype(int).sum(axis=1)
        
    # Data division
    if options.data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][options.data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
    
    window_size = options.window_size
    data_seq_len = window_size * options.patch_size

    # Define model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = get_anomaly_transformer(input_d_data=d_data,
                                    output_d_data=1 if options.loss=='bce' else d_data,
                                    patch_size=options.patch_size,
                                    d_embed=options.d_embed,
                                    hidden_dim_rate=4.,
                                    max_seq_len=window_size,
                                    positional_encoding=None,
                                    relative_position_embedding=True,
                                    transformer_n_layer=options.n_layer,
                                    transformer_n_head=8,
                                    dropout=options.dropout).to(device)
    
    # Load a checkpoint if exists.
    if options.checkpoint != None:
        model.load_state_dict(torch.load(options.checkpoint, map_location='cpu'))

    log_dir = os.path.join(config.LOG_DIR, time.strftime('%y%m%d%H%M%S_'+options.dataset, time.localtime(time.time())))
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))
    
    # hyperparameters save
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)
    
    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    # Train model.
    max_iters = options.max_steps + 1
    n_batch = options.batch_size
    valid_index_list = np.arange(len(train_data) - data_seq_len)
#     anomaly_weight = options.partial_loss / options.total_loss

    # Train loss
    lr = options.lr
    if options.loss == 'l1':
        train_loss = nn.L1Loss().to(device)
        rec_loss = nn.MSELoss().to(device)
    elif options.loss == 'mse':
        train_loss = nn.MSELoss().to(device)
        rec_loss = nn.MSELoss().to(device)
    elif options.loss == 'bce':
        train_loss = nn.BCELoss().to(device)
#         train_loss = lambda pred, gt: -(gt * torch.log(pred + 1e-8) + (1 - gt.bool().float()) * torch.log(1 - pred + 1e-8)).mean()
        sigmoid = nn.Sigmoid().to(device)
    
    
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
    
    def replacing_weights(interval_len):
        warmup_len = interval_len // 10
        return np.concatenate((np.linspace(0, options.replacing_weight, num=warmup_len),
                               np.full(interval_len-2*warmup_len, options.replacing_weight),
                               np.linspace(options.replacing_weight, 0, num=warmup_len)), axis=None)
    
    
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
    replacing_rate = (options.replacing_rate_max/10, options.replacing_rate_max)
    replacing_len_max = int(options.replacing_rate_max * data_seq_len)
    replacing_len_half_max = replacing_len_max // 2
    
    replacing_table = list(np.random.randint(int(data_seq_len*replacing_rate[0]), int(data_seq_len*replacing_rate[1]), size=10000))
    replacing_table_index = 0
    replacing_table_length = 10000
    
    # Synthesis probability
    soft_replacing_prob = 1 - options.soft_replacing
    uniform_replacing_prob = soft_replacing_prob - options.uniform_replacing
    peak_noising_prob = uniform_replacing_prob - options.peak_noising
    length_adjusting_prob = peak_noising_prob - options.length_adjusting if options.loss == 'bce' else peak_noising_prob
    white_noising_prob = options.white_noising
    
    # Soft replacing flip options
    if options.flip_replacing_interval == 'all':
        vertical_flip = True
        horizontal_flip = True
    elif options.flip_replacing_interval == 'vertical':
        vertical_flip = True
        horizontal_flip = False
    elif options.flip_replacing_interval == 'horizontal':
        vertical_flip = False
        horizontal_flip = True
    elif options.flip_replacing_interval == 'none':
        vertical_flip = False
        horizontal_flip = False

    
    # Start training.
    for i in range(options.initial_iter, max_iters):
        first_index = np.random.choice(valid_index_list, size=n_batch)
        x = []
        for j in first_index:
            x.append(torch.Tensor(train_data[j:j+data_seq_len].copy()).to(device))
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
        replacing_index = np.random.randint(0, (len(replacing_data)-replacing_lengths+1)[:, np.newaxis],
                                            size=(n_batch, d_data))
        target_index = np.random.randint(0, data_seq_len-replacing_lengths+1)

        # Replacing types and dimensions
        replacing_type = np.random.uniform(0., 1., size=(n_batch,))
        replacing_dim_numerical = np.random.uniform(0., 1., size=(n_batch, num_numerical))
        replacing_dim_categorical = np.random.uniform(0., 1., size=(n_batch, num_categorical))
        
        replacing_dim_numerical = replacing_dim_numerical\
                                  - np.maximum(replacing_dim_numerical.min(axis=1, keepdims=True), 0.3) <= 0.001
        replacing_dim_categorical = replacing_dim_categorical\
                                    - np.maximum(replacing_dim_categorical.min(axis=1, keepdims=True), 0.3) <= 0.001
        
#         replacing_dim = np.empty(n_batch, d_data, dtype=bool)
#         replacing_dim[numerical_column] = replacing_dim_numerical
#         replacing_dim[categorical_column] = replacing_dim_categorical

        x_rep = []  # list of replaced intervals
        x_anomaly = torch.zeros(n_batch, data_seq_len, device=device)  # list of anomaly points
        
        # Create anomaly intervals.
        for j, rep, tar, leng, typ, dim_num, dim_cat in zip(range(n_batch), replacing_index, target_index, replacing_lengths,
                                                            replacing_type, replacing_dim_numerical, replacing_dim_categorical):
            if leng > 0:
                x_rep.append(x[j][tar:tar+leng].clone())
                _x = x_rep[-1].clone().transpose(0, 1)
                rep_len_num = len(dim_num[dim_num])
                rep_len_cat = len(dim_cat[dim_cat]) if len(dim_cat) > 0 else 0
                target_column_numerical = numerical_column[dim_num]
                if rep_len_cat > 0:
                    target_column_categorical = categorical_column[dim_cat]
                
                # External interval replacing
                if typ > soft_replacing_prob:
                    # Replacing for numerical columns
                    _x_temp = []
                    col_num = np.random.choice(numerical_column, size=rep_len_num)
                    filp = np.random.randint(0, 2, size=(rep_len_num,2)) > 0.5
                    for _col, _rep, _flip in zip(col_num, rep[:rep_len_num], filp):
                        random_interval = replacing_data[_rep:_rep+leng, _col].copy()
                        # fliping options
                        if horizontal_flip and _flip[0]:
                            random_interval = random_interval[::-1].copy()
                        if vertical_flip and _flip[1]:
                            random_interval = 1 - random_interval
                        _x_temp.append(torch.from_numpy(random_interval))
                    _x_temp = torch.stack(_x_temp).to(device)
                    weights = torch.from_numpy(replacing_weights(leng)).float().unsqueeze(0).to(device)
                    _x[target_column_numerical] = _x_temp * weights + _x[target_column_numerical] * (1 - weights)

                    # Replacing for categorical columns
                    if rep_len_cat > 0:
                        _x_temp = []
                        col_cat = np.random.choice(categorical_column, size=rep_len_cat)
                        for _col, _rep in zip(col_cat, rep[-rep_len_cat:]):
                            _x_temp.append(torch.from_numpy(replacing_data[_rep:_rep+leng, _col].copy()))
                        _x_temp = torch.stack(_x_temp).to(device)
                        _x[target_column_categorical] = _x_temp

                        x_anomaly[j, tar:tar+leng] = 1
                        x[j][tar:tar+leng] = _x.transpose(0, 1)

                # Uniform replacing
                elif typ > uniform_replacing_prob:
                    _x[target_column_numerical] = torch.rand(rep_len_num, 1, device=device)
#                     _x[target_column_categorical] = torch.randint(0, 2, size=(rep_len_cat, 1), device=device).float()
                    x_anomaly[j, tar:tar+leng] = 1
                    x[j][tar:tar+leng] = _x.transpose(0, 1)

                # Peak noising
                elif typ > peak_noising_prob:
                    peak_index = np.random.randint(0, leng)
                    peak_value = (_x[target_column_numerical, peak_index] < 0.5).float().to(device)
                    peak_value = peak_value + (0.1 * (1 - 2 * peak_value)) * torch.rand(rep_len_num, device=device)
                    _x[target_column_numerical, peak_index] = peak_value

#                     peak_value = (_x[target_column_categorical, peak_index] < 0.5).float().to(device)
#                     _x[target_column_categorical, peak_index] = peak_value
                    
                    peak_index = tar + peak_index
                    tar_first = np.maximum(0, peak_index - options.patch_size)
                    tar_last = peak_index + options.patch_size + 1
                    
                    x_anomaly[j, tar_first:tar_last] = 1
                    x[j][tar:tar+leng] = _x.transpose(0, 1)
                    
                # Length adjusting (only for bce loss)
                elif typ > length_adjusting_prob:
                    # Lengthening
                    if leng > replacing_len_half_max:
                        scale = np.random.randint(2, 5)
                        _leng = leng - leng % scale
                        scaled_leng = _leng // scale
                        x[j][tar+_leng:] = x[j][tar+scaled_leng:-_leng+scaled_leng].clone()
                        x[j][tar:tar+_leng] = torch.repeat_interleave(x[j][tar:tar+scaled_leng], scale, axis=0)
                        x_anomaly[j, tar:tar+_leng] = 1
                    # Shortening
                    else:
                        origin_index = first_index[j]
                        if origin_index > replacing_len_max * 1.5:
                            scale = np.random.randint(2, 5)
                            _leng = leng * (scale - 1)
                            x[j][:tar] = torch.Tensor(train_data[origin_index-_leng:origin_index+tar-_leng].copy()).to(device)
                            x[j][tar:tar+leng] = torch.Tensor(train_data[origin_index+tar-_leng:origin_index+tar+leng:scale].copy()).to(device)
                            x_anomaly[j, tar:tar+leng] = 1
                    
                # White noising (deprecated)
                elif typ < white_noising_prob:
                    _x[target_column_numerical] = (_x[target_column_numerical]\
                                                   + torch.normal(mean=0., std=0.003, size=(rep_len_num, leng), device=device))\
                                                  .clamp(min=0., max=1.)
                    x_anomaly[j, tar:tar+leng] = 1
                    x[j][tar:tar+leng] = _x.transpose(0, 1)
                
                else:
                    x_rep[-1] = None
            
            else:
                x_rep.append(None)
            
        # Process data.
        z = torch.stack(x)
        y = model(z)

        # Compute losses.
        if options.loss == 'bce':
            y = y.squeeze(-1)
            loss = train_loss(sigmoid(y), x_anomaly)
#             partial_loss = 0

#             for pred, gt, ano_label, tar, leng in zip(y, x_rep, x_anomaly, target_index, replacing_lengths):
#                 if leng > 0 and gt != None:
#                     partial_loss += train_loss(sigmoid(pred[tar:tar+leng]), ano_label[tar:tar+leng])
#             loss += options.partial_loss * partial_loss
        
        else:
            loss = options.total_loss * train_loss(x_true, y)
            partial_loss = 0

            for pred, gt, tar, leng in zip(y, x_rep, target_index, replacing_lengths):
                if leng > 0 and gt != None:
                    partial_loss += train_loss(pred[tar:tar+leng], gt.to(device))
            if not torch.isnan(partial_loss):
                loss += options.partial_loss * partial_loss

#         if options.contrastive_loss > 0:
#             con_loss = contrastive_loss(features, x_anomaly)
#             loss += options.contrastive_loss * con_loss

        # Print training summary.
        if i % options.summary_steps == 0:
            with torch.no_grad():
                if options.loss == 'bce':
                    pred = (sigmoid(y) > 0.5).int()
                    x_anomaly = x_anomaly.bool().int()
                    total_data_num = n_batch * data_seq_len
                    
                    acc = (pred == x_anomaly).int().sum() / total_data_num
                    summary_writer.add_scalar('Train/Loss', loss.item(), i)
                    summary_writer.add_scalar('Train/Accuracy', acc, i)
                    
                    model.eval()
                    estimation = estimate(test_data, model,
                                          sigmoid if options.loss == 'bce' else nn.Identity().to(device),
                                          1 if options.loss == 'bce' else d_data,
                                          n_batch, options.window_sliding, divisions, None, device)
                    estimation = estimation[:, 0].cpu().numpy()
                    model.train()
                    
                    best_eval = (0, 0, 0)
                    best_rate = 0
                    for rate in np.arange(0.001, 0.301, 0.001):
                        evaluation = f1_score(test_label, estimation, rate, False, False)
                        if evaluation[2] > best_eval[2]:
                            best_eval = evaluation
                            best_rate = rate
                    summary_writer.add_scalar('Valid/Best Anomaly Rate', best_rate, i)
                    summary_writer.add_scalar('Valid/Precision', best_eval[0], i)
                    summary_writer.add_scalar('Valid/Recall', best_eval[1], i)
                    summary_writer.add_scalar('Valid/F1', best_eval[2], i)
                    
                    print(f'iteration: {i} | loss: {loss.item():.10f} | train accuracy: {acc:.10f}')
                    print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
                    
                else:
                    origin = rec_loss(z[:, :, numerical_column], x_true).item()
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

        nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)

        optimizer.step()
        scheduler.step_update(i)

    torch.save(model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_steps", default=150000, type=int)
    parser.add_argument("--summary_steps", default=500, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--initial_iter", default=0, type=int)
    
    parser.add_argument("--dataset", default='SMAP', type=str, help='SMAP/MSL/SMD/SWaT/WADI')
    parser.add_argument("--replacing_data", default=None, type=str, help='None(default)/SMAP/MSL/SMD/SWaT/WADI')
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--window_size", default=512, type=int)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--d_embed", default=512, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--replacing_rate_max", default=0.15, type=float)
    
    parser.add_argument("--soft_replacing", default=0.5, type=float)
    parser.add_argument("--uniform_replacing", default=0.15, type=float)
    parser.add_argument("--peak_noising", default=0.15, type=float)
    parser.add_argument("--length_adjusting", default=0.1, type=float)
    parser.add_argument("--white_noising", default=0.0, type=float)
    
    parser.add_argument("--flip_replacing_interval", default='all', type=str,
                        help='vertical/horizontal/all/none')
    parser.add_argument("--replacing_weight", default=0.7, type=float)
    
    parser.add_argument("--window_sliding", default=16, type=int)
    parser.add_argument("--data_division", default='total', type=str, help='channel/class/total')
    
    parser.add_argument("--loss", default='bce', type=str)
    parser.add_argument("--total_loss", default=0.2, type=float)
    parser.add_argument("--partial_loss", default=1., type=float)
    parser.add_argument("--contrastive_loss", default=0., type=float)
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)
    
    options = parser.parse_args()
    main(options)