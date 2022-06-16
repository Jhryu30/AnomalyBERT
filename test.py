import os, json
import numpy as np
import pandas as pd
import torch
import argparse

from utils.datasets import get_data



def main(options):
    # SMAP dataset
    (_, _), (test_smap, test_label_smap) = get_data('SMAP')
    
    # Test class
    with open('smap_test_channel.json', 'r') as f:
        test_class = json.load(f)

    # Load model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = torch.load(options.model, map_location=device)
    
    # List state dicts.
    state_dicts = []
    file_dir = options.state_dict
    if not file_dir.endswith('.pt') and os.path.exists(file_dir):
        for file in os.listdir(file_dir):
            if file.endswith('.pt'):
                state_dicts.append(os.path.join(file_dir, file))
    else:
        state_dicts.append(file_dir)
        
        
    # Define evaluation metrics.
    diff_sq = lambda pred, gt : (pred - gt) ** 2
    diff_abs = lambda pred, gt : (pred - gt).abs()
    
    l1_ptwise = lambda _diff_abs : _diff_abs.sum(dim=-1)  # pointwise L1 norm
    l1_window = lambda _l1_ptwise : _l1_ptwise.mean(dim=-1)  # window-wise L1 norm
    l2_ptwise = lambda _diff_sq : torch.sqrt(_diff_sq.sum(dim=-1))  # pointwise L2 norm
    l2_window = lambda _l2_ptwise : _l2_ptwise.mean(dim=-1)  # window-wise L2 norm
    frob_window = lambda _diff_sq : torch.sqrt(_diff_sq.sum(dim=(-1, -2)))  # window-wise Frobenius norm
    
    l1_area = lambda _diff_abs : _diff_abs.sum(dim=-2)  # area between curves
    l2_area = lambda _diff_sq : torch.sqrt(_diff_sq.sum(dim=-2))  # mse area between curves
    area_window = lambda _l1_area : torch.sqrt((_l1_area ** 2).sum(dim=-1))  # total area between curves
    area_mse = lambda _l2_area : _l2_area.sum(dim=-1)  # total mse area between curves
    
    # Evaludation function
    def evaluate(x):
        # Reconstruct data.
        y, _ = model(x)
        
        # Set binary columns.
        y[:, :, 1:] = (y[:, :, 1:] > 0.5).float()

        # Evaluate.
        _diff_sq = diff_sq(x, y)
        _diff_abs = diff_abs(x, y)

        _l1_ptwise = l1_ptwise(_diff_abs)
        _l2_ptwise = l2_ptwise(_diff_sq)

        _l1_window = l1_window(_l1_ptwise)
        _l2_window = l2_window(_l2_ptwise)
        _frob_window = frob_window(_diff_sq)
        
        _l1_area = l1_area(_diff_abs)
        _l2_area = l2_area(_diff_sq)
        _area_window = area_window(_l1_area)
        _area_mse = area_mse(_l2_area)
        
        unsq_area_mse = _area_mse.unsqueeze(-1)
        
        return torch.stack([_l1_ptwise, _l2_ptwise, _l2_ptwise * unsq_area_mse, _l2_ptwise * (unsq_area_mse ** 2), _l2_ptwise * (unsq_area_mse ** 3)]),\
               torch.stack([_l1_window, _l2_window, _frob_window, _area_window, _area_mse, _l1_area[:, 0], _l2_area[:, 0]]).transpose(0, 1)
    
    # Test model.
    for state_dict in state_dicts:
        model.load_state_dict(torch.load(state_dict, map_location='cpu'))
        model.eval()
        
        # Test each class.
#         for _class, class_index in test_class.items():
        for channel, class_index in enumerate(test_class):
            _test_smap = test_smap[class_index[0]:class_index[1]]
            _test_label_smap = test_label_smap[class_index[0]:class_index[1]]

            # Window setting
            window_size = model.max_seq_len
            data_len = len(_test_label_smap)
            last_window = data_len - window_size + 1
            n_batch = options.batch_size
            window_sliding = options.window_sliding
            batch_sliding = n_batch * window_size
            _batch_sliding = n_batch * window_sliding

            # Error lists
            eval_lists = torch.zeros(data_len, 12, device=device)
            count = 0
            checked_index = 50000

            with torch.no_grad():
                _first = -batch_sliding
                for first in range(0, last_window-batch_sliding+1, batch_sliding):
                    for i in range(first, first+window_size, window_sliding):
                        # Call mini-batch data.
                        x = torch.Tensor(_test_smap[i:i+batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)

                        # Evaludate and record errors.
                        _ptwise, eval_values = evaluate(x)
                        _ptwise = _ptwise.view(5, batch_sliding).contiguous().transpose(0, 1)
                        eval_values = eval_values.repeat_interleave(window_size, dim=0)
                        eval_lists[i:i+batch_sliding, :5] += _ptwise
                        eval_lists[i:i+batch_sliding, 5:] += eval_values

                        count += n_batch

                        if count > checked_index:
                            print(count, 'windows are computed.')
                            checked_index += 50000

                    _first = first

                _first += batch_sliding
                for first, last in zip(range(_first, last_window, _batch_sliding),
                                       list(range(_first+_batch_sliding, last_window, _batch_sliding)) + [last_window]):
                    # Call mini-batch data.
                    x = []
                    for i in range(first, last, window_sliding):
                        x.append(torch.Tensor(_test_smap[i:i+window_size].copy()))

                    # Reconstruct data.
                    x = torch.stack(x).to(device)

                    # Evaludate and record errors.
                    _ptwise, eval_values = evaluate(x)
                    _ptwise = _ptwise.permute(1, 2, 0)
                    for i, j in enumerate(range(first, last, window_sliding)):
                        eval_lists[j:j+window_size, :5] += _ptwise[i]
                        eval_lists[j:j+window_size, 5:] += eval_values[[i]]

                    count += n_batch

                    if count > checked_index:
                        print(count, 'windows are computed.')
                        checked_index += 50000

                # Compute mean errors.
                window_overlap = window_size // window_sliding
                n_errors = torch.full((data_len,), window_overlap)
                n_errors[:window_size-window_sliding] = torch.repeat_interleave(torch.arange(1, window_overlap), window_sliding)
                n_errors[-window_size+window_sliding:] = torch.repeat_interleave(torch.arange(window_overlap-1, 0, -1), window_sliding)
                eval_lists = eval_lists / n_errors.unsqueeze(1).to(device)

                # Save errors.
                columns = ['pointwise_L1', 'pointwise_L2', 'prod_L2_area_1', 'prod_L2_area_2', 'prod_L2_area_3',
                           'window-wise_L1', 'window-wise_L2', 'Frobenius', 'area_between_curves', 'mse_area',
                           'L1_area_dim_1', 'L2_area_dim_1']
                df = pd.DataFrame(eval_lists.cpu().numpy(), columns=columns)
                df['anomaly_label'] = _test_label_smap
#                 df.to_csv(state_dict[:-3]+'_eval_class_'+_class+'.csv')
                df.to_csv(state_dict[:-3]+'_eval_channel_{}.csv'.format(channel))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--state_dict", required=True, type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--window_sliding", default=1, type=int)
    
    options = parser.parse_args()
    main(options)