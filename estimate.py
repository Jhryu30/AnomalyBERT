import os, json
import numpy as np
import torch
import argparse

import utils.config as config



def main(options):
    # Load test data.
    test_data = np.load(config.TEST_DATASET[options.dataset]).copy()
    test_label = np.load(config.TEST_LABEL[options.dataset]).copy()
    
    # Load model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = torch.load(options.model, map_location=device)
    model.load_state_dict(torch.load(options.state_dict, map_location='cpu'))
    model.eval()
    
    # Data division
    if options.data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][options.data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
    
    # Estimation settings
    window_size = model.max_seq_len * model.patch_size
    n_column = len(test_data[0]) if options.reconstruction_output else 1
    n_batch = options.batch_size
    window_sliding = options.window_sliding
    batch_sliding = n_batch * window_size
    _batch_sliding = n_batch * window_sliding
    
    output_values = torch.zeros(len(test_data), n_column, device=device)
    count = 0
    checked_index = options.check_count
    
    post_activation = torch.nn.Identity().to(device) if options.reconstruction_output\
                      else torch.nn.Sigmoid().to(device)
    
    # Record output values.
    for division in divisions:
        data_len = division[1] - division[0]
        last_window = data_len - window_size + 1
        _test_data = test_data[division[0]:division[1]]
        _output_values = torch.zeros(data_len, n_column, device=device)
    
        with torch.no_grad():
            _first = -batch_sliding
            for first in range(0, last_window-batch_sliding+1, batch_sliding):
                for i in range(first, first+window_size, window_sliding):
                    # Call mini-batch data.
                    x = torch.Tensor(_test_data[i:i+batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)

                    # Evaludate and record errors.
                    y = post_activation(model(x))
                    _output_values[i:i+batch_sliding] += y.view(-1, n_column)

                    count += n_batch

                    if count > checked_index:
                        print(count, 'windows are computed.')
                        checked_index += options.check_count

                _first = first

            _first += batch_sliding

            for first, last in zip(range(_first, last_window, _batch_sliding),
                                   list(range(_first+_batch_sliding, last_window, _batch_sliding)) + [last_window]):
                # Call mini-batch data.
                x = []
                for i in range(first, last, window_sliding):
                    x.append(torch.Tensor(_test_data[i:i+window_size].copy()))

                # Reconstruct data.
                x = torch.stack(x).to(device)

                # Evaludate and record errors.
                y = post_activation(model(x))
                for i, j in enumerate(range(first, last, window_sliding)):
                    _output_values[j:j+window_size] += y[i]

                count += n_batch

                if count > checked_index:
                    print(count, 'windows are computed.')
                    checked_index += options.check_count

            # Compute mean values.
            window_overlap = window_size // window_sliding
            n_overlap = torch.full((data_len,), window_overlap)
            n_overlap[:window_size-window_sliding] = torch.repeat_interleave(torch.arange(1, window_overlap), window_sliding)
            n_overlap[-window_size+window_sliding:] = torch.repeat_interleave(torch.arange(window_overlap-1, 0, -1), window_sliding)
            _output_values = _output_values / n_overlap.unsqueeze(-1).to(device)
            
            # Record values for the division.
            output_values[division[0]:division[1]] = _output_values
        
    # Save results.
    output_values = output_values.cpu().numpy()
    outfile = options.state_dict[:-3] + '_results.npy' if options.outfile == None else options.outfile
    np.save(outfile, output_values)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--dataset", default='SMAP', type=str, help='SMAP/MSL/SMD')
    
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--state_dict", required=True, type=str)
    parser.add_argument("--outfile", default=None, type=str)
    
    parser.add_argument("--data_division", default='total', type=str, help='channel/class/total')
    parser.add_argument("--check_count", default=5000, type=int)
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--window_sliding", default=16, type=int)
    parser.add_argument('--reconstruction_output', default=False, action='store_true')
    
    options = parser.parse_args()
    main(options)