import json, os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import utils.config as config



# Exponential weighted moving average
def ewma(series, weighting_factor=0.9):
    current_factor = 1 - weighting_factor
    _ewma = series.copy()
    for i in range(1, len(_ewma)):
        _ewma[i] = _ewma[i-1] * weighting_factor + _ewma[i] * current_factor
    return _ewma


def f1_score(gt, pr, anomaly_rate=0.05, adjust=True):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)])
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    # quantile cut
    pa = pr.copy()
    q = np.quantile(pa, 1-anomaly_rate)
    pa = (pa > q).astype(int)

    # point adjustment
    if adjust:
        for s, e in intervals:
            interval = slice(s, e)
            if pa[interval].sum() > 0:
                pa[interval] = 1

    # confusion matrix
    TP = (gt * pa).sum()
    TN = ((1 - gt) * (1 - pa)).sum()
    FP = ((1 - gt) * pa).sum()
    FN = (gt * (1 - pa)).sum()

    assert (TP + TN + FP + FN) == len(gt)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, f1_score



# Compute evaluation metrics.
def compute(options):
    # Load test data, estimation results, and label.
    test_data = np.load(config.TEST_DATASET[options.dataset])
    test_label = np.load(config.TEST_LABEL[options.dataset])
    data_dim = len(test_data[0])

    if options.data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][options.data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
        
    output_values = np.load(options.result)
    if output_values.ndim == 2:
        output_values = output_values[:, 0]
    
    if options.smooth_scores:
        smoothed_values = ewma(output_values, options.smoothing_weight)
        
    # Result text file
    if options.outfile == None:
        prefix = options.result[:-4]
        result_file = prefix + '_evaluations.txt'
    else:
        prefix = options.outfile[:-4]
        result_file = options.outfile
    result_file = open(result_file, 'w')
        
    # Save test data and output results in figures.
    if options.save_figures:
        save_folder = prefix + '_figures/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        for i, index in enumerate(divisions):
            label = test_label[index[0]:index[1]]
            
            fig, axs = plt.subplots(data_dim, 1, figsize=(20, data_dim))
            for j in range(data_dim):
                axs[j].plot(test_data[index[0]:index[1], j], alpha=0.6)
                axs[j].scatter(np.arange(index[1]-index[0])[label], test_data[index[0]:index[1]][label, j],
                                  c='r', s=1, alpha=0.8)
            fig.savefig(save_folder+'data_division_{}.jpg'.format(i), bbox_inches='tight')
            plt.close()
            
            fig, axs = plt.subplots(1, figsize=(20, 5))
            axs.plot(output_values[index[0]:index[1]], alpha=0.6)
            axs.scatter(np.arange(index[1]-index[0])[label], output_values[index[0]:index[1]][label],
                        c='r', s=1, alpha=0.8)
            fig.savefig(save_folder+'score_division_{}.jpg'.format(i), bbox_inches='tight')
            plt.close()
            
            if options.smooth_scores:
                fig, axs = plt.subplots(1, figsize=(20, 5))
                axs.plot(smoothed_values[index[0]:index[1]], alpha=0.6)
                axs.scatter(np.arange(index[1]-index[0])[label], smoothed_values[index[0]:index[1]][label],
                            c='r', s=1, alpha=0.8)
                fig.savefig(save_folder+'smoothed_score_division_{}.jpg'.format(i), bbox_inches='tight')
                plt.close()
        
    # Compute F1-scores.
    # F1 Without PA
    result_file.write('<F1-score without point adjustment>\n\n')
    best_eval = (0, 0, 0)
    best_rate = 0
    for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
        evaluation = f1_score(test_label, output_values, rate, False)
        result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
        if evaluation[2] > best_eval[2]:
            best_eval = evaluation
            best_rate = rate
    result_file.write('\nBest F1-score\n')
    result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
    print('Best F1-score without point adjustment')
    print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
    
    # F1 With PA
    result_file.write('<F1-score with point adjustment>\n\n')
    best_eval = (0, 0, 0)
    best_rate = 0
    for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
        evaluation = f1_score(test_label, output_values, rate, True)
        result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
        if evaluation[2] > best_eval[2]:
            best_eval = evaluation
            best_rate = rate
    result_file.write('\nBest F1-score\n')
    result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
    print('Best F1-score with point adjustment')
    print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
    
    if options.smooth_scores:
        # F1 Without PA
        result_file.write('<F1-score of smoothed scores without point adjustment>\n\n')
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = f1_score(test_label, smoothed_values, rate, False)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest F1-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
        print('Best F1-score of smoothed socres without point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
        
        # F1 With PA
        result_file.write('<F1-score of smoothed scores with point adjustment>\n\n')
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = f1_score(test_label, smoothed_values, rate, True)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest F1-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
        print('Best F1-score of smoothed socres with point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
    
    # Close file.
    result_file.close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='SMAP', type=str, help='SMAP/MSL/SMD/SWaT/WADI')
    parser.add_argument("--result", required=True, type=str)
    parser.add_argument("--outfile", default=None, type=str)
    parser.add_argument("--data_division", default='total', type=str, help='channel/class/total')
    
    parser.add_argument('--smooth_scores', default=False, action='store_true')
    parser.add_argument("--smoothing_weight", default=0.9, type=float)
    
    parser.add_argument('--save_figures', default=False, action='store_true')
    
    parser.add_argument("--min_anomaly_rate", default=0.001, type=float)
    parser.add_argument("--max_anomaly_rate", default=0.3, type=float)
    
    options = parser.parse_args()
    compute(options)