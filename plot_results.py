import os
import torch
import cycler
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt

root_dir = "../gate-detection-FCN/"
eval_frames = pd.read_csv(os.path.join(root_dir, "logs/eval_IoU_PA_inference.csv"))

def ewma_average(loss, alpha):
    """
    Exponentially weighted moving average for smooth loss-plotting.
    Inputs: loss: 1-dim loss data to plot; alpha: discount factor.
    S_t = loss_t, t = 0
    S_t = alpha * S_(t - 1) + (1 - alpha) * y_t, t > 0
    """
    s = np.zeros(len(loss))
    for t in range(0, len(loss)):
        if t == 0:
            s[t] = loss[t]
        else:
            s[t] = alpha * s[t - 1] + (1 - alpha) * loss[t]
    return s


def plot_loss_curve():
    vgg_configs = ['5', '11']
    upsample_modes = ['32s', '16s', '8s']
    batch_norms = [False, True]
    # set plot style
    sb.set_style('whitegrid')
    num_curves = 9
    color = plt.cm.Spectral(np.linspace(0, 1, num_curves))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    mpl.rcParams.update({'font.size': 20})

    plt.figure(dpi=500, figsize=(10, 7))
    for upsample_mode in upsample_modes:
        for batch_norm in batch_norms:
            for vgg_config in vgg_configs:
                log_path = os.path.join(root_dir,
                                        "logs/loss_vgg-{}_{}_bn_{}.txt".format(vgg_config, upsample_mode, str(batch_norm)))
                if os.path.exists(log_path):
                    loss = np.loadtxt(log_path, dtype=np.float32, delimiter=',')
                    averaged_loss = ewma_average(loss[:, 1], 0.95)
                    label = 'vgg{}-{}-bn'.format(vgg_config, upsample_mode) if batch_norm \
                        else 'vgg{}-{}'.format(vgg_config, upsample_mode)
                    plt.plot(averaged_loss, label=label, linewidth=3)

    plt.xlabel("epochs (iteration number)")
    plt.ylabel("loss")
    plt.xlim(0, 1000)
    plt.ylim(0, 0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir+'figures/loss_curve.png')


def plot_bar(metric_id, ax, legend):
    assert metric_id in ['IoU', 'pixel acc.'], "Metric not defined: must either be \'IoU\' or \'pixel acc.\'"
    metric = {'IoU':1, 'pixel acc.':2}
    upsample_modes = ["32s", "16s", "8s"]

    bar_labels = ['vgg5', 'vgg11', 'vgg11-bn']
    width = 0.1
    # bar plot for IoU
    for vgg_config in range(3): # vgg-5, vgg-11, vgg-11-bn
        steps = [i+(vgg_config-1)*width for i in range(len(upsample_modes))]
        ax.bar(x=steps,
                height=eval_frames.iloc[[i + vgg_config for i in range(0, 9, 3)], metric[metric_id]],
                width=width,
                label=bar_labels[vgg_config])
    
    ax.set_xticks([i for i in range(len(upsample_modes))])
    ax.set_xticklabels(upsample_modes)
    #ax.set_xlabel("FCN upsampling structures", fontsize=20)
    ax.set_ylabel("mean {}".format(metric_id), fontsize=20)
    ax.set_ylim(0, 1.0)

    if legend:
        ax.legend(loc='lower right', framealpha=0.95)


def plot_IoU_PixelAcc():
    plt.figure(dpi=500, figsize=(10, 5))
    # set plot style
    sb.set_style('whitegrid')
    num_interval = 10
    color = plt.cm.tab10(np.linspace(0, 1, num_interval)) # tab10 with 10 intervals
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    mpl.rcParams.update({'font.size': 20})
    # subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plot_bar('IoU', ax1, legend=False)
    plot_bar('pixel acc.', ax2, legend=True)
    plt.tight_layout()
    plt.savefig(root_dir+'figures/mean_IoU_PixelAcc.png')


if __name__ == '__main__':
    plot_loss_curve()
    plot_IoU_PixelAcc()
