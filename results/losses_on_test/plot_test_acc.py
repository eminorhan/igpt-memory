import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp

data_dir = '/home/emin/Documents/brady/losses_on_test/data_files'

def read_dir(dir):
    files = os.listdir(dir)
    files.sort()
    losses = [np.load(os.path.join(dir, f)) for f in files]
    return losses

def compute_accuracy(seen, unseen):
    seen_losses = read_dir(seen)
    unseen_losses = read_dir(unseen)
    accs = [np.mean(x<y, 1) for x, y in zip(seen_losses, unseen_losses)]
    return np.array(accs)

def compute_triplet_mean_ste(condition):
    '''condition: novel, exemplar, state'''
    imagenet_losses = compute_accuracy(os.path.join(data_dir, condition + '_seen_imagenet'), os.path.join(data_dir, condition + '_unseen_imagenet'))
    saycam_losses = compute_accuracy(os.path.join(data_dir, condition + '_seen_saycam'), os.path.join(data_dir, condition + '_unseen_saycam'))
    tabularasa_losses = compute_accuracy(os.path.join(data_dir, condition + '_seen_tabularasa'), os.path.join(data_dir, condition + '_unseen_tabularasa'))
    i_m = np.mean(imagenet_losses, 0)
    i_s = np.std(imagenet_losses, 0) / np.sqrt(imagenet_losses.shape[0])

    s_m = np.mean(saycam_losses, 0)
    s_s = np.std(saycam_losses, 0) / np.sqrt(saycam_losses.shape[0])

    t_m = np.mean(tabularasa_losses, 0)
    t_s = np.std(tabularasa_losses, 0) / np.sqrt(tabularasa_losses.shape[0])

    return 100*i_m, 100*i_s, 100*s_m, 100*s_s, 100*t_m, 100*t_s

def plot_triplet_mean_ste():
    '''condition: novel, exemplar, state'''
    x30 = np.linspace(1, 30, 30)
    x100 = np.linspace(1, 100, 100)

    plt.clf()

    # PLOTTING -- 1
    i_m, i_s, s_m, s_s, t_m, t_s = compute_triplet_mean_ste('novel')
    human_m, human_s = 92.5*np.ones(100), 1.6*np.ones(100)
    ax1 = plt.subplot(231)
    plt.plot(x100, human_m, '-', color='darkorange')
    plt.fill_between(x100, human_m - human_s, human_m + human_s, color='bisque')
    plt.plot(x30, i_m, '-', color='r')
    plt.fill_between(x30, i_m - i_s, i_m + i_s, color='pink')
    plt.plot(x30, s_m, '-', color='b')
    plt.fill_between(x30, s_m - s_s, s_m + s_s, color='lightskyblue')
    plt.plot(x100, t_m, '-', color='k')
    plt.fill_between(x100, t_m - t_s, t_m + t_s, color='lightgray')
    plt.xlim([0, 101])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['50', '60', '70', '80', '90', '100'], fontsize=9)
    plt.xticks([1, 25, 50, 75, 100], ['1', '', '50', '', '100'], fontsize=9)
    plt.xlabel('Number of exposures (epochs)', fontsize=9)
    plt.ylabel('Accuracy (%)', fontsize=9)
    plt.title('Novel', fontsize=9)
    plt.text(75, 88, 'Humans', fontsize=8, color='darkorange')
    plt.text(48, 59, 'Tabula rasa', fontsize=8, color='k')
    plt.text(48, 55, 'ImageNet-pretrained', fontsize=8, color='r')
    plt.text(48, 51, 'SAYCam-pretrained', fontsize=8, color='b')
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    # PLOTTING -- 2
    i_m, i_s, s_m, s_s, t_m, t_s = compute_triplet_mean_ste('exemplar')
    human_m, human_s = 87.6*np.ones(100), 1.8*np.ones(100)
    ax2 = plt.subplot(232)
    plt.plot(x100, human_m, '-', color='darkorange')
    plt.fill_between(x100, human_m - human_s, human_m + human_s, color='bisque')
    plt.plot(x30, i_m, '-', color='r')
    plt.fill_between(x30, i_m - i_s, i_m + i_s, color='pink')
    plt.plot(x30, s_m, '-', color='b')
    plt.fill_between(x30, s_m - s_s, s_m + s_s, color='lightskyblue')
    plt.plot(x100, t_m, '-', color='k')
    plt.fill_between(x100, t_m - t_s, t_m + t_s, color='lightgray')
    plt.xlim([0, 101])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['', '', '', '', '', ''], fontsize=9)
    plt.xticks([1, 25, 50, 75, 100], ['', '', '', '', ''], fontsize=9)
    plt.title('Exemplar', fontsize=9)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

    # PLOTTING -- 3
    i_m, i_s, s_m, s_s, t_m, t_s = compute_triplet_mean_ste('state')
    print(s_m)
    print(s_s)

    human_m, human_s = 87.2*np.ones(100), 1.8*np.ones(100)
    ax3 = plt.subplot(233)
    plt.plot(x100, human_m, '-', color='darkorange')
    plt.fill_between(x100, human_m - human_s, human_m + human_s, color='bisque')
    plt.plot(x30, i_m, '-', color='r')
    plt.fill_between(x30, i_m - i_s, i_m + i_s, color='pink')
    plt.plot(x30, s_m, '-', color='b')
    plt.fill_between(x30, s_m - s_s, s_m + s_s, color='lightskyblue')
    plt.plot(x100, t_m, '-', color='k')
    plt.fill_between(x100, t_m - t_s, t_m + t_s, color='lightgray')
    plt.xlim([0, 101])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['', '', '', '', '', ''], fontsize=9)
    plt.xticks([1, 25, 50, 75, 100], ['', '', '', '', ''], fontsize=9)
    plt.title('State', fontsize=9)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')

    # increase the spacing between subplots
    plt.subplots_adjust(hspace=0.9)    

    mp.rcParams['axes.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 1.15
    mp.rcParams['font.sans-serif'] = ['FreeSans']
    mp.rcParams['mathtext.fontset'] = 'cm'

    plt.savefig('accs_on_test.pdf', bbox_inches='tight')

plot_triplet_mean_ste()