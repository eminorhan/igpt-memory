import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp

data_dir = '/home/emin/Documents/brady/losses_on_test/data_files_size'
data_dir_n = '/home/emin/Documents/brady/losses_on_test/data_files_n'

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

def compute_triplet_mean_ste_size():
    '''conditions: novel, exemplar, state'''
    novel_mini_losses = compute_accuracy(os.path.join(data_dir, 'novel_seen_mini'), os.path.join(data_dir, 'novel_unseen_mini'))
    novel_control_losses = compute_accuracy(os.path.join(data_dir, 'novel_seen_control'), os.path.join(data_dir, 'novel_unseen_control'))

    exemplar_mini_losses = compute_accuracy(os.path.join(data_dir, 'exemplar_seen_mini'), os.path.join(data_dir, 'exemplar_unseen_mini'))
    exemplar_control_losses = compute_accuracy(os.path.join(data_dir, 'exemplar_seen_control'), os.path.join(data_dir, 'exemplar_unseen_control'))

    state_mini_losses = compute_accuracy(os.path.join(data_dir, 'state_seen_mini'), os.path.join(data_dir, 'state_unseen_mini'))
    state_control_losses = compute_accuracy(os.path.join(data_dir, 'state_seen_control'), os.path.join(data_dir, 'state_unseen_control'))

    mini_losses = np.concatenate((novel_mini_losses, exemplar_mini_losses, state_mini_losses))
    control_losses = np.concatenate((novel_control_losses, exemplar_control_losses, state_control_losses))

    m_m = np.mean(mini_losses, 0)
    m_s = np.std(mini_losses, 0) / np.sqrt(mini_losses.shape[0])

    c_m = np.mean(control_losses, 0)
    c_s = np.std(control_losses, 0) / np.sqrt(control_losses.shape[0])

    return 100*m_m, 100*m_s, 100*c_m, 100*c_s

def compute_triplet_mean_ste_n():
    '''conditions: novel, exemplar, state'''
    novel_1pt_losses = compute_accuracy(os.path.join(data_dir_n, 'novel_seen_1pt'), os.path.join(data_dir_n, 'novel_unseen_1pt'))
    novel_10pt_losses = compute_accuracy(os.path.join(data_dir_n, 'novel_seen_10pt'), os.path.join(data_dir_n, 'novel_unseen_10pt'))
    novel_100pt_losses = compute_accuracy(os.path.join(data_dir_n, 'novel_seen_100pt'), os.path.join(data_dir_n, 'novel_unseen_100pt'))

    exemplar_1pt_losses = compute_accuracy(os.path.join(data_dir_n, 'exemplar_seen_1pt'), os.path.join(data_dir_n, 'exemplar_unseen_1pt'))
    exemplar_10pt_losses = compute_accuracy(os.path.join(data_dir_n, 'exemplar_seen_10pt'), os.path.join(data_dir_n, 'exemplar_unseen_10pt'))
    exemplar_100pt_losses = compute_accuracy(os.path.join(data_dir_n, 'exemplar_seen_100pt'), os.path.join(data_dir_n, 'exemplar_unseen_100pt'))

    state_1pt_losses = compute_accuracy(os.path.join(data_dir_n, 'state_seen_1pt'), os.path.join(data_dir_n, 'state_unseen_1pt'))
    state_10pt_losses = compute_accuracy(os.path.join(data_dir_n, 'state_seen_10pt'), os.path.join(data_dir_n, 'state_unseen_10pt'))
    state_100pt_losses = compute_accuracy(os.path.join(data_dir_n, 'state_seen_100pt'), os.path.join(data_dir_n, 'state_unseen_100pt'))

    losses_1pt = np.concatenate((novel_1pt_losses, exemplar_1pt_losses, state_1pt_losses))
    losses_10pt = np.concatenate((novel_10pt_losses, exemplar_10pt_losses, state_10pt_losses))
    losses_100pt = np.concatenate((novel_100pt_losses, exemplar_100pt_losses, state_100pt_losses))

    mean_1pt = np.mean(losses_1pt, 0)
    ste_1pt = np.std(losses_1pt, 0) / np.sqrt(losses_1pt.shape[0])

    mean_10pt = np.mean(losses_10pt, 0)
    ste_10pt = np.std(losses_10pt, 0) / np.sqrt(losses_10pt.shape[0])

    mean_100pt = np.mean(losses_100pt, 0)
    ste_100pt = np.std(losses_100pt, 0) / np.sqrt(losses_100pt.shape[0])

    return 100*mean_1pt, 100*ste_1pt, 100*mean_10pt, 100*ste_10pt, 100*mean_100pt, 100*ste_100pt,

def plot_triplet_mean_ste():
    '''condition: novel, exemplar, state'''
    x25 = np.linspace(1, 25, 25)
    x50 = np.linspace(1, 50, 50)
    x45 = np.linspace(1, 45, 45)
    x30 = np.linspace(1, 30, 30)

    plt.clf()

    # PLOTTING (1) -- SIZE
    i_m, i_s, s_m, s_s= compute_triplet_mean_ste_size()
    ax = plt.subplot(221)
    plt.plot(x50, i_m, '-', color='olive')
    plt.fill_between(x50, i_m - i_s, i_m + i_s, color='beige')
    plt.plot(x25, s_m, '-', color='indigo')
    plt.fill_between(x25, s_m - s_s, s_m + s_s, color='plum')
    plt.xlim([0, 51])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['50', '60', '70', '80', '90', '100'], fontsize=12)
    plt.xticks([1, 12.5, 25, 37.5, 50], ['1', '', '25', '', '50'], fontsize=12)
    plt.xlabel('Number of exposures (epochs)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of model size', fontsize=12)
    plt.text(9, 51, 'iGPT-mini (~4x smaller)', fontsize=12, color='olive')
    plt.text(9, 56, 'iGPT-S', fontsize=12, color='indigo')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # PLOTTING (2) -- N
    mean_1pt, ste_1pt, mean_10pt, ste_10pt, mean_100pt, ste_100pt = compute_triplet_mean_ste_n()
    ax2 = plt.subplot(222)
    
    plt.plot(x45, mean_1pt, '-', color='darkgoldenrod')
    plt.fill_between(x45, mean_1pt - ste_1pt, mean_1pt + ste_1pt, color='gold')

    plt.plot(x45, mean_10pt, '-', color='lightseagreen')
    plt.fill_between(x45, mean_10pt - ste_10pt, mean_10pt + ste_10pt, color='aquamarine')

    plt.plot(x30, mean_100pt, '-', color='crimson')
    plt.fill_between(x30, mean_100pt - ste_100pt, mean_100pt + ste_100pt, color='pink')

    plt.xlim([0, 51])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['50', '60', '70', '80', '90', '100'], fontsize=12)
    plt.xticks([1, 12.5, 25, 37.5, 50], ['1', '', '25', '', '50'], fontsize=12)
    plt.xlabel('Number of exposures (epochs)', fontsize=12)
    # plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of pretraining data size', fontsize=12)
    plt.text(5, 51, '1% ', fontsize=12, color='darkgoldenrod')
    plt.text(5, 55, '10% ', fontsize=12, color='lightseagreen')
    plt.text(5, 59, '100% ', fontsize=12, color='crimson')
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

    mp.rcParams['axes.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 1.15
    mp.rcParams['font.sans-serif'] = ['FreeSans']
    mp.rcParams['mathtext.fontset'] = 'cm'

    plt.savefig('accs_on_test_size_n.pdf', bbox_inches='tight')

plot_triplet_mean_ste()