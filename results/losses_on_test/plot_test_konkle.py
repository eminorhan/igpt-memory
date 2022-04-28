import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp

data_dir = '/home/emin/Documents/brady/losses_on_test/data_files_konkle'

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
    '''condition: novel, 1_exemplar, 2_exemplar, 4_exemplar, 8_exemplar, 16_exemplar'''
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
    '''condition: novel, 1_exemplar, 2_exemplar, 4_exemplar, 8_exemplar, 16_exemplar'''
    x25 = np.linspace(1, 25, 25)
    x150 = np.linspace(1, 150, 150)

    i_m_novel, i_s_novel, s_m_novel, s_s_novel, t_m_novel, t_s_novel = compute_triplet_mean_ste('novel')
    i_m_1_exemplar, i_s_1_exemplar, s_m_1_exemplar, s_s_1_exemplar, t_m_1_exemplar, t_s_1_exemplar = compute_triplet_mean_ste('1_exemplar')
    i_m_2_exemplar, i_s_2_exemplar, s_m_2_exemplar, s_s_2_exemplar, t_m_2_exemplar, t_s_2_exemplar = compute_triplet_mean_ste('2_exemplar')
    i_m_4_exemplar, i_s_4_exemplar, s_m_4_exemplar, s_s_4_exemplar, t_m_4_exemplar, t_s_4_exemplar = compute_triplet_mean_ste('4_exemplar')
    i_m_8_exemplar, i_s_8_exemplar, s_m_8_exemplar, s_s_8_exemplar, t_m_8_exemplar, t_s_8_exemplar = compute_triplet_mean_ste('8_exemplar')
    i_m_16_exemplar, i_s_16_exemplar, s_m_16_exemplar, s_s_16_exemplar, t_m_16_exemplar, t_s_16_exemplar = compute_triplet_mean_ste('16_exemplar')

    plt.clf()

    # PLOTTING -- 1 Tabula rasa
    ax1 = plt.subplot(231)
    plt.plot(x150, t_m_novel, '-', color=[0, 0, 0])
    plt.fill_between(x150, t_m_novel - t_s_novel, t_m_novel + t_s_novel, color='lightgray')

    plt.plot(x150, t_m_1_exemplar, '-', color=[.1, .1, .1])
    plt.fill_between(x150, t_m_1_exemplar - t_s_1_exemplar, t_m_1_exemplar + t_s_1_exemplar, color='lightgray')

    plt.plot(x150, t_m_2_exemplar, '-', color=[.2, .2, .2])
    plt.fill_between(x150, t_m_2_exemplar - t_s_2_exemplar, t_m_2_exemplar + t_s_2_exemplar, color='lightgray')

    plt.plot(x150, t_m_4_exemplar, '-', color=[.3, .3, .3])
    plt.fill_between(x150, t_m_4_exemplar - t_s_4_exemplar, t_m_4_exemplar + t_s_4_exemplar, color='lightgray')

    plt.plot(x150, t_m_8_exemplar, '-', color=[.5, .5, .5])
    plt.fill_between(x150, t_m_8_exemplar - t_s_8_exemplar, t_m_8_exemplar + t_s_8_exemplar, color='lightgray')

    plt.plot(x150, t_m_16_exemplar, '-', color=[.7, .7, .7])
    plt.fill_between(x150, t_m_16_exemplar - t_s_16_exemplar, t_m_16_exemplar + t_s_16_exemplar, color='lightgray')

    plt.xlim([0, 151])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['50', '60', '70', '80', '90', '100'], fontsize=9)
    plt.xticks([1, 50, 100, 150], ['1', '', '', '150'], fontsize=9)
    plt.xlabel('Number of exposures (epochs)', fontsize=9)
    plt.ylabel('Accuracy (%)', fontsize=9)
    plt.title('Tabula rasa', color='k', fontsize=9)
    plt.text(120, 71, 'Novel', fontsize=8, color=[0, 0, 0])
    plt.text(120, 67, '1 ex.', fontsize=8, color=[.1, .1, .1])
    plt.text(120, 63, '2 ex.', fontsize=8, color=[.2, .2, .2])
    plt.text(120, 59, '4 ex.', fontsize=8, color=[.3, .3, .3])
    plt.text(120, 55, '8 ex.', fontsize=8, color=[.5, .5, .5])
    plt.text(114, 51, '16 ex.', fontsize=8, color=[.7, .7, .7])
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    # PLOTTING -- 2 ImageNet-pretrained
    ax2 = plt.subplot(232)
    plt.plot(x25, i_m_novel, '-', color=[0.1, 0, 0])
    plt.fill_between(x25, i_m_novel - i_s_novel, i_m_novel + i_s_novel, color='lightpink')

    plt.plot(x25, i_m_1_exemplar, '-', color=[0.3, 0, 0])
    plt.fill_between(x25, i_m_1_exemplar - i_s_1_exemplar, i_m_1_exemplar + i_s_1_exemplar, color='lightpink')

    plt.plot(x25, i_m_2_exemplar, '-', color=[0.5, 0, 0])
    plt.fill_between(x25, i_m_2_exemplar - i_s_2_exemplar, i_m_2_exemplar + i_s_2_exemplar, color='lightpink')

    plt.plot(x25, i_m_4_exemplar, '-', color=[0.7, 0, 0])
    plt.fill_between(x25, i_m_4_exemplar - i_s_4_exemplar, i_m_4_exemplar + i_s_4_exemplar, color='lightpink')

    plt.plot(x25, i_m_8_exemplar, '-', color=[0.9, 0, 0])
    plt.fill_between(x25, i_m_8_exemplar - i_s_8_exemplar, i_m_8_exemplar + i_s_8_exemplar, color='lightpink')

    plt.plot(x25, i_m_16_exemplar, '-', color=[1, 0, 0])
    plt.fill_between(x25, i_m_16_exemplar - i_s_16_exemplar, i_m_16_exemplar + i_s_16_exemplar, color='lightpink')

    plt.xlim([0, 26])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['', '', '', '', '', ''], fontsize=9)
    plt.xticks([1, 13, 25], ['1', '', '25'], fontsize=9)
    plt.title('ImageNet-pretrained', color='r', fontsize=9)
    plt.text(20, 71, 'Novel', fontsize=8, color=[.1, 0, 0])
    plt.text(20, 67, '1 ex.', fontsize=8, color=[.3, 0, 0])
    plt.text(20, 63, '2 ex.', fontsize=8, color=[.5, 0, 0])
    plt.text(20, 59, '4 ex.', fontsize=8, color=[.7, 0, 0])
    plt.text(20, 55, '8 ex.', fontsize=8, color=[.9, 0, 0])
    plt.text(18.9, 51, '16 ex.', fontsize=8, color=[1, 0, 0])
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

    # PLOTTING -- 3 SAYCam-pretrained
    ax3 = plt.subplot(233)
    plt.plot(x25, s_m_novel, '-', color=[0, 0, 0.1])
    plt.fill_between(x25, s_m_novel - s_s_novel, s_m_novel + s_s_novel, color='lightblue')

    plt.plot(x25, s_m_1_exemplar, '-', color=[0, 0, 0.3])
    plt.fill_between(x25, s_m_1_exemplar - s_s_1_exemplar, s_m_1_exemplar + s_s_1_exemplar, color='lightblue')

    plt.plot(x25, s_m_2_exemplar, '-', color=[0, 0, 0.5])
    plt.fill_between(x25, s_m_2_exemplar - s_s_2_exemplar, s_m_2_exemplar + s_s_2_exemplar, color='lightblue')

    plt.plot(x25, s_m_4_exemplar, '-', color=[0, 0, 0.7])
    plt.fill_between(x25, s_m_4_exemplar - s_s_4_exemplar, s_m_4_exemplar + s_s_4_exemplar, color='lightblue')

    plt.plot(x25, s_m_8_exemplar, '-', color=[0, 0, 0.9])
    plt.fill_between(x25, s_m_8_exemplar - s_s_8_exemplar, s_m_8_exemplar + s_s_8_exemplar, color='lightblue')

    plt.plot(x25, s_m_16_exemplar, '-', color=[0, 0, 1])
    plt.fill_between(x25, s_m_16_exemplar - s_s_16_exemplar, s_m_16_exemplar + s_s_16_exemplar, color='lightblue')

    plt.xlim([0, 26])
    plt.ylim([50, 100])
    plt.yticks([50, 60, 70, 80, 90, 100], ['', '', '', '', '', ''], fontsize=9)
    plt.xticks([1, 13, 25], ['1', '', '25'], fontsize=9)
    plt.title('SAYCam-pretrained', color='b', fontsize=9)
    plt.text(20, 71, 'Novel', fontsize=8, color=[0, 0, .1])
    plt.text(20, 67, '1 ex.', fontsize=8, color=[0, 0, .3])
    plt.text(20, 63, '2 ex.', fontsize=8, color=[0, 0, .5])
    plt.text(20, 59, '4 ex.', fontsize=8, color=[0, 0, .7])
    plt.text(20, 55, '8 ex.', fontsize=8, color=[0, 0, .9])
    plt.text(18.9, 51, '16 ex.', fontsize=8, color=[0, 0, 1])
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

    plt.savefig('accs_on_test_konkle.pdf', bbox_inches='tight')

plot_triplet_mean_ste()