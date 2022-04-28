import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp

saved_study_losses_file = 'losses_on_study.npz'
data = np.load(saved_study_losses_file)
imagenet_losses, saycam_losses, tabularasa_losses = data['imagenet_losses'], data['saycam_losses'], data['tabularasa_losses']

i_m = np.mean(imagenet_losses, 0)
i_s = np.std(imagenet_losses, 0) / np.sqrt(imagenet_losses.shape[0])

s_m = np.mean(saycam_losses, 0)
s_s = np.std(saycam_losses, 0) / np.sqrt(saycam_losses.shape[0])

t_m = np.mean(tabularasa_losses, 0)
t_s = np.std(tabularasa_losses, 0) / np.sqrt(tabularasa_losses.shape[0])

x30 = np.linspace(1, 30, 30)
x100 = np.linspace(1, 100, 100)

# PLOTTING
plt.clf()
ax = plt.subplot(111)
plt.plot(x30, i_m, '-', color='r')
plt.fill_between(x30, i_m - i_s, i_m + i_s, color='pink')

plt.plot(x30, s_m, '-', color='b')
plt.fill_between(x30, s_m - s_s, s_m + s_s, color='lightskyblue')

plt.plot(x100, t_m, '-', color='k')
plt.fill_between(x100, t_m - t_s, t_m + t_s, color='lightgray')

plt.xlim([0, 101])
plt.ylim([0, 3])
plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3], ['0', '', '1', '', '2', '', '3'], fontsize=15)
plt.xticks([1, 25, 50, 75, 100], ['1', '', '50', '', '100'], fontsize=15)
plt.xlabel('Number of exposures (epochs)', fontsize=15)
plt.ylabel('Loss on study set (pixelwise $nll$)', fontsize=15)
plt.text(60, 2.75, 'Tabula rasa', fontsize=15, color='k')
plt.text(60, 2.55, 'ImageNet-pretrained', fontsize=15, color='r')
plt.text(60, 2.35, 'SAYCam-pretrained', fontsize=15, color='b')

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

mp.rcParams['axes.linewidth'] = 0.75
mp.rcParams['patch.linewidth'] = 0.75
mp.rcParams['patch.linewidth'] = 1.15
mp.rcParams['font.sans-serif'] = ['FreeSans']
mp.rcParams['mathtext.fontset'] = 'cm'

plt.savefig('losses_on_study.pdf', bbox_inches='tight')