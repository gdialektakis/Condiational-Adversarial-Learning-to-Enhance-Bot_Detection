import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme()


def plot_heatmaps():
    precision = pd.read_csv('precision.csv')
    g_mean = pd.read_csv('g-mean.csv')

    y_labels = ['Original', 'ADASYN', 'CGAN', 'AC-GAN']
    plt.figure()
    colour = sns.color_palette("Blues", as_cmap=True)
    ax_pr = sns.heatmap(precision, annot=True, fmt="f", square=True, xticklabels=True, yticklabels=y_labels,
                        cmap=colour, vmin=0.5, vmax=1)

    plt.plot(ax=ax_pr)
    plt.title('Precision')
    plt.show()

    plt.figure()
    ax_g_mean = sns.heatmap(g_mean, annot=True, fmt="f", square=True, xticklabels=True, yticklabels=y_labels,
                            cmap=colour, vmin=0.5, vmax=1)
    plt.plot(ax=ax_g_mean)
    plt.title('G-mean')
    plt.show()


plot_heatmaps()