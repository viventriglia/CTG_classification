import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import KernelPCA

def plot_confusion_matrix(Y_test, Y_pred, cmap='YlGnBu', figsize=(8,6), save_fig=False):
    label = ['N', 'S', 'P']
    
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_matrix(Y_test, Y_pred),
                annot = True, xticklabels = label, yticklabels = label, cmap = cmap)
    if save_fig:
        plt.savefig(f'confusion_matrix.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    
def plot_correlation_matrix(df, style='white', figsize=(11,9), save_fig=False):
    sns.set(style=style)
    
    # Correlation matrix
    corr = df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    if save_fig:
        plt.savefig(f'correlation_matrix.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    
def plot_distribution(df, figsize=(8,6), save_fig=False):
    colours = ['tab:green', 'gold', 'tab:red']
    distinct_count = df['NSP'].value_counts()
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x = distinct_count.index, y = distinct_count, palette = colours)
    ax.bar_label(ax.containers[0])
    sns.despine(left=True, bottom=True)
    
    x_ticks = np.arange(len(distinct_count))
    plt.xticks(x_ticks, ('Normal','Suspect', 'Pathologic'), fontsize=13)
    plt.ylabel('No. of cases', fontsize=13)
    plt.title(f'Distribution of N-S-P cases (out of {len(df)})', fontsize=15, style='italic')
    if save_fig:
        plt.savefig(f'distribution_of_cases.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    

def plot_PCA(X, Y, classifiers, titles, figsize=(10,5), cmap='viridis', savefig=False):
    models = zip(titles,classifiers)
    
    for title, kpca in models:
        X_PCA = kpca.fit_transform(X)
        
        style.use('ggplot')
        
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, projection='3d')
        loc = [1,2,3]
        classes = ['Normal','Suspect','Pathologic']
        x3d = X_PCA[:,0]
        y3d = X_PCA[:,1]
        z3d = X_PCA[:,2]
        
        plot = ax1.scatter(x3d, y3d, z3d, c=Y, cmap=cmap)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        cb = plt.colorbar(plot)
        cb.set_ticks(loc)
        cb.set_ticklabels(classes)
        
        plt.title(title, style='italic')
        if savefig:
            plt.savefig(f'PCA_{name}.png',
                        dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()