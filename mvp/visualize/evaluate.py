import numpy as np
import matplotlib.pyplot as plt


def draw_distribution(data_items, labels, show=True, save=None, **kwargs):
    for data_item, label in zip(data_items, labels):
        plt.hist(data_item, label=label, alpha=0.5, **kwargs)
    plt.legend()
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()


def draw_detection_roc(normal_values, attack_values, show=True, save=None):
    all_values = np.stack([normal_values, attack_values])
    tpr_data = []
    fpr_data = []
    roc_auc = 0
    for thres in np.arange(all_values.min()-0.1, all_values.max()+1, 0.02).tolist():
        TP = np.sum(attack_values > thres)
        FP = np.sum(normal_values > thres)
        P = len(attack_values)
        N = len(normal_values)
        PP = TP + FP
        PN = P + N - PP
        TPR = TP / P
        FPR = FP / N
        if TPR * (1 - FPR) > roc_auc:
            roc_auc = TPR * (1 - FPR)
        tpr_data.append(TPR)
        fpr_data.append(FPR)
    
    plt.plot(fpr_data, tpr_data, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.gca().set_aspect('equal', adjustable='box')
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()