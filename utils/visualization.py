import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

def plot_loss_curve(history):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="训练损失")
    plt.plot(history.history["val_loss"], label="验证损失")
    plt.title("训练和验证损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_curve(history):
    """绘制训练和验证准确率曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="训练准确率")
    plt.plot(history.history["val_accuracy"], label="验证准确率")
    plt.title("训练和验证准确率曲线")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """绘制 ROC 曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC 曲线 (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("假正率 (FPR)")
    plt.ylabel("真正率 (TPR)")
    plt.title("ROC 曲线")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 示例数据
    history = {
        "loss": [0.5, 0.3, 0.2, 0.1, 0.05],
        "val_loss": [0.6, 0.4, 0.3, 0.2, 0.15],
        "accuracy": [0.8, 0.85, 0.9, 0.92, 0.95],
        "val_accuracy": [0.75, 0.8, 0.85, 0.88, 0.9],
    }

    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.6, 0.7])

    # 绘制损失曲线
    plot_loss_curve(history)

    # 绘制准确率曲线
    plot_accuracy_curve(history)

    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, class_names=["Class 0", "Class 1"])

    # 绘制 ROC 曲线
    plot_roc_curve(y_true, y_pred_proba)