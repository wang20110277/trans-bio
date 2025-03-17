import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    """计算精确率"""
    return precision_score(y_true, y_pred)

def calculate_recall(y_true, y_pred):
    """计算召回率"""
    return recall_score(y_true, y_pred)

def calculate_f1(y_true, y_pred):
    """计算 F1 分数"""
    return f1_score(y_true, y_pred)

def calculate_roc_auc(y_true, y_pred_proba):
    """计算 ROC AUC 分数"""
    return roc_auc_score(y_true, y_pred_proba)

def calculate_confusion_matrix(y_true, y_pred):
    """计算混淆矩阵"""
    return confusion_matrix(y_true, y_pred)

def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """计算所有评估指标"""
    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred),
        "recall": calculate_recall(y_true, y_pred),
        "f1": calculate_f1(y_true, y_pred),
        "roc_auc": calculate_roc_auc(y_true, y_pred_proba),
        "confusion_matrix": calculate_confusion_matrix(y_true, y_pred),
    }
    return metrics

def print_metrics(metrics):
    """打印评估指标"""
    print("模型评估结果：")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1 分数 (F1 Score): {metrics['f1']:.4f}")
    print(f"ROC AUC 分数: {metrics['roc_auc']:.4f}")
    print("混淆矩阵 (Confusion Matrix):")
    print(metrics["confusion_matrix"])

if __name__ == "__main__":
    # 示例数据
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.6, 0.7])

    # 计算所有评估指标
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)

    # 打印评估结果
    print_metrics(metrics)