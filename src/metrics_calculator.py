import numpy as np

# Метрики регрессии
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Метрики классификации
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class MetricsCalculator:

    # функция для расчёта метрик регрессии
    def regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)

        return {
            'MAE': mean_absolute_error(y_true, y_pred),  # средняя абсолютная ошибка
            'MSE': mse,  # средняя квадратичная ошибка
            'RMSE': np.sqrt(mse),  # корень из MSE
            'R2': r2_score(y_true, y_pred)  # коэффициент детерминации
        }

    # базовые метрики классификации по предсказанным классам
    def classification_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    # ROC-AUC по вероятностям положительного класса
    def roc_auc(self, y_true, y_proba):
        return roc_auc_score(y_true, y_proba)

    # confusion matrix
    def confusion_matrix_values(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        return {
            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1]
        }

    # всё вместе для бинарной классификации
    def full_classification_metrics(self, y_true, y_pred, y_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = cm[0, 0]
        metrics['fp'] = cm[0, 1]
        metrics['fn'] = cm[1, 0]
        metrics['tp'] = cm[1, 1]

        return metrics