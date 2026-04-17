import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DatasetPreprocessor(BaseEstimator, TransformerMixin):

    # список таргетов
    targets = ['IC50, mM', 'CC50, mM', 'SI']

    def __init__(
        self,
        target_column=None,
        drop_unnamed=True,
        drop_leakage=True,
        drop_constant_features=True,
        drop_corr=True,
        corr_threshold=0.9,
        log_features=False,
        log_skew_threshold=2.0,
        log_min_unique_values=10
    ):
        self.target_column = target_column
        self.drop_unnamed = drop_unnamed
        self.drop_leakage = drop_leakage
        self.drop_constant_features = drop_constant_features
        self.drop_corr = drop_corr
        self.corr_threshold = corr_threshold
        self.log_features = log_features
        self.log_skew_threshold = log_skew_threshold
        self.log_min_unique_values = log_min_unique_values

        self.constant_features_ = []
        self.correlated_features_ = []
        self.log_features_ = []

    def _get_leakage_columns(self):
        leakage_map = {
            'IC50, mM': ['SI'],
            'CC50, mM': ['SI'],
            'SI': ['IC50, mM', 'CC50, mM']
        }
        return leakage_map.get(self.target_column, [])

    def fit(self, X, y=None):
        X_fit = X.copy()

        # Подготавливаем данные для вычисления правил преобразования
        if self.drop_unnamed:
            X_fit = X_fit.drop(columns=['Unnamed: 0'], errors='ignore')

        if self.drop_leakage:
            X_fit = X_fit.drop(columns=self._get_leakage_columns(), errors='ignore')

        # Находим константные признаки
        if self.drop_constant_features:
            nunique = X_fit.nunique()
            self.constant_features_ = nunique[nunique == 1].index.tolist()
        else:
            self.constant_features_ = []

        # Исключаем константные признаки перед анализом корреляций и асимметрии
        X_fit = X_fit.drop(columns=self.constant_features_, errors='ignore')
        # Находим сильно коррелированные признаки
        if self.drop_corr:
            corr_matrix = X_fit.corr().abs()

            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            self.correlated_features_ = [
                col for col in upper.columns
                if any(upper[col] > self.corr_threshold)
            ]
        else:
            self.correlated_features_ = []

        # Исключаем коррелированные признаки перед выбором признаков для логарифмирования
        X_fit = X_fit.drop(columns=self.correlated_features_, errors='ignore')
        # Ищем признаки для логарифмирования
        if self.log_features:
            # исключаем таргеты из анализа
            X_for_log = X_fit.drop(columns=self.targets, errors='ignore')
            nunique = X_for_log.nunique()
            skew = X_for_log.skew()
            min_values = X_for_log.min()

            self.log_features_ = [
                col for col in X_for_log.columns
                if nunique[col] > self.log_min_unique_values
                and skew[col] > self.log_skew_threshold
                and min_values[col] > -1
            ]
        else:
            self.log_features_ = []

        return self

    def transform(self, X):
        X = X.copy()

        # Удаляем служебный столбец
        if self.drop_unnamed:
            X = X.drop(columns=['Unnamed: 0'], errors='ignore')

        # Удаляем leakage-признаки
        if self.drop_leakage:
            X = X.drop(columns=self._get_leakage_columns(), errors='ignore')

        # Удаляем константные признаки
        if self.drop_constant_features and self.constant_features_:
            X = X.drop(columns=self.constant_features_, errors='ignore')

        # Удаляем сильно коррелированные признаки
        if self.drop_corr and self.correlated_features_:
            X = X.drop(columns=self.correlated_features_, errors='ignore')

        # Логарифмируем признаки с сильной правой асимметрией
        if self.log_features and self.log_features_:
            for col in self.log_features_:
                X[col] = np.log1p(X[col])

        return X