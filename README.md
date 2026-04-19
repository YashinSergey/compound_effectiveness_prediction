# Machine Learning Coursework

## Описание проекта

В рамках работы были решены задачи регрессии и классификации для датасета с химическими соединениями.

Рассматривались три целевые переменные:
- IC50
- CC50
- SI (рассчитывается как CC50 / IC50)

---

## Структура проекта

<pre>
.
├── notebooks/
│   ├── eda.ipynb
│   ├── IC50_regression.ipynb
│   ├── CC50_regression.ipynb
│   ├── SI_regression.ipynb
│   ├── IC50_classification.ipynb
│   ├── CC50_classification.ipynb
│   ├── SI_classification_median.ipynb
│   └── SI_classification_more_than_eight.ipynb
│
├── src/
│   ├── dataset_preprocessor.py
│   └── metrics_calculator.py
│
├── data/
│   └── raw/
│       └── classicMLCourseWorkData.xlsx
│
├── requirements.txt
└── README.md
</pre>

---

## Установка

Создание виртуального окружения:

```bash
python -m venv .venv
source .venv/bin/activate
```

pip install -r requirements.txt

### Используемые библиотеки  

 - pandas
 - numpy
 - scikit-learn
 - matplotlib
 - seaborn
 - openpyxl

### Предобработка данных  

 - удаление пропущенных значений
 - удаление константных признаков
 - удаление сильно коррелированных признаков
 - логарифмирование признаков с сильной асимметрией
 - предотвращение утечки данных (leakage)

### Решаемые задачи

Регрессия:  
 - предсказание IC50
 - предсказание CC50
 - предсказание SI

Классификация:  
 - IC50 > медианы
 - CC50 > медианы
 - SI > медианы
 - SI > 8

### Используемые модели

Регрессия:  
 - LinearRegression
 - RandomForestRegressor
 - GradientBoostingRegressor

Классификация:  
 - LogisticRegression
 - RandomForestClassifier
 - GradientBoostingClassifier

### Метрики  

Регрессия:   
 - MAE
 - MSE
 - RMSE
 - R²

Классификация:   
 - Accuracy
 - Precision
 - Recall
 - F1-score
 - ROC-AUC

### Результаты   

Лучшие результаты показаны для задач классификации IC50 и CC50
Для них RandomForestClassifier показывает высокое качество.   
Задачи, связанные с SI, не решаются из-за недостатка информации в признаках
