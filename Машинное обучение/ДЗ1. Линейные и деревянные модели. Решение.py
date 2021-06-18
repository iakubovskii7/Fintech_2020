#!/usr/bin/env python
# coding: utf-8

# # Предсказание уровня заработной платы  
# 
# https://archive.ics.uci.edu/ml/datasets/adult
# 
# В данном ДЗ вам нужно будет предсказывать уровень заработной платы респондента: больше 50к долларов в год или меньше.
# 
# **Описание признаков**:
# 
# 1. Категорийные признаки
# 
# `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. Individual work category
# 
# `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. Individual's highest education degree
# 
# `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. Individual marital status
# 
# `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. Individual's occupation
# 
# `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. Individual's relation in a family
# 
# `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. Race of Individual
# 
# `sex`: Female, Male.
# 
# `native-country`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. Individual's native country
# 
# 2. Количественные признаки
# 
# `age`: continuous. Age of an individual
# 
# `fnlwgt`: final weight, continuous. The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of the US. These are prepared monthly for us by Population Division here at the Census Bureau.
# 
# `capital-gain`: continuous.
# 
# `capital-loss`: continuous.
# 
# `hours-per-week`: continuous. Individual's working hour per week

# In[165]:


import pandas as pd
import os
import warnings 
import re # модуль для регулярных выражений
warnings.filterwarnings("ignore")
os.chdir("/Users/iakubovskii/Machine_Learning/RANEPA/Fintech_2020/Машинное обучение/Данные/ДЗ1/")
adult = pd.read_pickle("adult").dropna()
import numpy as np

# In[177]:


# Список категориальных признаков
cat_cols = ['Workclass', 'Education', 'Marital_Status', 'Occupation',
           'Relationship', 'Race', 'Sex', 'Country']

# Список количественных признаков
quant_cols = list(set(adult.columns) - set(cat_cols))
quant_cols.remove("Target")
# Удалим лишние пробелы перед значениями внутри категориальных столбцов
for cat_col in cat_cols:
    adult[cat_col] = adult[cat_col].str.strip(" ")


# Преобразуем категориальные признаки при помощи дамми-кодирования
new_df = pd.concat([adult.drop(cat_cols, axis=1),
                   pd.get_dummies(adult[cat_cols], drop_first=True)],
                   axis=1)
# Удалим строки с пропусками
new_df = new_df.dropna()
# Исправим неточности в целевой переменной
print(new_df['Target'].value_counts())
new_df['Target'] = new_df['Target'].str.strip(" ")
new_df['Target'] = new_df['Target'].map({">50K.": ">50K",
                                       "<=50K.": "<=50K"}).fillna(new_df['Target'])
print(new_df['Target'].value_counts())


# ##############################################################################################################
# # Задания
# 
# ВНИМАНИЕ!!!
# ВЕЗДЕ, где есть параметр **random_state**, устанавливайте его равным **своему номеру**, иначе у нас могут не совпасть результаты и будет плохо.
# 
# Результаты округляем до 5 знака после запятой. Например, ROC AUC = 0.56156
# 
# Задание *найти оптимальный гиперпараметр* подразумевает 5  фолдовую стратифицированную кросс-валидацию с random_state, равным **вашему номеру**.
# 
# По умолчанию мы используем ВСЕ ПРИЗНАКИ из датасетов.

# In[343]:


students_random_state = {
 'Базуева Мария Дмитриевна': 993,
 'Бориско Данила Ильич': 1956,
 'Братков Герман Сергеевич': 210,
 'Орлан Суван-оол': 211,
 'Валл Федор Викторович': 188,
 'Егорова Анна Сергеевна': 25,
 'Едовина Алина Игоревна': 35,
 'Загарнюк Елизавета Максимовна': 979,
 'Захаров Алексей Сергееивч': 587,
 'Калёнов Алексей Аркадьевич': 1334,
 'Карасева Алина Александровна': 1265,
 'Каширин Егор Михайлович': 1672,
 'Косинов Андрей Вячеславович': 940,
 'Красиков Евгений Владимирович': 601,
 'Кузьмин Никита Кириллович': 452,
 'Монгуш Тенгиз Анатольевич': 668,
 'Мурадян Акоп Араратович': 1155,
 'Наумова Анастасия Юрьевна': 1020,
 'Панчук Александр Сергеевич': 1125,
 'Пашинина Татьяна Викторовна': 268,
 'Пустоваров Артем Андреевич': 187,
 'Роговая Тамара Олеговна': 1472,
 'Селезнев Дмитрий Владимирович': 734,
 'Сидорякин Виталий Дмитриевич': 1554,
 'Филиппов Антон Павлович': 126,
 'Фрольцов Григорий Максимович': 1723,
 'Хамитов Давид Альбертович': 1944,
 'Хомушку Ганна Алексеевна': 582,
 'Царева Мария Сергеевна': 1336}


# In[299]:

def get_all_solutions(fio):
    
    your_number_random_state = students_random_state[fio]
    # your_number_random_state = students_random_state[fio]
    # Для расчета доли, по которой будем делить датасеты
    max_ = max(list(map(lambda x: np.log(x), students_random_state.values())))
    frac_student = np.log(your_number_random_state) / max_
    # your_number_random_state = 188
    # Уникальный датасет для каждого студента
    df_model = new_df.sample(frac=frac_student, random_state=your_number_random_state)


    # Определим целевую переменную 
    X, y = df_model.drop("Target", axis=1), df_model['Target'].map({">50K": 1, "<=50K": 0})


    # In[300]:


    # Разобьем на тренировочную и тестовую (уже уменьшенный датасет для каждого студента)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                       random_state=your_number_random_state)

    # Стандартизируем количественные признаки

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train[quant_cols] = sc.fit_transform(X_train[quant_cols])
    X_test[quant_cols] = sc.transform(X_test[quant_cols])

    # 1. При помощи метода ближайших соседей сделайте прогноз для числа соседей *со всеми объясняющими признаками*, равным 11. Чему равен `Recall` на тестовой выборке?

    final_df = pd.DataFrame(index=[fio], columns=np.arange(1, 11))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import recall_score
    knn_11 = KNeighborsClassifier(n_neighbors=11)
    knn_11.fit(X_train, y_train)
    y_predict = knn_11.predict(X_test)
    task1 = np.round(recall_score(y_test, y_predict), 5)
    final_df.loc[fio, 1] = task1

    # 2. Среди следующих значений коэффициентов регуляризации (без кросс-валидации) выберите тот, для которого `ROC AUC` score на тестовой выборке для логистической регрессии *со всеми факторами* будет минимальным. В ответе укажите значение минимального ROC AUC

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    def get_roc_auc_lr(C):
        lr = LogisticRegression(C=C, random_state=your_number_random_state)
        lr.fit(X_train, y_train)
        y_predict_proba = lr.predict_proba(X_test)[:, 1]
        roc = np.round(roc_auc_score(y_test, y_predict_proba), 5)
        return roc
    C_list = [0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.9, 1]
    roc_auc_list = list(map(get_roc_auc_lr, C_list))
    task2 = min(roc_auc_list)
    final_df.loc[fio, 2] = task2

    # 3. Чему равен `Precision` для дерева решений с параметрами по умолчанию (реализация sklearn, random_state соответствует вашему номеру)?

    # In[303]:


    from sklearn.metrics import precision_score
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=your_number_random_state)
    tree.fit(X_train, y_train)
    y_predict = tree.predict(X_test)
    task3 = np.round(precision_score(y_test, y_predict), 5)
    final_df.loc[fio, 3] = task3

    # 4. Удалите признаки, для которых коэффициенты в LASSO регрессии равны нулю (на тренировочной выборке) и сделайте прогноз на тестовой выборке при помощи логистической регрессии с коэффициентов регуляризации C=0.5. В ответ запишите полученный `Recall`.
    # 
    # LASSO регрессия для логистической регрессии юзается вот так:
    # 
    # `LogisticRegression(penalty='l1', solver='liblinear', random_state=your_number_random_state)`

    # In[304]:


    from sklearn.metrics import recall_score
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=your_number_random_state)
    lasso.fit(X_train, y_train)
    lasso_coefs = lasso.coef_
    lasso_coefs_zero = dict(zip(X_train.columns, lasso_coefs.flatten()))
    lasso_coefs_zero_dict = {key: value for key, value in lasso_coefs_zero.items() if value == 0}

    X_train_postlasso = X_train.drop(lasso_coefs_zero_dict.keys(), axis=1)
    X_test_postlasso = X_test.drop(lasso_coefs_zero_dict.keys(), axis=1)

    lr = LogisticRegression(C=0.5)
    lr.fit(X_train_postlasso, y_train)
    y_predict_postlasso = lr.predict(X_test_postlasso)
    task4 = np.round(recall_score(y_test, y_predict_postlasso), 5)
    final_df.loc[fio, 4] = task4


    # 5. При помощи кросс-валидации найдите оптимальный параметр max_depth на границах [1, 50] для дерева решений.
    # В ответ запишите ROC AUC для оптимальной `максимальной глубины дерева`.

    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV
    tree = DecisionTreeClassifier(random_state=your_number_random_state)
    parameters = {'max_depth': np.linspace(1, 50, 50)}
    gcv = GridSearchCV(tree, param_grid=parameters, cv=kf, scoring='roc_auc')
    gcv.fit(X_train, y_train)
    task5 = np.round(roc_auc_score(y_test, gcv.predict_proba(X_test)[:, 1]), 5)
    final_df.loc[fio, 5] = task5


    # 6. Обучите модель бэггинга на тренировочном наборе данных для всех признаков для количества деревьев, равного 200.
    # Чему равна `OOB ошибка` на тренировочной выборке?

    # In[306]:


    from sklearn.ensemble import BaggingClassifier
    bdt = BaggingClassifier(n_estimators=200, random_state=your_number_random_state,
                           oob_score=True)
    bdt.fit(X_train, y_train)
    task6 = np.round(bdt.oob_score_, 5)
    final_df.loc[fio, 6] = task6


    # 7. Обучите на тренировочном датасете модель бэггинга с числом деревьев, равным 300, и сделайте прогноз на тестовой выборке. Чему равна `F1 мера`?

    # In[307]:


    from sklearn.metrics import f1_score
    bdt = BaggingClassifier(n_estimators=300, random_state=your_number_random_state,
                           oob_score=True)
    bdt.fit(X_train, y_train)
    task7 = np.round(f1_score(y_test, bdt.predict(X_test)), 5)
    final_df.loc[fio, 7] = task7


    # 8. На тренировочном датасете выберите оптимальные гиперпараметры (через `Accuracy`) для дерева решений из такого множества:
    # 
    #  - max_depth = значения от 1 до 20 включительно
    #  - max_features = значения от 15 до 35 включительно
    #  
    # Чему равен `Accuracy` на тестовом наборе данных с таким набором гиперпараметров?

    # In[308]:


    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    tree = DecisionTreeClassifier(random_state=your_number_random_state)
    parameters = {'max_depth': np.arange(1, 21),
                 'max_features': np.arange(15, 36)}
    gcv = GridSearchCV(tree, param_grid=parameters, cv=kf, scoring='accuracy')
    gcv.fit(X_train, y_train)
    task8 = np.round(accuracy_score(y_test, gcv.predict(X_test)), 5)
    final_df.loc[fio, 8] = task8


# 9. Подберите оптимальное значение  max_features = [3, 5, 10, 15, 25, 30, 35, 40] для бэггинга,
    # где в качестве базового алгоритма будет логистическая регрессия с C=0.5.
    # Параметр n_estimators настройте равным 25. Оптимизируемся на `Recall`.
    # Чему равен `Precision` на тестовой выборке с лучшим гиперпараметров из тренировочной выборки?

# In[309]:

    lr = LogisticRegression(C=0.5)
    bagdt = BaggingClassifier(base_estimator=lr, n_estimators=25,
                              random_state=your_number_random_state)
    parameters = {'max_features':  [3, 5, 10, 15, 25, 30, 35, 40] }
    gcv = GridSearchCV(bagdt, param_grid=parameters, cv=kf, scoring='recall')
    gcv.fit(X_train, y_train)
    task9 = np.round(precision_score(y_test, gcv.predict(X_test)), 5)
    final_df.loc[fio, 9] = task9
    os.chdir("/Users/iakubovskii/Machine_Learning/RANEPA/Fintech_2020/HW1_ML_reslts")
    final_df.to_pickle(fio)
    return
for student in students_random_state.keys():
    get_all_solutions(student)
    print(student)

# 10. Задание на бутстрап для линейной регрессии. При помощи метода бутстрапированных остатков регрессии
# вычислите 95% доверительный интервал для коэффициента перед переменной *Age*, где зависимая переменная *fnlwgt*.
# Регрессия с константой! Также вычислите 95% доверительный интервал при помощи стандартного метода в линейной регрессии при допущении о выполнении условий ГМ.
# В ответ запишите `отношение длины бутстрапированного доверительного интервала к длине стандартного доверительного интервала`.




# Проверка
#
# df = []
# for file in glob.glob("*"):
#     df.append(pd.read_pickle(file))
#
# df = pd.concat(df)
# df.sort_index().to_excel("Student_results_correct.xlsx")
# df.columns = df.columns.astype(str)
# answers = pd.read_clipboard()
# answers = answers.applymap(lambda x: x.replace(",", ".") if type(x) == str else x).astype(float)
# answers.set_index(['ФИО'], inplace=True)
# answers.sort_index().to_excel("Student_results.xlsx")
# (df.sort_index() == answers.sort_index()).to_excel("Student_results_correct_true_false.xlsx")




