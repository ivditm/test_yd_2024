from dataclasses import dataclass
from IPython.display import display
import ipywidgets as widgets
import itertools
import logging
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import plotly.express as px
import re
from scipy import stats as sst
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from typing import Optional, Union
import warnings


register_matplotlib_converters()
warnings.filterwarnings('ignore')


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)


def get_full_information(data: pd.DataFrame) -> None:
    """
    Функция дает нам полное
    представление о данных
    """
    logging.info('первое представление о данных')
    for _ in [data.head(), data.describe()]:
        display(_)
        print("*" * 100)
    data.info()
    data.hist(figsize=(15, 10))
    plt.show()


def find_nan(data: pd.DataFrame) -> dict:
    """
    Функция считает по каждой колонке, сколько в ней пропусков,
    если пропусков 0, то с колонкой все хорошо
    и ее мы трогать не будем
    """
    return {column: data[column].isna().sum() / len(data)
            for column in data.columns
            if data[column].isna().sum() != 0}


def test_find_nan():
    df1 = pd.DataFrame({'A': [1, 2, None, 4], 'B': [
                       None, 6, 7, 8], 'C': [9, 10, 11, None]})
    assert find_nan(df1) == {'A': 0.25, 'B': 0.25, 'C': 0.25}
    df2 = pd.DataFrame(
        {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]})
    assert find_nan(df2) == {}
    print('OK')


def drop_nan_less_five_percent(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция удаляет строки по колонкам,
    если кол-во пропусков в них составляло
    менее 5 процентов
    """
    columns_with_nan = find_nan(data)
    for column, percent in columns_with_nan.items():
        if percent < .05:
            data = data[data[column].notna()]
    return data


def test_drop_nan_less_five_percent():
    df1 = pd.DataFrame(
        {'a': [None] + [i for i in range(20)], 'b': [i for i in range(21)]})
    assert drop_nan_less_five_percent(df1).equals(
        pd.DataFrame(data={'a': [float(i) for i in range(20)],
                           'b': [i for i in range(1, 21)]},
                     index=[i for i in range(1, 21)]))
    df2 = pd.DataFrame({'A': [None, None, None, None], 'B': [
                       None, None, None, None], 'C': [9, 10, 11, 12]})
    assert drop_nan_less_five_percent(df2).equals(df2)
    df3 = pd.DataFrame(
        {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]})
    assert drop_nan_less_five_percent(df3).equals(df3)
    print('OK')


def works_dupblicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция работает с дубликатами
    """
    for column in data.columns:
        if data.dtypes[column] == 'object' and isinstance(
                data[column][0], str):
            data[column] = data[column].str.lower()
    if data.duplicated().sum() > 0:
        logging.info('нашли дубликаты и удалили их')
        privios_data = len(data)
        data = data.drop_duplicates()
        logging.info('потеря данных в %:')
        logging.info(round((1 - (len(data) / privios_data)) * 100, 2))
    else:
        logging.info('дубликатов не найдено')
    return data


def test_works_duplicates():
    df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'alice'],
                       'age': [25, 30, 35, 25],
                       'date': ['2020-01-01', '2020-02-02',
                                '2020-03-03', '2020-01-01']})
    df_result = works_dupblicates(df)
    assert len(df_result) == 3
    assert not df_result.duplicated().any()
    print('ОК')


def change_type_to_date(data: pd.DataFrame) -> pd.DataFrame:
    """
    Меняет формат на время
    """
    pattern = r'\d{4}-\d{2}-\d{2}'
    for column in data.columns:
        match = re.match(pattern, str(data[column][0]))
        if match:
            data[column] = pd.to_datetime(data[column])
            logging.info('поменяли на временной формат колоку')
            logging.info(column)
    return data


def test_change_type_to_date():
    df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'alice'],
                       'age': [25, 30, 35, 40],
                       'date': ['2020-01-01', '2020-02-02',
                                '2020-03-03', '2020-04-04']})
    df_result = change_type_to_date(df)
    assert isinstance(df_result['date'][0], pd.Timestamp)
    assert not isinstance(df_result['name'][0], pd.Timestamp)
    print('OK')


def testing():
    test_find_nan()
    test_drop_nan_less_five_percent()
    test_works_duplicates()
    test_change_type_to_date()


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция содержит основную логику
    первичной обработки данных
    """
    try:
        get_full_information(data)
        if len(find_nan(data)) > 0:
            data = drop_nan_less_five_percent(data)
            display(
                pd.DataFrame(
                    data={'columns': find_nan(data).keys(),
                          'nan_percent': find_nan(
                              data).values()}
                ).style.background_gradient('coolwarm'))
        else:
            logging.info('пропусков не нашел')
        data = change_type_to_date(data)
        data = works_dupblicates(data)
        return data
    except BaseException as error_message:
        return (error_message)


class Regressor:
    """Класс линейной регрессии"""

    def __init__(self, data, predictor, target, text):
        self.data = data
        self.predictor = predictor
        self.target = target
        self.text = text
        if self.data[self.predictor].isna().sum() > 0:
            self.data = self.data[self.data[self.predictor].notna()]
            logging.info('были исключены пропуски')
        if self.data[self.target].isna().sum() > 0:
            self.data = self.data[self.data[self.target].notna()]
            logging.info('были исключены пропуски')

    def analysis(self):
        if (self.predictor in self.data.columns and
                self.target in self.data.columns):
            regression = sm.OLS(
                self.data[self.predictor], sm.add_constant(
                    self.data[self.target]))
            result = regression.fit()
            return {
                'params': result.params,
                'bse': result.bse,
                'pvalues': result.pvalues,
                't_test_const': result.t_test([1, 0]),
                't_test_regr': result.t_test([0, 1]),
                'f_test': result.f_test(np.identity(2)),
                'summary': result.summary()
            }
        else:
            raise ValueError(
                'Столбцы predictor и/или target не найдены в датафрейме')

    def get_plot(self, savefig=None):
        if (self.predictor in self.data.columns and
                self.target in self.data.columns):
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            plt.figure(figsize=(10, 6))
            plot = sns.jointplot(
                x=self.predictor,
                y=self.target,
                data=self.data,
                kind='reg')
            plot.fig.suptitle(self.text[0], x=1.15, y=1.05, fontsize=16)
            plot.set_axis_labels(self.text[1], self.text[2], fontsize=12)
            if savefig is not None:
                plt.savefig(savefig)
            plt.show()
        else:
            raise ValueError(
                'Столбцы predictor и/или target не найдены в датафрейме')

    def get_table(self):
        display(self.analysis()['summary'])


class ReportReg:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns
        self.methods = {'table': 'get_table', 'plot': 'get_plot'}
        self.graphs = ['table', 'plot']

    def get_report(self):
        @widgets.interact(graphs=self.graphs,
                          predictors=self.columns, targets=self.columns)
        def get_reg_anal(graphs, predictors, targets):
            reg = Regressor(
                self.data, predictors, targets, [
                    f'Взаимосвязь {predictors} и {targets}',
                    predictors, targets])
            getattr(reg, self.methods[graphs])()


class Cluster:
    """Класс для кластеризации данных"""

    def __init__(self, columns: list[str],
                 data: pd.DataFrame, n_max: int = 5) -> None:
        """Инициализация."""
        self.data = data
        self.columns = columns
        self.x: pd.DataFrame = self.data[self.columns]
        self.scaler = StandardScaler()
        self.scaler.fit(self.x)
        self.x_st = self.scaler.transform(self.x)
        self.n_max = n_max
        self.K = range(2, self.n_max)
        self.score: list = []

    def get_dendrogram(self, percent: float = 0.05) -> None:
        """Строим дендограмму для оценки оптимальноко кол-ва кластеров."""
        df: pd.DataFrame = pd.DataFrame(
            self.x_st, columns=self.columns).sample(
            frac=percent)
        linked = linkage(df, method='ward')
        plt.figure(figsize=(15, 10))
        dendrogram(linked, orientation='top')
        plt.title('Hierarchical clustering')
        plt.show()

    @property
    def eblow_score(self) -> list[Union[int, float]]:
        """Зашумление данных по методу локтя."""
        distortions: list = []
        for k in self.K:
            km = KMeans(n_clusters=k)
            km.fit(self.x_st)
            distortions.append(km.inertia_)
        return distortions

    def get_eblow(self) -> None:
        """Визуализируем метод локтя."""
        plt.figure(figsize=(16, 8))
        plt.plot(self.K, self.eblow_score, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    @property
    def silhouette_scores(self) -> list[Union[int, float]]:
        """
        Коэффициенты по методу силуэта от 0-1
        1 - высокая точность модели
        0 - отсутствие точности модели.
        """
        for k in self.K:
            km = KMeans(n_clusters=k)
            labels = km.fit_predict(self.x_st)
            self.score.append(silhouette_score(self.x_st, labels))
        return self.score

    def get_silhouette_score(self) -> None:
        """Визуализация метода силуэта."""
        plt.figure(figsize=(16, 8))
        if len(self.score) == 0:
            plt.plot(self.K, self.silhouette_scores, 'bx-')
        else:
            plt.plot(self.K, self.score, 'bx-')
        plt.xlabel('k')
        plt.ylabel('silhouette_score')
        plt.title('The Silhouette Method showing the optimal k')
        plt.show()

    @property
    def optimal_k(self):
        """
        Возвращает оптимальное кол-во ксластеров
        по методу силуэта как более точного метода.
        """
        if len(self.score) == 0:
            return sorted(list(zip(self.K, self.silhouette_scores)),
                          key=lambda x: x[1], reverse=True)[0][0]
        else:
            return sorted(list(zip(self.K, self.score)),
                          key=lambda x: x[1], reverse=True)[0][0]

    def cluster(self, k: Optional[int] = None) -> list[float]:
        """Возвращает предсказания для модельки ближайших соседей"""
        if k is None:
            km = KMeans(n_clusters=self.optimal_k, random_state=0)
        else:
            km = KMeans(n_clusters=k, random_state=0)
        return km.fit_predict(self.x_st)

    def show_clusters_on_plot(self, x_name, y_name,
                              k, cluster_name='cluster') -> None:
        """Отрисовывает распределение признаков по кластеру"""
        self.data['cluster'] = self.cluster(k)
        plt.figure(figsize=(5, 5))
        sns.scatterplot(x=self.data[x_name], y=self.data[y_name],
                        hue=self.data[cluster_name], palette='Paired')
        plt.title('{} vs {}'.format(x_name, y_name))
        plt.show()

    def plot(self) -> None:
        """Рисуем все графики"""
        col_pairs: list = list(itertools.combinations(self.columns, 2))
        for pair in col_pairs:
            self.show_clusters_on_plot(pair[0], pair[1], 'cluster')


@dataclass
class ClusterReport:
    """Класс печатает отчет по модели Cluster."""

    cluster_object: Cluster
    attrs_score = {'eblow': 'get_eblow',
                   'silhouette': 'get_silhouette_score',
                   'dendrogram': 'get_dendrogram'}

    def print_report(self) -> None:
        """Отрисовывает распределение фичей по кластерам."""
        widgets.interact(
            self.cluster_object.show_clusters_on_plot,
            x_name=widgets.Dropdown(
                options=self.cluster_object.columns, description="Select x"),
            y_name=widgets.Dropdown(
                options=self.cluster_object.columns, description="Select y"),
            k=widgets.IntSlider(
                min=2,
                max=100,
                value=self.cluster_object.optimal_k,
                step=1,
                description="k"))

        @widgets.interact(types_score=self.attrs_score.keys())
        def plot_score(types_score):
            getattr(self.cluster_object, self.attrs_score[types_score])()


class Classifier:
    """Класс классификации"""

    def __init__(self, cluster: Cluster) -> None:
        """Инициализация, все необходимое возьмем из Cluster"""
        self.cluster = cluster
        self.x = self.cluster.x_st
        self.y = self.cluster.cluster()
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = train_test_split(self.x,
                                                       self.y,
                                                       test_size=0.2,
                                                       random_state=0)

    def logistic_regression(self) -> tuple[list[Union[int, float]],
                                           Optional[list[Union[int, float]]],
                                           pd.DataFrame]:
        """Сюда положим логистическую регрессию."""
        model = LogisticRegression(solver='liblinear')
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)
        if self.cluster.optimal_k == 2:
            probabilities: list = model.predict_proba(self.x_test)[:, 1]
        else:
            probabilities = None
        if self.cluster.optimal_k == 2:
            coeff_cluster = {
                'coeff': model.coef_[i] for i in range(
                    self.cluster.optimal_k - 1)}
        else:
            coeff_cluster = {f'coeff_cluster_{i}': model.coef_[
                i] for i in range(self.cluster.optimal_k)}
        coeff_cluster['feature'] = self.cluster.columns
        coeff = pd.DataFrame(data=coeff_cluster)
        return predictions, probabilities, coeff

    def tree_classifier(self) -> tuple[list[Union[int, float]],
                                       Optional[list[Union[int, float]]],
                                       pd.DataFrame]:
        """Сдесь модель решающего дерева."""
        tree_model = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
        tree_model.fit(self.x_train, self.y_train)
        tree_predictions = tree_model.predict(self.x_test)
        if self.cluster.optimal_k == 2:
            tree_probabilities = tree_model.predict_proba(self.x_test)[:, 1]
        else:
            tree_probabilities = None
        return tree_predictions, tree_probabilities, tree_model

    def plot_model(self, model) -> None:
        """Отрисовываем модельки."""
        if model == 'reg':
            _, _, coeff = self.logistic_regression()
            display(coeff.style.background_gradient('coolwarm'))
        else:
            _, _, tree_model = self.tree_classifier()
            plt.figure(figsize=(25, 20))
            tree.plot_tree(tree_model,
                           filled=True,
                           feature_names=self.cluster.columns,
                           class_names=[f'cluster_{i}' for i in range(
                               self.cluster.optimal_k)])
            plt.show()

    def metrics(self, model: str) -> pd.DataFrame:
        """Посмотри на качество моделек."""
        if model == 'reg':
            predictions, probabilities, _ = self.logistic_regression()
        else:
            predictions, probabilities, _ = self.tree_classifier()
        if self.cluster.optimal_k == 2:
            df = pd.DataFrame({'metric': ['Accuracy', 'Precision',
                                          'Recall', 'F1', 'ROC_AUC'],
                               'coeff': [accuracy_score(self.y_test,
                                                        predictions),
                                         precision_score(
                                             self.y_test, predictions),
                                         recall_score(
                                             self.y_test, predictions),
                                         f1_score(self.y_test, predictions),
                                         roc_auc_score(
                                             self.y_test, probabilities)
                                         ]}).style.background_gradient(
                                             'coolwarm')
        else:
            df = pd.DataFrame({'metric': ['Accuracy', 'Precision',
                                          'Recall', 'F1'],
                               'coeff': [accuracy_score(self.y_test,
                                                        predictions),
                                         precision_score(
                                   self.y_test, predictions, average='micro'),
                recall_score(
                                   self.y_test, predictions, average='micro'),
                f1_score(
                                   self.y_test, predictions, average='micro')
            ]}).style.background_gradient('coolwarm')

        return df


class BoxMath:
    """Parents math class for anomalies"""

    CONSTANT: Optional[Union[int, float]] = None

    def __init__(self, df: pd.DataFrame, column: str) -> None:
        """
        инициализация
        """
        self.df = df
        self.column = column
        if self.column not in set(self.df.columns):
            raise ValueError('ошибка при формировании класса')
        if self.df[self.column].isna().sum() > 0:
            self.df = self.df[self.df[self.column].notna()]
            logging.info('были исключены пропуски')

    @property
    def metrics(self) -> dict[str: Union[int, float]]:
        """
        Рассчитываем показатели:
        необходмые для вычислений
        """
        pass

    @property
    def max_not_anomal(self) -> Union[int, float]:
        """
        Вычисляет верхнюю границу усов
        """
        pass

    @property
    def min_not_anomal(self) -> Union[int, float]:
        """
        Вычисляет нижнюю границу усов
        """
        pass

    def is_distribution_normal(self, alpha: float = 0.05) -> bool:
        """определяет нормальное ли распределение"""
        if sst.shapiro(self.df[self.column])[1] < alpha:
            return False
        return True

    @property
    def anomalies(self) -> pd.Series:
        """
        аномальные значения
        """
        return (self.df[
            (self.df[self.column] < self.min_not_anomal)
            |
            (self.df[self.column] > self.max_not_anomal)
        ][self.column])

    @property
    def anomalies_indexs(self) -> pd.Series:
        """Возращает индексы аномальных строк"""
        return self.anomalies.index


class BoxIQR(BoxMath):
    """
    Класс содержит анализ по выбросам
    """

    CONSTANT: float = 1.5

    @property
    def metrics(self) -> dict[str: Union[int, float]]:
        """
        Рассчитываем показатели:
        -второй квартиль
        -третий квартиль
        -межквартильный размах
        """
        q75, q25 = np.percentile(self.df[self.column], [75, 25])
        metrics: dict[str: float] = {'iqr': q75 - q25,
                                     'q75': q75,
                                     'q25': q25}
        return metrics

    @property
    def max_not_anomal(self) -> Union[int, float]:
        """
        Вычисляет верхнюю границу усов
        """
        max_not_anomal: float = self.metrics['q75'] + \
            self.CONSTANT * self.metrics['iqr']
        if (max_not_anomal > self.df[self.column].max()):
            return self.df[self.column].max()
        else:
            return max_not_anomal

    @property
    def min_not_anomal(self) -> Union[int, float]:
        """
        Вычисляет нижнюю границу усов
        """
        min_not_anomal: float = self.metrics['q25'] - \
            self.CONSTANT * self.metrics['iqr']
        if min_not_anomal < self.df[self.column].min():
            return self.df[self.column].min()
        else:
            return min_not_anomal

    @property
    def min_p_anom(self) -> Union[int, float]:
        """минималльный положительный выброс по доходности"""
        return self.anomalies[self.anomalies > 0].min()


class BoxSTD(BoxMath):
    """Считает выбросы в нормальном распределении"""
    CONSTANT: int = 3

    @property
    def metrics(self) -> dict[str:Union[int, float]]:
        """возращает стандартное отклонение и среднее"""
        return {'std': self.df[self.column].std(),
                'mean': self.df[self.column].mean()}

    @property
    def max_not_anomal(self) -> Union[int, float]:
        """Возвращает верхнюю границу"""
        max_not_anomal: float = self.metrics['mean'] + \
            self.CONSTANT * self.metrics['std']
        if max_not_anomal > self.df[self.column].max():
            return self.df[self.column].max()
        else:
            return max_not_anomal

    @property
    def min_not_anomal(self) -> Union[int, float]:
        """Возвращает нижнюю границу"""
        min_not_anomal: float = self.metrics['mean'] - \
            self.CONSTANT * self.metrics['std']
        if min_not_anomal < self.df[self.column].min():
            return self.df[self.column].min()
        else:
            return min_not_anomal


class BoxVisualisation:
    """
    Строим графики к выбросам
    """

    def __init__(self, math_object: BoxMath,
                 level: Optional[str] = None) -> None:
        """инициализация"""
        self.math_object = math_object
        self.level = level
        if self.level is not None and isinstance(self.level, str):
            self.rank = self.math_object.df.groupby(
                self.level)[
                self.math_object.column].median().fillna(0).sort_values()[
                ::-1].index
        elif self.level is not None and not isinstance(self.level, str):
            raise TypeError('suplements parametr must be str')
        elif (self.level is not None and
                self.level not in set(self.math_object.df.columns)):
            raise TypeError('column not found - 404')

    def represent_violinplot(self, plots_size: tuple[int] = (15, 15)) -> None:
        sns.set_palette('dark')
        plt.figure(figsize=plots_size)
        plt.title("Violin plot", loc="left")
        if self.level is not None:
            sns.violinplot(x=self.math_object.df[self.math_object.column],
                           y=self.math_object.df[self.level])
        else:
            sns.violinplot(x=self.math_object.df[self.math_object.column])
        plt.show()

    def represent_box(self, plots_size: tuple[int] = (15, 15)) -> None:
        """выводит бокс-плот"""
        sns.set_palette('dark')
        plt.figure(figsize=plots_size)
        plt.title('Бокс-плот')
        if self.level is not None:
            sns.boxplot(x=self.math_object.df[self.math_object.column],
                        y=self.math_object.df[self.level],
                        order=self.rank, color='mediumpurple')
        else:
            sns.boxplot(x=self.math_object.df[self.math_object.column])
        plt.show()

    def represent_in_detail(self, plots_size: tuple[int] = (15, 15)) -> None:
        """приближенный бокс-плот"""

        sns.set_palette('dark')
        plt.figure(figsize=plots_size)
        plt.title('Приближенный бокс-плот')
        if self.level is not None:
            sns.boxplot(x=self.math_object.df[self.math_object.column],
                        y=self.math_object.df[self.level],
                        order=self.rank, color='mediumpurple'
                        ).set_xlim([self.math_object.min_not_anomal,
                                    self.math_object.max_not_anomal])
        else:
            sns.boxplot(
                x=self.math_object.df[self.math_object.column]
            ).set_xlim([self.math_object.min_not_anomal,
                        self.math_object.max_not_anomal])
        plt.show()

    def represent_scatter(self, plots_size: tuple[int] = (20, 20)) -> None:
        """"выводит точечный график"""
        sns.set_palette('dark')
        plt.figure(figsize=plots_size)
        plt.title('Точечный график')
        if self.level is not None:
            sns.scatterplot(x=self.math_object.df.index,
                            y=self.math_object.df[self.math_object.column],
                            hue=self.math_object.df[self.level])
        else:
            sns.scatterplot(x=self.math_object.df.index,
                            y=self.math_object.df[self.math_object.column])
        plt.axhline(y=self.math_object.max_not_anomal,
                    color='red',
                    linestyle='dotted',
                    label='Максимальное значение без выбросов')
        plt.axhline(y=self.math_object.min_not_anomal,
                    color='red',
                    linestyle='dotted',
                    label='Минимальное значение без выбросов')
        plt.axhline(y=self.math_object.df[self.math_object.column].median(),
                    color='green',
                    linestyle='--',
                    label='Медиана')
        plt.axhline(y=self.math_object.df[self.math_object.column].mean(),
                    color='pink',
                    linestyle='--',
                    label='Среднее значение')
        plt.legend()
        plt.show()

    def represent_histplot(self, plots_size: tuple = (15, 15)) -> None:
        """распределение с выбросами"""
        sns.set_palette('dark')
        plt.figure(figsize=plots_size)
        sns.kdeplot(self.math_object.df[self.math_object.column])
        plt.axvline(x=self.math_object.df[self.math_object.column].mean(
        ), color='green', linestyle='--', label='Среднее значение')
        plt.axvline(
            x=self.math_object.min_not_anomal,
            color='orange',
            linestyle=':',
            label='Минимальное значение без выбросов')
        plt.axvline(
            x=self.math_object.max_not_anomal,
            color='orange',
            linestyle=':',
            label='Максимальное значение без выбросов')
        plt.title('гистограмма')
        plt.legend()
        plt.show()


@dataclass
class PrinterReport:
    """Выводим отчет"""

    math_object: BoxMath
    visual_object: BoxVisualisation
    visualisations = {'boxplot': 'represent_box',
                      'detail_box_plot': 'represent_in_detail',
                      'scatterplot': 'represent_scatter',
                      'violinplot': 'represent_violinplot',
                      'histplot': 'represent_histplot'}

    def print_result(self) -> None:

        @widgets.interact(graphs=self.visualisations.keys())
        def plot_graphs(graphs):
            getattr(self.visual_object, self.visualisations[graphs])()
            plt.show()

        for metric, value in self.math_object.metrics.items():
            print(f'{metric} - {value}')
        print(f"минимальное неаномальное значение "
              f"{self.math_object.min_not_anomal} \n"
              f"максимальное неаномальное значение "
              f"{self.math_object.max_not_anomal}")
        print('выбросов % - ', round((len(self.math_object.anomalies) /
              len(self.math_object.df)) * 100, 2))

        button = widgets.Button(description="Узнать больше!",
                                button_style='info')

        def on_button_clicked(b):
            display(pd.DataFrame({'anomalies': self.math_object.anomalies,
                                  'index': self.math_object.anomalies_indexs}))
        button.on_click(on_button_clicked)
        display(button)


def is_distribution_normal(data: pd.Series, alpha: float = .05) -> bool:
    """определяет нормальное ли распределение"""
    if sst.shapiro(data)[1] < alpha:
        return False
    return True


def calc_hypoteses(data_1, data_2,
                   alpha=0.05, equal_var=False,
                   bonferonnie=None):
    if bonferonnie is not None:
        alpha = alpha/bonferonnie
    results = sst.ttest_ind(data_1, data_2, equal_var=equal_var)
    if results.pvalue < alpha:
        return (f'Вывод: Отвергаем нулевую гипотезу о'
                f' статистической незначимости при уровни значимости: '
                f'{round(results.pvalue, 5)}')
    else:
        return (f'Вывод: Не получилось отвергнуть нулевую гипотезу'
                f' о статистической незначимости при уровни значимости: '
                f'{round(results.pvalue, 5)}')


def anal_hyp(data_1, data_2, alpha=0.05, bonferonnie=None):
    if is_distribution_normal(data_1) and is_distribution_normal(data_2):
        calc_hypoteses(data_1, data_2, equal_var=True,
                       alpha=alpha, bonferonnie=bonferonnie)
    calc_hypoteses(data_1, data_2, alpha=alpha, bonferonnie=bonferonnie)


def make_full_analyses_anomalies(
        data: pd.DataFrame, column: str, factor: Optional[str] = None
        ) -> None:
    if is_distribution_normal(data[column]):
        math: BoxMath = BoxSTD(data, column)
    else:
        math: BoxMath = BoxIQR(data, column)
    visualisation: BoxVisualisation = BoxVisualisation(math, factor)
    report: PrinterReport = PrinterReport(math, visualisation)
    report.print_result()


class StatistiqueSeries:
    def __init__(self, values: pd.Series) -> None:
        self.values = pd.Series(values).sort_values()
        self.counts = self.values.value_counts()
        self.stat_table = pd.DataFrame(
            {'x': self.counts.index, 'n': self.counts.values})
        num_bins = int(np.sqrt(len(self.values)))
        self.stat_table['x_interval'] = pd.cut(
            self.stat_table['x'], bins=num_bins)
        self.stat_table = self.stat_table.groupby(
            'x_interval', as_index=False).agg({'n': 'sum'})
        self.stat_table['start'] = self.stat_table['x_interval'].apply(
            lambda x: x.left)
        self.stat_table['end'] = self.stat_table['x_interval'].apply(
            lambda x: x.right)
        self.stat_table['center'] = self.stat_table['x_interval'].apply(
            lambda x: x.mid)
        self.stat_table.drop(columns='x_interval', inplace=True)
        self.stat_table[['start', 'end', 'center']] = self.stat_table[[
            'start', 'end', 'center']].astype(str).astype(float)
        self.stat_table.loc[0, 'start'] = self.values.min()
        self.stat_table['f'] = self.stat_table['n'].apply(
            lambda x: x / self.stat_table['n'].sum())
        self.stat_table['F'] = self.stat_table['f'].cumsum()
        self.stat_table['n*x'] = self.stat_table.apply(
            lambda row: row['center'] * row['n'], axis=1)
        self.stat_table['n*x/sum n*x'] = self.stat_table.apply(
            lambda row:
            row['center'] * row['n'] / self.stat_table['n*x'].sum(),
            axis=1)
        self.stat_table['G'] = self.stat_table['n*x/sum n*x'].cumsum()

    @property
    def moyenne(self) -> float:
        """среднее"""
        return 1 / self.stat_table['n'].sum() * \
            (self.stat_table['center'] * self.stat_table['n']).sum()

    @property
    def var(self) -> float:
        """дисперсия"""
        return 1 / self.stat_table['n'].sum() * (
            self.stat_table['center'] ** 2 * self.stat_table['n']
            ).sum() - self.moyenne ** 2

    @property
    def ecart_type(self) -> float:
        """среднеквадратическое отклонение"""
        return sqrt(self.var)

    @property
    def coef_of_var(self) -> float:
        """коэффициент вариации"""
        return self.ecart_type / self.moyenne

    def quartile(self, value: float) -> float:
        """квартили"""
        row = self.stat_table.loc[self.stat_table['F'] >= value].iloc[0]
        if row.name == 0 or self.stat_table.loc[row.name - 1, 'F'] is None:
            return row['start'] + \
                ((row['end'] - row['start']) / row['f']) * (value)
        else:
            return row['start'] + ((row['end'] - row['start']) / row['f']) * \
                (value - self.stat_table.loc[row.name - 1, 'F'])

    @property
    def medianne(self) -> float:
        """медианна"""
        return self.quartile(0.5)

    @property
    def mode(self) -> list[float, float]:
        """модальный интервал"""
        row = self.stat_table.loc[self.stat_table['n'].max()].iloc[0]
        return [row['start'], row['end']]

    @property
    def courbe_cumulative(self):
        """комулята"""
        X = []
        for st, end in zip(self.stat_table['start'], self.stat_table['end']):
            if st not in X:
                X.append(st)
            if end not in X:
                X.append(end)
        Y = [0, *self.stat_table['F']]
        plt.plot(X, Y)
        plt.show()

    @property
    def gini_plot(self) -> float:
        """кривая Лоренца и коэфф. Джини"""
        Y = [0, *self.stat_table['F']]
        G = [0, *self.stat_table['G']]
        plt.plot(Y, G, label='Lorenz Curve')
        plt.plot(G, G, label='Line of Equality')
        plt.fill_between(
            Y,
            G,
            G,
            color='skyblue',
            alpha=0.5,
            label='Gini coeff')
        plt.fill_between(
            Y,
            G,
            Y,
            color='lightgreen',
            alpha=0.5,
            label='Gini coeff')
        plt.legend(loc='upper left')
        plt.show()
        gini_coefficient = 2 * (0.5 - np.trapz(G, Y))
        return gini_coefficient

    def moment_centre(self, ordre: int) -> float:
        """центральный момент n-ого порядка"""
        return 1 / self.stat_table['n'].sum() * (
            self.stat_table['n'] * (self.stat_table['center']
                                    - self.moyenne) ** ordre).sum()

    def moment_centre_a(self, ordre: int, value: int) -> float:
        """центральный момент n-ого порядка к значению a"""
        return 1 / self.stat_table['n'].sum() * (
            self.stat_table['n'] * (
                self.stat_table['center'] - value) ** ordre).sum()

    def simple_ordre(self, ordre: int) -> float:
        """простой момент"""
        return 1 / self.stat_table['n'].sum() * (
            self.stat_table['n'] * self.stat_table['center'] ** ordre).sum()

    @property
    def yule(self):
        return (
            ((self.quartile(.75) - self.quartile(.5))
             - (self.quartile(.5) - self.quartile(.25)))
            / ((self.quartile(.75) - self.quartile(.5))
               + (self.quartile(.5) - self.quartile(.25))))

    @property
    def pearson(self):
        return self.moment_centre(3) ** 2 / self.moment_centre(2) ** 3

    @property
    def fisher(self):
        return sqrt(self.pearson)

    @property
    def aplatissement_pearson(self):
        return self.moment_centre(4) / self.var ** 2

    @property
    def aplatissement_fisher(self):
        return self.aplatissement_pearson - 3


def create_stat_table(series: pd.Series) -> None:
    stats = StatistiqueSeries(series)
    moyenne_widget = widgets.FloatText(
        value=stats.moyenne, description='Среднее:')
    var_widget = widgets.FloatText(value=stats.var, description='Дисперсия:')
    ecart_type_widget = widgets.FloatText(
        value=stats.ecart_type,
        description='Среднеквадратическое отклонение:')
    coef_of_var_widget = widgets.FloatText(
        value=stats.coef_of_var,
        description='Коэффициент вариации:')
    quartile_widget = widgets.FloatSlider(
        min=0, max=1, step=0.01, value=0.5, description='Квартиль:')
    quartile_value_widget = widgets.FloatText(description='Значение квартиля:')
    mediane = widgets.FloatText(value=stats.medianne, description='Медиана:')
    gini = widgets.FloatText(value=stats.gini_plot, description='Джини:')
    ordre_widget = widgets.IntText(value=1, description='Порядок:')
    value_widget = widgets.FloatText(value=0, description='Значение:')
    moment_widget = widgets.Dropdown(
        options=[
            'Центральный момент',
            'Центральный момент к значению a',
            'Простой момент'],
        description='Момент:')
    moment_value_widget = widgets.FloatText(description='Значение момента:')

    def update_moment_value(change):
        if moment_widget.value == 'Центральный момент':
            moment_value_widget.value = stats.moment_centre(ordre_widget.value)
        elif moment_widget.value == 'Центральный момент к значению a':
            moment_value_widget.value = stats.moment_centre_a(
                ordre_widget.value, value_widget.value)
        else:
            moment_value_widget.value = stats.simple_ordre(ordre_widget.value)
    ordre_widget.observe(update_moment_value, 'value')
    value_widget.observe(update_moment_value, 'value')
    moment_widget.observe(update_moment_value, 'value')

    def update_quartile_value(change):
        quartile_value_widget.value = stats.quartile(change['new'])
    quartile_widget.observe(update_quartile_value, 'value')

    attribute_widget = widgets.Dropdown(
        options=[
            'Юль',
            'Пирсон',
            'Фишер',
            'Аплатиссемент Пирсон',
            'Аплатиссемент Фишер'],
        description='Атрибут:')
    attribute_value_widget = widgets.FloatText(
        description='Значение атрибута:')

    def update_attribute_value(change):
        if attribute_widget.value == 'Юль':
            attribute_value_widget.value = stats.yule
        elif attribute_widget.value == 'Пирсон':
            attribute_value_widget.value = stats.pearson
        elif attribute_widget.value == 'Фишер':
            attribute_value_widget.value = stats.fisher
        elif attribute_widget.value == 'Аплатиссемент Пирсон':
            attribute_value_widget.value = stats.aplatissement_pearson
        elif attribute_widget.value == 'Аплатиссемент Фишер':
            attribute_value_widget.value = stats.aplatissement_fisher
    attribute_widget.observe(update_attribute_value, 'value')
    display(moyenne_widget, var_widget, ecart_type_widget, coef_of_var_widget,
            quartile_widget, quartile_value_widget, mediane, gini,
            ordre_widget,
            value_widget, moment_widget, moment_value_widget, attribute_widget,
            attribute_value_widget)
    stats.courbe_cumulative


def make_group_table(grop_column: Union[str, list[str]],
                     agg_column: str,
                     agg_function: str,
                     data: pd.DataFrame,
                     sorting_column: str = '%') -> pd.DataFrame:
    """группирует по признаку/признкам"""
    if not isinstance(grop_column, list) and not isinstance(grop_column, str):
        raise TypeError('grop_column must be list or string')
    if not isinstance(agg_column, str):
        raise TypeError('agg_column must be string')
    if not isinstance(agg_function, str):
        raise TypeError('agg_function must be string')
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be DataFrame')
    table: pd.DataFrame = (
        data
        .groupby(by=grop_column)
        .agg({agg_column: agg_function})
        .reset_index()
        .rename(columns={agg_column: f'{agg_column}_{agg_function}'}))
    table['%'] = round((table[
        f'{agg_column}_{agg_function}'
        ] / sum(table[f'{agg_column}_{agg_function}'])) * 100, 2)
    return table.sort_values(
        by=sorting_column, ascending=False).reset_index(drop=True)


def make_pieplot_plotly(group_column: str,
                        agg_column: str,
                        agg_function: str,
                        title: str,
                        data: pd.DataFrame) -> None:
    df = make_group_table(group_column, agg_column, agg_function, data)
    fig = px.pie(df,
                 values=f'{agg_column}_{agg_function}',
                 names=group_column,
                 title=title)
    fig.show()


if __name__ == '__main__':
    testing()
