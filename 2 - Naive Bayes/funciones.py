import pandas as pd
from sklearn.model_selection import StratifiedKFold
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import itertools
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

class NaiveBayesAIDS:

    def __init__(self, m, valores_posibles):
        ''' 
        Parametros
        ----------
        m: tamaño equivalente de muestra
        valores_posibles: diccionario con los valores posibles que puede tomar cada atributo
        '''
        self.m = m
        self.valores_posibles = valores_posibles
        self.probabilidades_0 = None
        self.probabilidades_1 = None

    def fit(self, X, Y):
        '''
        Aplica el algoritmo de Naive Bayes para entrenar el modelo
        Guarda el logaritmo de las probabilidades de cada valor de cada atributo para cada clase
        para facilitar el cálculo de la probabilidad de una entrada en el método predict
        y evitar errores al multiplicar valores muy pequeños
        
        para evitar eventos con probabilidad 0,
        se aplica la m-estimacion: P(X|Y) = (e + m*p) / (m + n)
        asumiendo p = 1 / |valores|

        Parametros
        ----------
        X: DataFrame con los datos de entrenamiento
        Y: Resultados de cada entrada en X
        
        '''

        self.total_entradas = len(X)
        self.totales_0 = len(X[Y == 0])
        self.totales_1 = len(X[Y == 1])
        self.probabilidades_0 = {}
        self.probabilidades_1 = {}

        for cat in self.valores_posibles.keys():
            self.probabilidades_0[cat] = {}
            self.probabilidades_1[cat] = {}

            p = 1 / len(self.valores_posibles[cat])

            for val in self.valores_posibles[cat]:
                # P(X|Y=0)
                e = len(X[(X[cat] == val) & (Y == 0)])
                self.probabilidades_0[cat][val] = (e + self.m * p) / (self.m + self.totales_0)
                
                # P(X|Y=1)
                e = len(X[(X[cat] == val) & (Y == 1)])
                self.probabilidades_1[cat][val] = (e + self.m * p) / (self.m + self.totales_1)

    def __predict_proba_entrada(self, X):
        resultado_0 = self.totales_0 / self.total_entradas
        resultado_1 = self.totales_1 / self.total_entradas

        for cat in self.valores_posibles.keys():
            resultado_0 *= self.probabilidades_0[cat][X[cat]]
            resultado_1 *= self.probabilidades_1[cat][X[cat]]
        
        norma = resultado_0 + resultado_1
        return [resultado_0 / norma, resultado_1 / norma]

    def predict_proba(self, X):
        if (not self.probabilidades_0 or not self.probabilidades_1):
            raise Exception("El modelo no ha sido entrenado")

        Y_probs_predicho = [self.__predict_proba_entrada(X.iloc[i]) for i in range(len(X))]
        
        return Y_probs_predicho

    def __predict_proba_log_entrada(self, X):
        resultado_0 = math.log(self.totales_0 / self.total_entradas)
        resultado_1 = math.log(self.totales_1 / self.total_entradas)

        for cat in self.valores_posibles.keys():
            resultado_0 += math.log(self.probabilidades_0[cat][X[cat]])
            resultado_1 += math.log(self.probabilidades_1[cat][X[cat]])
        
        return [resultado_0, resultado_1]
    
    def predict_proba_log(self, X):
        if (not self.probabilidades_0 or not self.probabilidades_1):
            raise Exception("El modelo no ha sido entrenado")

        Y_probs_predicho = [self.__predict_proba_log_entrada(X.iloc[i]) for i in range(len(X))]
        
        return Y_probs_predicho

    def __predict_entrada(self, X):

        resultado_0 = math.log(self.totales_0 / self.total_entradas)
        resultado_1 = math.log(self.totales_1 / self.total_entradas)

        for cat in self.valores_posibles.keys():
            resultado_0 += math.log(self.probabilidades_0[cat][X[cat]])
            resultado_1 += math.log(self.probabilidades_1[cat][X[cat]])
        
        return 0 if resultado_0 > resultado_1 else 1

    def predict(self, X):
        if (not self.probabilidades_0 or not self.probabilidades_1):
            raise Exception("El modelo no ha sido entrenado")

        Y_predicho = [self.__predict_entrada(X.iloc[i]) for i in range(len(X))]
        return Y_predicho

    def get_params(self, deep=True):
        return {
            'm': self.m,
            'valores_posibles': self.valores_posibles
            }


    def set_params(self, **params):
        if 'm' in params:
            self.m = params['m']
        
        if 'valores_posibles' in params:
            self.valores_posibles = params['valores_posibles']
            
        return self

def validacion_cruzada(X_train, Y_train, m, atributos_a_categorizar, valores_posibles, splits):
    skf = StratifiedKFold(n_splits = splits)

    scores_accuracy = []
    scores_precision = []
    scores_recall = []
    scores_f1 = []
    
    for train_index, val_index in skf.split(X_train, Y_train):
        
        X_train_cv, X_val_cv= X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv= Y_train.iloc[train_index], Y_train.iloc[val_index]

        X_train_copy = X_train_cv.copy()
        X_val_copy = X_val_cv.copy()
        
        discretizer = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy='kmeans', subsample=200_000, random_state=12345)
        puntos_corte = {}
        for atributo in atributos_a_categorizar:
            X_train_copy[atributo] = discretizer.fit_transform(X_train_copy[[atributo]]).astype(int)
            puntos_corte[atributo] = discretizer.bin_edges_[0][1:3]
            valores_posibles[atributo] = np.unique(X_train_copy[atributo])

        # Discretizo el resto del conjunto de datos utilizando los puntos de corte aplicados a X_train
        for atributo in atributos_a_categorizar:
            X_val_copy[atributo] = np.digitize(X_val_copy[atributo], puntos_corte[atributo])
        
        # Defino y entreno al modelo
        modelo_cv = NaiveBayesAIDS(m, valores_posibles)
        modelo_cv.fit(X_train_copy, y_train_cv)
        y_pred = modelo_cv.predict(X_val_copy)

        # Calculo métricas
        scores_accuracy.append(accuracy_score(y_val_cv, y_pred))
        scores_precision.append(precision_score(y_val_cv, y_pred, pos_label=0))
        scores_recall.append(recall_score(y_val_cv, y_pred, pos_label=0))
        scores_f1.append(f1_score(y_val_cv, y_pred, pos_label=0))
    
    return np.mean(scores_accuracy), np.mean(scores_precision), np.mean(scores_recall), np.mean(scores_f1)

def seleccionar_subconjunto_a_eliminar(X_train, Y_train, m, potenciales_atributos_a_dropear, valores_posibles, atributos_a_categorizar):
    accuracy_max = -1
    f1_max = -1
    subset_optimo = []

    for r in range(len(potenciales_atributos_a_dropear) + 1):
        for subset in itertools.combinations(potenciales_atributos_a_dropear, r):
            valores_posibles_aux = valores_posibles.copy()
            
            for key in list(subset):
                valores_posibles_aux.pop(key)

            atributos_a_categorizar_aux = atributos_a_categorizar.copy()
            for key in list(subset):
                if key in atributos_a_categorizar_aux:
                    atributos_a_categorizar_aux.remove(key)

            accuracy, _, _, f1 = validacion_cruzada(X_train, Y_train, m, atributos_a_categorizar_aux, valores_posibles_aux, 5)

            if (f1 > f1_max):
                accuracy_max = accuracy
                f1_max = f1
                subset_optimo = subset

    return subset_optimo, accuracy_max, f1_max
        
def get_accuracy_precision_recall_f1(Y_real, Y_predicho, objetivo=0):
    """
    Calcula la accuracy, precision, recall y f1
    """
    accuracy  = accuracy_score(Y_real, Y_predicho)
    precision = precision_score(Y_real, Y_predicho, pos_label=objetivo)
    recall    = recall_score(Y_real, Y_predicho, pos_label=objetivo)
    f1        = f1_score(Y_real, Y_predicho, pos_label=objetivo)

    return accuracy, precision, recall, f1  

def plot_metricas(resultados, cant):
    m_values = list(resultados.keys())[:cant]
    accuracy_values = [resultados[m][0] for m in m_values]
    f1_values = [resultados[m][3] for m in m_values]

    min_accuracy = min(accuracy_values)
    min_f1 = min(f1_values)
    min_both = min(min_accuracy, min_f1)

    plt.figure(figsize=(8, 6))
    plt.plot(m_values, accuracy_values, label='Accuracy', color='blue')
    plt.plot(m_values, f1_values, label='F1', color='green')
    
    plt.xlabel('M')
    plt.ylabel('Valor Métrica')
    plt.title('Curvas de Accuracy y F1 para diferentes valores de M')
    plt.legend()
    plt.xlim(0, cant+5)
    plt.ylim([min_both - 0.030, 0.95])
    plt.grid(True)
    plt.show()

def curva_precision_recall(modelo, X, Y):
    probas = modelo.predict_proba(X)

    probs_clase_0 = [p[0] for p in probas]

    precision, recall, thresholds = precision_recall_curve(Y, probs_clase_0, pos_label=0)

    thresholds = np.append(thresholds, 1)

    # Primera gráfica
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precision, label='Precision', color='blue')
    plt.plot(thresholds, recall, label='Recall', color='green')

    plt.xlabel('Umbral')
    plt.ylabel('Precision / Recall')
    plt.title('Curva Precision-Recall')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.grid(True)

    # Diferencias para encontrar el umbral óptimo
    diferencias = np.abs(precision - recall)
    indice_minimo = np.argmin(diferencias)
    umbral_mejor = thresholds[indice_minimo]

    print(f'Umbral donde se da el cruce: {umbral_mejor}')


def curva_precision_recall_log(modelo, X, Y):
    prob_log = modelo.predict_proba_log(X)

    # Me quedo con clase 0
    prob_log_0 = [item[0] for item in prob_log]

    precision_log, recall_log, thresholds_log = precision_recall_curve(Y, prob_log_0, pos_label=0)

    precision_log = precision_log[:-1]
    recall_log = recall_log[:-1]

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds_log, precision_log, label='Precision', color='blue')
    plt.plot(thresholds_log, recall_log, label='Recall', color='green')

    plt.xlabel('Umbral')
    plt.ylabel('Precision / Recall')
    plt.title('Curva Precision-Recall (probabilidad logarítimica)')
    plt.legend()
    plt.grid(True)
    plt.show()

    diferencias = np.abs(precision_log - recall_log)
    indice_minimo = np.argmin(diferencias)
    umbral_mejor = thresholds_log[indice_minimo]

    print(f'Umbral donde se da el cruce: {umbral_mejor}')

def plot_confusion_matrix(Y_real, Y_predicho):
    plt.figure(figsize=(7, 5))

    sns.heatmap(confusion_matrix(Y_real, Y_predicho), annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0, 1])

    plt.ylabel('Clase verdadera')
    plt.xlabel('Clase predicha')
    plt.title('Matriz de confusión')

    plt.show()