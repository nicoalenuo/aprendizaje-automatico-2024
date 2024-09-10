import pandas as pd
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import chi2_contingency
import itertools
import math
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')

OBJETIVO = 'cid'
DATASET_FILE = 'data.csv'

class NaiveBayesAIDS:

    def __init__(self, m, valores_posibles, atributos_a_categorizar, preprocesar=True):
        ''' 
        Parametros
        ----------
        m: tamaño equivalente de muestra
        valores_posibles: map {categoria, [valores posibles que puede tomar la categoria]} NO incluyendo la categoria objetivo
        atributos_a_categorizar: lista de atributos que se categorizarán por ser continuos o tener un rango amplio de valores posibles
        preprocesar: booleano que indica si se deben categorizar los atributos en el array atributos_a_categorizar durante el entrenamiento o no (por ya estar categorizados)
        '''
        self.m = m
        self.valores_posibles = valores_posibles
        self.atributos_a_categorizar = atributos_a_categorizar
        self.preprocesar = preprocesar
        self.puntos_corte = None
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

        # Se copian los datos para no modificar los originales
        X_copy = X.copy()
        Y_copy = Y.copy()

        # Si los atributos a categorizar no estan categorizados, se ejecuta el siguiente bloque
        if (self.preprocesar):
            discretizer = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy='kmeans', random_state=12345)
            self.puntos_corte = {}
            for atributo in self.atributos_a_categorizar:
                X_copy[atributo] = discretizer.fit_transform(X_copy[[atributo]]).astype(int)
                self.puntos_corte[atributo] = discretizer.bin_edges_[0][1:3]
                self.valores_posibles[atributo] = np.unique(X_copy[atributo])
        
        self.total_entradas = len(X_copy)
        self.totales_0 = len(X_copy[Y_copy == 0])
        self.totales_1 = len(X_copy[Y_copy == 1])
        self.probabilidades_0 = {}
        self.probabilidades_1 = {}

        for cat in self.valores_posibles.keys():
            self.probabilidades_0[cat] = {}
            self.probabilidades_1[cat] = {}

            p = 1 / len(self.valores_posibles[cat])

            for val in self.valores_posibles[cat]:
                # P(X|Y=0)
                e = len(X_copy[(X_copy[cat] == val) & (Y_copy == 0)])
                self.probabilidades_0[cat][val] = (e + self.m * p) / (self.m + self.totales_0)
                
                # P(X|Y=1)
                e = len(X_copy[(X_copy[cat] == val) & (Y_copy == 1)])
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
        
        # Se copian los datos para no modificar los originales
        # Además, se categorizan los atributos con los puntos de corte calculados en el entrenamiento
        X_copy = X.copy()
        for categoria in self.atributos_a_categorizar:
            X_copy[categoria] = np.digitize(X_copy[categoria], self.puntos_corte[categoria])

        Y_probs_predicho = [self.__predict_proba_entrada(X_copy.iloc[i]) for i in range(len(X))]
        
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
        
        # Se copian los datos para no modificar los originales
        # Además, se categorizan los atributos con los puntos de corte calculados en el entrenamiento
        X_copy = X.copy()
        for categoria in self.atributos_a_categorizar:
            X_copy[categoria] = np.digitize(X_copy[categoria], self.puntos_corte[categoria])

        Y_predicho = [self.__predict_entrada(X_copy.iloc[i]) for i in range(len(X))]

        return Y_predicho

    def get_params(self, deep=True):
        # Devuelve un diccionario con todos los parámetros del modelo
        params = vars(self).copy()
        if deep:
            for key, value in params.items():
                if hasattr(value, 'get_params'):
                    params[key] = value.get_params(deep=True)

        return params


    def set_params(self, **parameters):
        valid_params = self.get_params(deep=True)
        for parameter, value in parameters.items():
            if parameter in valid_params:
                setattr(self, parameter, value)
        return self
  
# -----------------------
# Todo esto para el informe
# -----------------------

if (__name__ == '__main__'):
    X = dataset.copy().drop(columns=[OBJETIVO, 'pidnum'])
    Y = dataset[OBJETIVO].copy()

    valores_posibles = {}
    for categoria in X.columns:
        valores_posibles[categoria] = X[categoria].unique()

    atributos_a_categorizar = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']    

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 12345, stratify=Y)
    X_train, X_validacion, Y_train, Y_validacion = train_test_split(X_train, Y_train, test_size = 0.15, random_state = 12345, stratify=Y_train)
    
    coso = NaiveBayesAIDS(30, valores_posibles, atributos_a_categorizar, True)
    coso.fit(X_train, Y_train)
    Y_pred = coso.predict(X_validacion)
    accuracy = accuracy_score(Y_validacion, Y_pred)
    print(f'Accuracy: {accuracy}')

    # -----------------------
    # Curva Precision-Recall
    # -----------------------

    probas = coso.predict_proba(X_test)

    probs_clase_0 = [p[0] for p in probas]

    precision, recall, thresholds = precision_recall_curve(Y_test, probs_clase_0, pos_label=0)

    thresholds = np.append(thresholds, 1)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision, label='Precision', color='blue')
    plt.plot(thresholds, recall, label='Recall', color='green')

    plt.xlabel('Threshold')
    plt.ylabel('Precision / Recall')
    plt.title('Curva Precision-Recall')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0]) 
    plt.legend()
    plt.grid(True)
    plt.show()

    diferencias = np.abs(precision - recall)
    indice_minimo = np.argmin(diferencias)
    umbral_mejor = thresholds[indice_minimo]
    precision_mejor = precision[indice_minimo]
    recall_mejor = recall[indice_minimo]

    print(f'Umbral óptimo: {umbral_mejor}')
    print(f'Precision óptima: {precision_mejor}')
    print(f'Recall óptimo: {recall_mejor}')

    # -----------------------
    # Selección de atributos
    # -----------------------
    '''
    discretizer = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy='kmeans', random_state=12345)
    puntos_corte = {}
    for atributo in atributos_a_categorizar:
        X_train[atributo] = discretizer.fit_transform(X_train[[atributo]]).astype(int)
        puntos_corte[atributo] = discretizer.bin_edges_[0][1:3]
        valores_posibles[atributo] = np.unique(X_train[atributo])

    pd.options.display.float_format = '{:.12f}'.format
    correlacion = []

    for atributo in X.columns:
        tabla_contingencia = pd.crosstab(X[atributo], Y)
        _, p, _, _ = chi2_contingency(tabla_contingencia)
        correlacion.append((atributo, p))

    df_correlacion = pd.DataFrame(correlacion, columns=['Atributo', 'p'])

    # Para auementar la cantidad de atributos a dropear, se puede aumentar el valor de p
    atributos_a_dropear = df_correlacion[df_correlacion['p'] > 0.01]['Atributo'].tolist()

    print("Tabla de correlación:")
    print(df_correlacion.to_string(index=False))

    print(f'\nAtributos a dropear: {atributos_a_dropear}')
    
    accuracy_max = 0
    subset_optimo = []

    for r in range(len(atributos_a_dropear) + 1):
        for subset in itertools.combinations(atributos_a_dropear, r):
            X_aux = X_train.copy().drop(columns=list(subset))
            #quitar las claves subset de valores_posibles
            valores_posibles_aux = valores_posibles.copy()
            for key in list(subset):
                valores_posibles_aux.pop(key)
            atributos_a_categorizar_aux = list(set(atributos_a_categorizar) - set(list(subset)))
            coso = NaiveBayesAIDS(30, valores_posibles_aux, atributos_a_categorizar_aux, False)
            coso.set_params(puntos_corte=puntos_corte)
            coso.fit(X_aux, Y_train)
            Y_pred = coso.predict(X_validacion)
            accuracy = accuracy_score(Y_validacion, Y_pred)

            if accuracy > accuracy_max:
                print(f'Accuracy maximo: {accuracy}')
                accuracy_max = accuracy
                subset_optimo = subset

    print(f'Accuracy maximo: {accuracy_max}')
    print(f'Subset optimo: {subset_optimo}')
    '''