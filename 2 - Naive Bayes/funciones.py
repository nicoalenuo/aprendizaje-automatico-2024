import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

dataset = pd.read_csv('data.csv').drop(columns=['pidnum'])

OBJETIVO = 'cid'
DATASET_FILE = 'data.csv'


class NaiveBayesAIDS:

    def __init__(self, m, valores_posibles, atributos_a_categorizar, threshold):
        '''
        m: tamaño equivalente de muestra
        valores_posibles: map {categoria, [valores posibles]} NO incluyendo la categoria objetivo
        '''
        self.m = m
        self.valores_posibles = valores_posibles
        self.atributos_a_categorizar = atributos_a_categorizar
        self.puntos_corte = None
        self.probabilidades_0 = None
        self.probabilidades_1 = None
        self.entrenado = False
        self.discretizer = KBinsDiscretizer(n_bins=3, encode="ordinal",strategy='quantile',random_state=12345)
        self.threshold=threshold

    def fit(self, X_train, y):
        '''
        X_train: DataFrame con los datos de entrenamiento
        

        se aplica la m-estimacion: P(X|Y) = (e + m*p) / (m + n)
        asumiendo p = 1 / |valores|
        '''

        X_train[OBJETIVO] = y

        X_copy = X_train.copy()
        Y_copy = y.copy()

        puntos_corte = {}
        for atributo in self.atributos_a_categorizar:
            X_copy[atributo], puntos_corte_atributo = categorizar_atributo(X_copy[atributo], Y_copy, 3)
            puntos_corte[atributo] = puntos_corte_atributo.copy()
            self.valores_posibles[atributo] = [i for i in range(len(puntos_corte[atributo]) + 1)]

        self.puntos_corte = puntos_corte
        
        '''
        X_copy = X_train.copy()
                
        puntos_corte = {}
        for atributo in self.atributos_a_categorizar:
            X_copy[atributo] = discretizer.fit_transform(X_copy[[atributo]])
            bin_edges = discretizer.bin_edges_
            if len(bin_edges[0]) == 4:
                puntos_corte[atributo] = bin_edges[0][1:3]
            else:
                puntos_corte[atributo] = bin_edges[0][1:2] #Algunos casos no se generan 2 cortes
        self.puntos_corte = puntos_corte
        '''

        probabilidades_0 = {}
        probabilidades_1 = {}
        
        self.totales = len(X_copy)
        self.totales_0 = len(X_copy[X_copy[OBJETIVO] == 0])
        self.totales_1 = len(X_copy[X_copy[OBJETIVO] == 1])

        for cat in self.valores_posibles.keys():
            probabilidades_0[cat] = {}
            probabilidades_1[cat] = {}

            # P(X|Y=0)
            for val in self.valores_posibles[cat]:
                p = 1 / len(self.valores_posibles[cat])
                e = len(X_copy[(X_copy[cat] == val) & (X_copy[OBJETIVO] == 0)])

                probabilidades_0[cat][val] = (e + self.m*p) / (self.m + self.totales_0)

            # P(X|Y=1)
            for val in self.valores_posibles[cat]:
                p = 1 / len(self.valores_posibles[cat])
                e = len(X_copy[(X_copy[cat] == val) & (X_copy[OBJETIVO] == 1)])

                probabilidades_1[cat][val] = (e + self.m*p) / (self.m + self.totales_1)

        self.probabilidades_0 = probabilidades_0
        self.probabilidades_1 = probabilidades_1
        self.entrenado = True


    def predecir_entrada(self, X):
        X_copy = X.copy()

        for cat in self.atributos_a_categorizar: 
            valor = 0
            puntos_corte = self.puntos_corte[cat]
            for i in range(len(puntos_corte)):
                if X[cat].iloc[0] > puntos_corte[i]:
                    valor = i + 1
                else:
                    break
            X_copy[cat] = valor
        
        res_0 = self.totales_0 / self.totales
        res_1 = self.totales_1 / self.totales

        for cat in X_copy.columns.difference([OBJETIVO]):
            res_0 *= self.probabilidades_0[cat][X_copy[cat].iloc[0]]
            res_1 *= self.probabilidades_1[cat][X_copy[cat].iloc[0]]
        
        return 0 if res_0 > res_1 else 1

    
    def predict(self, X):
        if (not self.entrenado):
            raise Exception("El modelo no ha sido entrenado")

        Y_predicho = [self.predecir_entrada(X.iloc[[i]]) for i in range(len(X))]

        return Y_predicho

    def get_params(self, deep=True):
        # Devuelve un diccionario con todos los parámetros del modelo
        return {
            'm': self.m,
            'valores_posibles': self.valores_posibles,
            'atributos_a_categorizar': self.atributos_a_categorizar,
            'threshold': self.threshold,
        }

    def set_params(self, **parameters):
        valid_params = self.get_params(deep=True)
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
#########

def get_entropia(Y):
    valores_unicos = Y.unique()
    total = len(Y)
    entropia = 0

    for unico in valores_unicos:
        cantidad = len(Y[Y == unico])
        entropia -= (cantidad / total) * (math.log2(cantidad / total) if cantidad != 0 else 0)

    return entropia

def calcular_ganancia_informacion(X, Y, punto_corte):
    """
    Calcula la ganancia de información al dividir el atributo X en un punto de corte
    """
    Y_izquierda = Y[X <= punto_corte]
    Y_derecha = Y[X > punto_corte]
    
    entropia_total = get_entropia(Y)
    entropia_izquierda = get_entropia(Y_izquierda)
    entropia_derecha = get_entropia(Y_derecha)
    
    peso_izquierda = len(Y_izquierda) / len(Y)
    peso_derecha = len(Y_derecha) / len(Y)
    
    ganancia_informacion = entropia_total - (peso_izquierda * entropia_izquierda + peso_derecha * entropia_derecha)
    
    return ganancia_informacion

def categorizar_atributo(X, Y, max_range_split):
    """
    Categoriza un atributo X con respecto al resultado Y, maximizando la ganancia de información.
    Para esto, calcula la ganancia de información al dividir el atributo en todos los puntos de corte posibles
    y selecciona los max_range_split - 1 puntos de corte que maximizan la ganancia de información.
    Devuelve también los puntos de corte seleccionados
    """
    datos = pd.DataFrame({'X': X, 'Y': Y}).sort_values(by='X').drop_duplicates(subset='X')

    puntos_corte = []
    
    for i in range(1, len(datos)):
        if datos.iloc[i]['Y'] != datos.iloc[i-1]['Y']:
            punto_corte_actual = (datos.iloc[i]['X'] + datos.iloc[i-1]['X']) / 2
            ganancia_actual = calcular_ganancia_informacion(datos['X'], datos['Y'], punto_corte_actual)
            puntos_corte.append((punto_corte_actual, ganancia_actual))
    
    puntos_corte_ordenados = sorted(puntos_corte, key=lambda x: x[1], reverse=True)
    
    puntos_corte_seleccionados = sorted([p[0] for p in puntos_corte_ordenados[:max_range_split-1]])
    
    X_discretizado = []
    for valor in X:
        bin_asignado = 0
        for punto in puntos_corte_seleccionados:
            if valor > punto:
                bin_asignado += 1
            else:
                break
        X_discretizado.append(bin_asignado)
    
    return X_discretizado, puntos_corte_seleccionados        


if (__name__ == '__main__'):

    atributos_a_categorizar = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']

    X = dataset.copy().drop(columns=[OBJETIVO])
    Y = dataset[OBJETIVO].copy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 12345, stratify=Y)
    X_train, X_validacion, Y_train, Y_validacion = train_test_split(X_train, Y_train, test_size = 0.15, random_state = 12345, stratify=Y_train)

    valores_posibles = {}

    for categoria in dataset.columns:
        valores_posibles[categoria] = dataset[categoria].unique()

    m = 30
    coso = NaiveBayesAIDS(m, valores_posibles, atributos_a_categorizar, 0.5)
    coso.fit(X_train, Y_train)

    Y_pred = coso.predict(X_validacion)

    accuracy = accuracy_score(Y_validacion, Y_pred)

    print(f'Accuracy: {accuracy}')