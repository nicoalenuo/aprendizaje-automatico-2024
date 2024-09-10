import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import chi2_contingency
import itertools
import math

dataset = pd.read_csv('data.csv')

OBJETIVO = 'cid'
DATASET_FILE = 'data.csv'

class NaiveBayesAIDS:

    def __init__(self, m, valores_posibles, atributos_a_categorizar, preprocesar):
        ''' 
        Parametros
        ----------
        m: tamaño equivalente de muestra
        valores_posibles: map {categoria, [valores posibles]} NO incluyendo la categoria objetivo
        atriubtos_a_categorizar: lista de atributos que se categorizarán
        '''
        self.m = m
        self.valores_posibles = valores_posibles
        self.atributos_a_categorizar = atributos_a_categorizar
        self.puntos_corte = None
        self.probabilidades_0 = None
        self.probabilidades_1 = None
        self.discretizer = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy='quantile', random_state=12345)
        self.preprocesar = preprocesar

    def fit(self, X, Y):
        '''
        Parametros
        ----------
        X: DataFrame con los datos de entrenamiento
        Y: Resultados de cada entrada en X
        
        se aplica la m-estimacion: P(X|Y) = (e + m*p) / (m + n)
        asumiendo p = 1 / |valores|
        '''

        # Se copian los datos para no modificar los originales
        X_copy = X.copy()
        Y_copy = Y.copy()

        if (self.preprocesar):
            self.puntos_corte = {}
            for atributo in self.atributos_a_categorizar:
                X_copy[atributo], puntos_corte_atributo = categorizar_atributo(X_copy[atributo], Y_copy, 3)
                self.puntos_corte[atributo] = puntos_corte_atributo.copy()
                self.valores_posibles[atributo] = [i for i in range(len(self.puntos_corte[atributo]) + 1)]
        
        '''                
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
        
        self.total_entradas = len(X_copy)
        self.totales_0 = len(X_copy[Y_copy == 0])
        self.totales_1 = len(X_copy[Y_copy == 1])
        self.probabilidades_0 = {}
        self.probabilidades_1 = {}

        for cat in self.valores_posibles.keys():
            self.probabilidades_0[cat] = {}
            self.probabilidades_1[cat] = {}

            for val in self.valores_posibles[cat]:
                p = 1 / len(self.valores_posibles[cat])
    
                # P(X|Y=0)
                e = len(X_copy[(X_copy[cat] == val) & (Y_copy == 0)])
                self.probabilidades_0[cat][val] = (e + self.m * p) / (self.m + self.totales_0)
                
                # P(X|Y=1)
                e = len(X_copy[(X_copy[cat] == val) & (Y_copy == 1)])
                self.probabilidades_1[cat][val] = (e + self.m * p) / (self.m + self.totales_1)


    def predecir_entrada(self, X):

        resultado_0 = self.totales_0 / self.total_entradas
        resultado_1 = self.totales_1 / self.total_entradas

        for cat in self.valores_posibles.keys():
            resultado_0 *= self.probabilidades_0[cat][X[cat]]
            resultado_1 *= self.probabilidades_1[cat][X[cat]]
        
        return 0 if resultado_0 > resultado_1 else 1

    
    def predict(self, X):
        if (not self.probabilidades_0 or not self.probabilidades_1):
            raise Exception("El modelo no ha sido entrenado")
        
        # Se copian los datos para no modificar los originales
        # Además, se categorizan los atributos con los puntos de corte calculados en el entrenamiento
        X_copy = X.copy()
        for categoria in self.atributos_a_categorizar:
            X_copy[categoria] = categorizar_con_puntos_de_corte(X_copy[categoria], self.puntos_corte[categoria])

        Y_predicho = [self.predecir_entrada(X_copy.iloc[i]) for i in range(len(X))]

        return Y_predicho

    def get_params(self, deep=True):
        # Devuelve un diccionario con todos los parámetros del modelo
        return {
            'm': self.m,
            'valores_posibles': self.valores_posibles,
            'atributos_a_categorizar': self.atributos_a_categorizar,
        }

    def set_params(self, **parameters):
        valid_params = self.get_params(deep=True)
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
# -----------------------
# Funciones para discretizar atributos
# -----------------------

def get_entropia(Y):
    valores_unicos = Y.unique()
    total = len(Y)
    entropia = 0

    for unico in valores_unicos:
        cantidad = len(Y[Y == unico])
        entropia -= (cantidad / total) * (math.log2(cantidad / total) if cantidad != 0 else 0)

    return entropia

def categorizar_con_puntos_de_corte(X, puntos_corte):
    """
    Categoriza un atributo X con respecto a los puntos de corte dados
    """
    X_discretizado = []
    for valor in X:
        bin_asignado = 0
        for punto in puntos_corte:
            if valor > punto:
                bin_asignado += 1
            else:
                break
        X_discretizado.append(bin_asignado)
    
    return X_discretizado

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
    
    X_discretizado = categorizar_con_puntos_de_corte(X, puntos_corte_seleccionados)
    
    return X_discretizado, puntos_corte_seleccionados        

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

    puntos_corte = {}
    for atributo in atributos_a_categorizar:
        X_train[atributo], puntos_corte_atributo = categorizar_atributo(X_train[atributo], Y_train, 3)
        puntos_corte[atributo] = puntos_corte_atributo.copy()
        valores_posibles[atributo] = [i for i in range(len(puntos_corte[atributo]) + 1)]

    '''
    m = 30
    coso = NaiveBayesAIDS(m, valores_posibles, atributos_a_categorizar, True)
    coso.fit(X_train, Y_train)

    Y_pred = coso.predict(X_validacion)

    accuracy = accuracy_score(Y_validacion, Y_pred)

    print(f'Accuracy: {accuracy}')
    '''


    # -----------------------
    # Selección de atributos
    # -----------------------

    pd.options.display.float_format = '{:.12f}'.format
    correlacion = []

    for x in X.columns:
        tabla_contingencia = pd.crosstab(X[x], Y)
        _, p, _, _ = chi2_contingency(tabla_contingencia)
        correlacion.append((x, p))

    df_correlacion = pd.DataFrame(correlacion, columns=['Atributo', 'p'])

    # Para auementar la cantidad de atributos a dropear, se puede aumentar el valor de p
    atributos_a_dropear = df_correlacion[df_correlacion['p'] > 0.05]['Atributo'].tolist()

    print("Tabla de correlación:")
    print(df_correlacion.to_string(index=False))

    print(f'\nAtributos a dropear: {atributos_a_dropear}')
    
    accuracy_max = 0
    subset_optimo = []

    for r in range(len(X.columns) + 1):
        for subset in itertools.combinations(atributos_a_dropear, r):
            X_aux = X.copy().drop(columns=list(subset))
            #quitar las claves subset de valores_posibles
            valores_posibles_aux = valores_posibles.copy()
            for key in list(subset):
                valores_posibles_aux.pop(key)
            atributos_a_categorizar_aux = list(set(atributos_a_categorizar) - set(list(subset)))
            coso = NaiveBayesAIDS(30, valores_posibles_aux, atributos_a_categorizar_aux, False)
            coso.set_params(puntos_corte=puntos_corte)
            coso.fit(X_train, Y_train)
            Y_pred = coso.predict(X_validacion)
            accuracy = accuracy_score(Y_validacion, Y_pred)
            if accuracy > accuracy_max:
                print(f'Accuracy maximo: {accuracy}')
                accuracy_max = accuracy
                subset_optimo = subset

    print(f'Accuracy maximo: {accuracy_max}')
    print(f'Subset optimo: {subset_optimo}')