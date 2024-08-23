import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
Atributos numericos que deberian ser discretizados:

no parecen ser utiles para predecir:
    pidnum
    time

parecen ser utiles para predecir:
    age
    wtkg
    karnof
    preanti
    cd40
    cd420
    cd80
    cd820

'''

DATASET_FILE = "data.csv"
OBJETIVO = 'cid'

class ArbolDecision:

    @staticmethod
    def ID3(X, Y):
        '''
        Genera un arbol de decision a partir de los datos X y Y.
        los nodos tienen 3 atributos
        label: Nombre del atributo a comparar, en caso de haber llegado a una hoja, es None, en caso de ser la raiz, es el atributo con mayor ganancia
        children: Lista de pares (valor, nodo) donde valor es el valor del atributo a comparar y nodo es el subarbol que se debe seguir
        result: En caso de ser una hoja, es el resultado final predicho de la clasificacion, en caso de ser un nodo, es None
        '''

        valores_unicos = Y.unique()

        if len(valores_unicos) == 0:
            return {'label': None, 'children': None, 'result': 0}

        if len(valores_unicos) == 1:
            return {'label': None, 'children': None, 'result' : valores_unicos[0]}
        
        mejor_atributo = get_mejor_atributo(X, Y)

        if mejor_atributo is None:
            return {'label': None, 'children': None, 'result': 0}

        children = []
        for valor in X[mejor_atributo].unique():
            subset_X = X[X[mejor_atributo] == valor].drop(columns=[mejor_atributo])
            subset_Y = Y[X[mejor_atributo] == valor]
            children.append((valor, ArbolDecision.ID3(subset_X, subset_Y)))

        return {
            'label': mejor_atributo,
            'children': children,
            'result': None
        }


    def entrenar(self, X, Y):
        self.arbol = ArbolDecision.ID3(X, Y)
    
    @staticmethod
    def predecir_entrada(arbol, X):
        '''
        Predice el resultado de una sola entrada X utilizando el arbol de decision
        '''
        if arbol['result'] is not None:
            return arbol['result']

        valor = X[arbol['label']]
        for valor_hijo, hijo in arbol['children']:
            if valor == valor_hijo:
                return ArbolDecision.predecir_entrada(hijo, X)

        return 0


    def predecir(self, X):
        '''
        Predice el resultado de X utilizando el arbol de decision
        '''
        Y_predicho = []
        for i in range(len(X)):
            Y_predicho.append(ArbolDecision.predecir_entrada(self.arbol, X.iloc[i]))
        
        return Y_predicho

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

def discretizar_atributo_por_resultado(X, Y, max_range_split):
    """
    Discretiza un atributo X con respecto al resultado Y, maximizando la pureza de las particiones
    """
    puntos_corte = []
    valores_unicos = sorted(X.unique())
    
    for i in range(1, len(valores_unicos)):
        punto_corte_actual = (valores_unicos[i-1] + valores_unicos[i]) / 2
        ganancia_actual = calcular_ganancia_informacion(X, Y, punto_corte_actual)
        puntos_corte.append((punto_corte_actual, ganancia_actual))
    
    puntos_corte_ordenados = sorted(puntos_corte, key = lambda x: x[1], reverse = True)
    
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
    
    return X_discretizado

def discretizar_atributos(dataset, atributos, max_range_split):
    """
    Discretiza los atributos en función del objetivo, respetando max_range_split.
    """
    dataset_discretizado = dataset.copy()
    
    for atributo in atributos:
        dataset_discretizado[atributo] = discretizar_atributo_por_resultado(dataset[atributo], dataset[OBJETIVO], max_range_split)
    
    return dataset_discretizado

def get_entropia(Y):
    cantidad_unicos = Y.unique()
    total = len(Y)
    entropia = 0

    for unico in cantidad_unicos:
        cantidad = len(Y[Y == unico])
        entropia -= (cantidad / total) * (math.log2(cantidad / total) if cantidad != 0 else 0)

    return entropia

def get_entropia_atributo(X, Y, atributo):
    valores_unicos = X[atributo].unique()
    total = len(X)
    entropia = 0

    for unico in valores_unicos:
        cantidad = len(X[X[atributo] == unico])
        entropia += (cantidad / total) * get_entropia(Y[X[atributo] == unico])

    return entropia


def get_mejor_atributo(X, Y):
    ganancia_max = None
    mejor_atributo = None

    entropia = get_entropia(Y)

    for atributo in X.columns:
        ganancia_actual = entropia - get_entropia_atributo(X, Y, atributo)
        if (mejor_atributo == None or ganancia_actual > ganancia_max):
            ganancia_max = ganancia_actual
            mejor_atributo = atributo

    return mejor_atributo

dataset = pd.read_csv(DATASET_FILE)

dataset_discretizado = discretizar_atributos(dataset.copy(), ['pidnum', 'time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820'], 3)

#Todo esto de aca abajo moverlo al informe

X_manual = dataset_discretizado.copy().drop(columns=[OBJETIVO])
Y_manual = dataset_discretizado[OBJETIVO].copy()

X_librerias = dataset.copy().drop(columns=[OBJETIVO])
Y_librerias = dataset[OBJETIVO].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X_manual, Y_manual, test_size=0.15, random_state = 12345)
X_train_librerias, X_test_librerias, Y_train_librerias, Y_test_librerias = train_test_split(X_librerias, Y_librerias, test_size=0.15, random_state=12345)

ArbolDecisionManual = ArbolDecision()
ArbolDecisionManual.entrenar(X_train, Y_train)
Y_predicho_manual = ArbolDecisionManual.predecir(X_test)
presicion_manual = accuracy_score(Y_test, Y_predicho_manual)

ArbolDecisionLibreria = DecisionTreeClassifier(criterion='entropy')
ArbolDecisionLibreria.fit(X_train_librerias, Y_train_librerias)
Y_predicho_arbol_libreria = ArbolDecisionLibreria.predict(X_test_librerias)
precision_arbol_libreria = accuracy_score(Y_test_librerias, Y_predicho_arbol_libreria)

RandomForest = RandomForestClassifier(criterion='entropy')
RandomForest.fit(X_train_librerias, Y_train_librerias)
Y_predicho_random_forest = RandomForest.predict(X_test_librerias)
presicion_random_forest = accuracy_score(Y_test_librerias, Y_predicho_random_forest)

print(f"Presicion del arbol de decision manual: {presicion_manual}")
print(f"Presicion del arbol de decision de libreria: {precision_arbol_libreria}")
print(f"Presicion del random forest de libreria: {presicion_random_forest}")
