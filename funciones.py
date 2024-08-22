import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np

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
MAX_RANGE_SPLIT = 3
OBJETIVO = 'cid'

VIVE = 0
MUERE = 1

dataset = pd.read_csv(DATASET_FILE)

def discretizar_atributo(dataset, atributos, max_range_split):
    return 0

def get_entropia(Y):
    cantidad_unicos = Y.unique()
    total = len(Y)
    entropia = 0

    for unico in cantidad_unicos:
        cantidad = len(Y[Y == unico])
        entropia -= (cantidad / total) * (np.log2(cantidad / total) if cantidad != 0 else 0)

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

    for atributo in X.columns.tolist():
        ganancia_actual = entropia - get_entropia_atributo(X, Y, atributo)
        if (mejor_atributo == None or ganancia_actual > ganancia_max):
            ganancia_max = ganancia_actual
            mejor_atributo = atributo

    return mejor_atributo

def ID3(X, Y):
    '''
    Genera un arbol de decision a partir de los datos X y Y
    los nodos tienen 3 atributos
    label: Nombre del atributo a comparar, en caso de haber llegado a una hoja, es None, en caso de ser la raiz, es el atributo con mayor ganancia
    children: Lista de nodos hijos, en caso de ser una hoja, es None
    result: En caso de ser una hoja, es el resultado final predicho de la clasificacion, en caso de ser un nodo, es None
    '''

    cantidad_unicos = Y.unique()

    if (len(cantidad_unicos) == 1):
        return {'label': None, 'children': None, 'result' : cantidad_unicos[0]}
    
    # En caso de que no hayan datos para cierto atributo, asumo que es 0
    if (len(cantidad_unicos) == 0):
        return {'label': None, 'children': None, 'result': 0}
    
    mejor_atributo = get_mejor_atributo(X, Y)

    X_nuevo = X.drop(columns=[mejor_atributo])

    return {
        'label': mejor_atributo,
        'children': [
            ID3(X_nuevo[X[mejor_atributo] == valor], Y[X[mejor_atributo] == valor])
            for valor in X[mejor_atributo].unique()
        ],
        'result': None
    }