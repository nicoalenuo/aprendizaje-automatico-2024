import pandas as pd
import math
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATASET_FILE = "data.csv"
OBJETIVO = 'cid'

class ArbolDecision:

    def __init__(self):
        self.arbol = None

    @staticmethod
    def ID3(X, Y, funcion_seleccion_atributo):
        '''
        Genera un arbol de decision a partir de los datos X y Y.
        los nodos tienen 3 atributos
        label: Nombre del atributo a comparar, en caso de haber llegado a una hoja, es None, en caso de ser la raiz, es el atributo con mayor ganancia
        children: Lista de pares (valor, nodo) donde valor es el valor del atributo a comparar y nodo es el subarbol que se debe seguir
        result: En caso de ser una hoja, es el resultado final predicho de la clasificacion, en caso de ser un nodo intermedio, es None
        '''

        valores_unicos = Y.unique()

        if len(valores_unicos) == 0:
            return {'label': None, 'children': None, 'result': 0}

        if len(valores_unicos) == 1:
            return {'label': None, 'children': None, 'result' : valores_unicos[0]}
        
        mejor_atributo = funcion_seleccion_atributo(X, Y)

        if mejor_atributo is None:
            return {'label': None, 'children': None, 'result': 0}

        children = []
        for valor in X[mejor_atributo].unique():
            subset_X = X[X[mejor_atributo] == valor].drop(columns=[mejor_atributo])
            subset_Y = Y[X[mejor_atributo] == valor]
            children.append((valor, ArbolDecision.ID3(subset_X, subset_Y, funcion_seleccion_atributo)))

        return {
            'label': mejor_atributo,
            'children': children,
            'result': None
        }


    def entrenar(self, X, Y, funcion_seleccion_atributo):
        if (funcion_seleccion_atributo not in [get_mejor_atributo_entropia, get_mejor_atributo_gain_ratio, get_mejor_atributo_impurity_reduction]):
            raise Exception("La funcion de seleccion de atributo no es valida")

        self.arbol = ArbolDecision.ID3(X, Y, funcion_seleccion_atributo)
    
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
        if self.arbol is None:
            raise Exception("El arbol no ha sido entrenado, por lo que no puede predecir")
            
        Y_predicho = [ArbolDecision.predecir_entrada(self.arbol, X.iloc[i]) for i in range(len(X))]
        
        return Y_predicho
    
# --------------------------------------------
# Funciones de entropía
# --------------------------------------------

def get_entropia(Y):
    valores_unicos = Y.unique()
    total = len(Y)
    entropia = 0

    for unico in valores_unicos:
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

# --------------------------------------------
# Funciones para discretizar atributos
# --------------------------------------------

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
    Para esto, calcula la ganancia de información al dividir el atributo en todos los puntos de corte posibles
    y selecciona los max_range_split - 1 puntos de corte que maximizan la ganancia de información
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

# --------------------------------------------
# Funciones de selección de atributo
# --------------------------------------------

def get_mejor_atributo_entropia(X, Y):
    '''
    Devuelve el mejor atributo para dividir el dataset en función de la entropía
    '''
    entropia_maxima = None
    mejor_atributo = None

    entropia = get_entropia(Y)

    for atributo in X.columns:

        entropia_atributo = entropia - get_entropia_atributo(X, Y, atributo)

        if (mejor_atributo == None or entropia_atributo > entropia_maxima):
            entropia_maxima = entropia_atributo
            mejor_atributo = atributo


    return mejor_atributo

def get_mejor_atributo_gain_ratio(X, Y):
    '''
    Devuelve el mejor atributo para dividir el dataset en función del gain ratio
    '''
    gain_ratio_maximo = None
    mejor_atributo = None

    entropia = get_entropia(Y)

    for atributo in X.columns:
        
        split_information = 0
        for valor in X[atributo].unique():
            split_information -= (len(X[X[atributo] == valor]) / len(X)) * math.log2(len(X[X[atributo] == valor]) / len(X))

        entropia_atributo = entropia - get_entropia_atributo(X, Y, atributo)

        gain_ratio = entropia_atributo / split_information if split_information != 0 else 0

        if (mejor_atributo == None or gain_ratio > gain_ratio_maximo):
            gain_ratio_maximo = gain_ratio
            mejor_atributo = atributo

    return mejor_atributo

def get_gini_atributo(X, Y, atributo):
    '''
    Devuelve el gini de un atributo
    '''

    valores_unicos = X[atributo].unique()
    total = len(X)
    gini = 0

    for unico in valores_unicos:
        cantidad = len(X[X[atributo] == unico])
        gini += (cantidad / total) * (1 - (len(Y[X[atributo] == unico][Y == 1]) / cantidad) ** 2 - (len(Y[X[atributo] == unico][Y == 0]) / cantidad) ** 2)

    return gini

def get_mejor_atributo_impurity_reduction(X, Y):
    '''
    Devuelve el mejor atributo para dividir el dataset en función de la reducción de impureza
    '''

    impurity_reduction_maxima = None
    mejor_atributo = None

    gini = 1 - (len(Y[Y == 1]) / len(Y)) ** 2 - (len(Y[Y == 0]) / len(Y)) ** 2

    for atributo in X.columns:
        gini_atributo = gini - get_gini_atributo(X, Y, atributo)

        if (mejor_atributo == None or gini_atributo > impurity_reduction_maxima):
            impurity_reduction_maxima = gini_atributo
            mejor_atributo = atributo

    return mejor_atributo


# --------------------------------------------
# Funciones de medidas y graficas
# --------------------------------------------

def get_accuracy_precision_recall_f1(Y_real, Y_predicho, objetivo=0):
    """
    Calcula la accuracy, precision, recall y f1
    """
    accuracy  = accuracy_score(Y_real, Y_predicho)
    precision = precision_score(Y_real, Y_predicho, pos_label=objetivo)
    recall    = recall_score(Y_real, Y_predicho, pos_label=objetivo)
    f1        = f1_score(Y_real, Y_predicho, pos_label=objetivo)

    return accuracy, precision, recall, f1

def entrenar_y_evaluar(X, Y, X_val, Y_val):
    '''
    Para un conjunto X, Y de entrenamiento, entrena un arbol utilizando las 3 funciones de seleccion de atributo
    y evalua el arbol en el conjunto de validacion X_val, Y_val
    '''
    
    criterios = {
        'Entropia': get_mejor_atributo_entropia,
        'Gain ratio': get_mejor_atributo_gain_ratio,
        'Impurity reduction': get_mejor_atributo_impurity_reduction
    }

    ArbolDecisionManual = ArbolDecision()
    resultados = {}

    for criterio, funcion_criterio in criterios.items():
        ArbolDecisionManual.entrenar(X, Y, funcion_criterio)
        Y_predicho = ArbolDecisionManual.predecir(X_val)
        resultados[criterio] = get_accuracy_precision_recall_f1(Y_val, Y_predicho, objetivo=0)

    return resultados

def plot_metrics(resultados, max_range_split):
    funciones_seleccion = resultados.keys()
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1']

    def convert_results_to_matrix(resultados):
        return {
            'Accuracy': [resultados[criterio][0] for criterio in funciones_seleccion],
            'Precision': [resultados[criterio][1] for criterio in funciones_seleccion],
            'Recall': [resultados[criterio][2] for criterio in funciones_seleccion],
            'F1': [resultados[criterio][3] for criterio in funciones_seleccion]
        }

    results_matrix = convert_results_to_matrix(resultados)

    x = np.arange(len(metricas))  
    width = 0.2  

    _, ax = plt.subplots(figsize=(7, 5))

    for i, metrica in enumerate(metricas):
        ax.bar(x[i] - width, results_matrix[metrica][0], width, color='skyblue', label='Entropía' if i == 0 else "")
        ax.bar(x[i], results_matrix[metrica][1], width, color='salmon', label='Gain Ratio' if i == 0 else "")
        ax.bar(x[i] + width, results_matrix[metrica][2], width, color='lightgreen', label='Impurity Reduction' if i == 0 else "")

    ax.set_xlabel('Métrica')
    ax.set_ylabel('Valor')
    ax.set_title(f'max_range_split = {max_range_split}')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas)

    ax.set_ylim(0.75, 1)

    handles = [plt.Rectangle((0, 0), 1, 1, color='skyblue'), plt.Rectangle((0, 0), 1, 1, color='salmon'), plt.Rectangle((0, 0), 1, 1, color='lightgreen')]

    ax.legend(handles, ['Entropía', 'Gain Ratio', 'Impurity reduction'], title='Criterio')

    plt.show()

def plot_metrics_comparativa(results_sin_dropear, results_con_dropear, max_range_split):
    metrics_names = ['Accuracy', 'F1']
    colores = ['skyblue', 'salmon']
    
    accuracy_sin_dropear, _, _, f1_sin_dropear = results_sin_dropear
    accuracy_con_dropear, _, _, f1_con_dropear = results_con_dropear

    resultados_sin_dropear = [accuracy_sin_dropear, f1_sin_dropear]
    resultados_con_dropear = [accuracy_con_dropear, f1_con_dropear]

    x = np.arange(len(metrics_names))  
    width = 0.35 

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - width / 2, resultados_sin_dropear, width, color=colores[0], label='Sin dropear')
    ax.bar(x + width / 2, resultados_con_dropear, width, color=colores[1], label='Con dropear')

    ax.set_xticks(x)  
    ax.set_xticklabels(metrics_names) 
    ax.set_ylim(0.75, 1)
    
    ax.set_xlabel('Métrica')
    ax.set_ylabel('Valor')
    ax.set_title(f'max_range_split={max_range_split}')
    ax.legend()

    plt.show()

def plot_accuracies_and_f1s(accuracies, f1s):
    num_modelos = len(accuracies)  
    x = np.arange(num_modelos) 
    width = 0.35 
    
    _, ax = plt.subplots(figsize=(7, 5))

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')

    bars2 = ax.bar(x + width/2, f1s, width, label='F1')

    ax.set_xlabel('Modelo')
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Accuracy y F1 por Modelo')
    ax.set_xticks(x)
    ax.set_xticklabels(['Predictor Simple', 'Manual', 'Árbol Librería', 'Random Forest'])
    ax.set_ylim(0.5, 1)
    ax.legend()

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords='offset points',
                        ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(Y_real, Y_predicho):
    plt.figure(figsize=(7, 5))

    sns.heatmap(confusion_matrix(Y_real, Y_predicho), annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0, 1])

    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    plt.title('Matriz de Confusión')

    plt.show()

