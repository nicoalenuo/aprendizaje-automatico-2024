import pandas as pd
import math
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OBJETIVO = 'cid'
DATASET_FILE = 'data.csv'

class NaieBayesAIDS:

    def __init__(self):
        self.entrenado = False

    def entrenar(self, X_train, m, valores_posibles, atributos_a_categorizar):
        '''
        X_train: DataFrame con los datos de entrenamiento
        m: tamaño equivalente de muestra
        valores_posibles: map {categoria, [valores posibles]} NO incluyendo la categoria objetivo

        se aplica la m-estimacion: P(X|Y) = (e + m*p) / (m + n)
        asumiendo p = 1 / |valores|
        '''

        # ACA HAY QUE CATEGORIZAR LAS VARIABLES USANDO ATRIBUTOS_A_CATEGORIZAR

        # TAMBIEN HAY QUE GUARDAR LOS PUNTOS DE CORTE EN SELF.PUNTOS_CORTE

        probabilidades_0 = {}
        probabilidades_1 = {}
        
        self.totales = len(X_train)
        self.totales_0 = len(X_train[X_train[OBJETIVO] == 0])
        self.totales_1 = len(X_train[X_train[OBJETIVO] == 1])

        for cat in valores_posibles.keys():
            probabilidades_0[cat] = {}
            probabilidades_1[cat] = {}

            # P(X|Y=0)
            for val in valores_posibles[cat]:
                p = 1 / len(valores_posibles[cat])
                e = len(X_train[(X_train[cat] == val) & (X_train[OBJETIVO] == 0)])

                probabilidades_0[cat][val] = (e + m*p) / (m + self.totales_0)

            # P(X|Y=1)
            for val in valores_posibles[cat]:
                p = 1 / len(valores_posibles[cat])
                e = len(X_train[(X_train[cat] == val) & (X_train[OBJETIVO] == 1)])

                probabilidades_1[cat][val] = (e + m*p) / (m + self.totales_1)

        self.probabilidades_0 = probabilidades_0
        self.probabilidades_1 = probabilidades_1
        self.entrenado = True


    def predecir_entrada(self, X, threshold=0.5):
        #CATEGORIZAR X USANDO SELF.PUNTOS_CORTE

        res_0 = self.totales_0 / self.totales
        res_1 = self.totales_1 / self.totales

        for cat in X.columns.difference([OBJETIVO]):
            res_0 *= self.probabilidades_0[cat][X[cat].iloc[0]]
            res_1 *= self.probabilidades_1[cat][X[cat].iloc[0]]
        
        return 0 if res_0 > res_1 else 1

    
    def predecir(self, X, threshold=0.5):
        if (not self.entrenado):
            raise Exception("El modelo no ha sido entrenado")

        Y_predicho = [self.predecir_entrada(X.iloc[[i]], threshold) for i in range(len(X))]

        return Y_predicho