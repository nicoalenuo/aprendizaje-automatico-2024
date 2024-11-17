import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Entrenador:
    def __init__(self, modelo, dataloader, criterion, optimizer, dos_salidas=False): # se recuerda que el learning rate "lr" viene dentro de "optimizer"
        self.modelo = modelo
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.dos_salidas = dos_salidas
    

    def entrenar(self): # entrena 1 época

        self.modelo.train() # modo entrenamiento
        for batch_x, batch_y in self.dataloader:
            self.optimizer.zero_grad() # gradientes en 0
            outputs = self.modelo(batch_x).squeeze(1)
            loss = self.criterion(outputs, batch_y)
            loss.backward() # backpropagation
            self.optimizer.step()
    
    def evaluar_loss(self, dataloader):
        losses = 0.0

        self.modelo.eval() # modo evaluación
        with torch.no_grad():
            for x, y in dataloader:
                outputs = self.modelo(x).squeeze(1)
                loss = self.criterion(outputs, y)

                losses += loss.item()
    
        return losses / len(dataloader)
    
    def evaluar_accuracy(self, dataloader):
        '''
        Toma un dataloader y devuelve la precisión del modelo en el conjunto de datos
        usando el modelo entrenado
        '''

        valores_predichos = []
        valores_reales = []
        
        self.modelo.eval() # modo evaluación
        with torch.no_grad(): 
            for x_batch, y_batch in dataloader:
                pred = self.modelo(x_batch).squeeze(1)

                # Si tiene 2 salidas, se toma como predicción la que de el valor más alto
                if self.dos_salidas:
                    pred_binario = torch.argmax(pred, dim=1) 
                    y_batch_binario = torch.argmax(y_batch, dim=1)
                # Si tiene 1 salida, se espera que este entre 0 y 1, y se toma como predicción 1 si es mayor o igual a 0.5, 0 en caso contrario
                else:
                    pred_binario = torch.tensor([1 if p >= 0.5 else 0 for p in pred])
                    y_batch_binario = y_batch.float()
                
                valores_predichos.extend(pred_binario.to("cpu"))
                valores_reales.extend(y_batch_binario.to("cpu"))
        
        return accuracy_score(valores_reales, valores_predichos)

    def evaluar_metricas(self, dataloader):
        '''
        Toma un dataloader y devuelve diferentes metricas del modelo en el conjunto de datos
        usando el modelo entrenado
        '''

        valores_predichos = []
        valores_reales = []
        
        self.modelo.eval() # modo evaluación
        with torch.no_grad(): 
            for x_batch, y_batch in dataloader:
                pred = self.modelo(x_batch).squeeze(1)

                # Si tiene 2 salidas, se toma como predicción la que de el valor más alto
                if self.dos_salidas:
                    pred_binario = torch.argmax(pred, dim=1) 
                    y_batch_binario = torch.argmax(y_batch, dim=1)
                # Si tiene 1 salida, se espera que este entre 0 y 1, y se toma como predicción 1 si es mayor o igual a 0.5, 0 en caso contrario
                else:
                    pred_binario = torch.tensor([1 if p >= 0.5 else 0 for p in pred])
                    y_batch_binario = y_batch.float()
                
                valores_predichos.extend(pred_binario.to("cpu"))
                valores_reales.extend(y_batch_binario.to("cpu"))
        
        return (
            accuracy_score(valores_reales, valores_predichos),
            precision_score(valores_reales, valores_predichos),
            recall_score(valores_reales, valores_predichos),
            f1_score(valores_reales, valores_predichos, pos_label=0)
        )


