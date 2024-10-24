import torch

class Entrenador:
    def __init__(self, modelo, dataloader, criterion, optimizer): # se recuerda que el learning rate "lr" viene dentro de "optimizer"
        self.modelo = modelo
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
    

    def entrenar(self): # entrena 1 época

        losses = 0.0

        for batch_x, batch_y in self.dataloader:
            self.modelo.train() # modo entrenamiento
            self.optimizer.zero_grad() # gradientes en 0
            outputs = self.modelo(batch_x).squeeze(1)
            loss = self.criterion(outputs, batch_y)
            loss.backward() # backpropagation
            self.optimizer.step()

            losses += loss.item()

        return loss / len(self.dataloader)
    
    def evaluar_loss(self, dataloader):
        losses = 0.0

        self.modelo.eval() # modo evaluación
        with torch.no_grad():
            for x, y in dataloader:
                outputs = self.modelo(x).squeeze(1)
                loss = self.criterion(outputs, y)

                losses += loss.item()
    
        return losses / len(dataloader)
    
    # Accuracy en caso de que el modelo tenga una sola salida
    def evaluar_accuracy_singular(self, dataloader):
        correctas = 0
        total_muestras = 0
        
        self.modelo.eval() # modo evaluación
        with torch.no_grad(): 
            for x_batch, y_batch in dataloader:
                pred = self.modelo(x_batch).squeeze(1)

                pred_binarias = (pred >= 0.5)
                correctas += (pred_binarias == y_batch).sum().item()
                total_muestras += len(y_batch)
        
        accuracy = correctas / total_muestras

        return accuracy

    # Accuracy en caso de que el modelo tenga 2 salidas
    def evaluar_accuracy_dual(self, dataloader):
        correctas = 0
        total_muestras = 0
        
        self.modelo.eval() # modo evaluación
        with torch.no_grad(): 
            for x_batch, y_batch in dataloader:
                pred = self.modelo(x_batch)

                pred_clase = torch.argmax(pred, dim=1) 
                y_batch_clase = torch.argmax(y_batch, dim=1)   

                correctas += (pred_clase == y_batch_clase).sum().item()
                total_muestras += len(y_batch)
        
        accuracy = correctas / total_muestras
        return accuracy

