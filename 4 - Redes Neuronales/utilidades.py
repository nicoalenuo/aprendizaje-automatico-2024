import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---- Data Loader ----

def crear_data_loader(X, Y, batch_size, dos_salidas=False, device='cpu'):
    '''
    Crea y devuelve un dataloader con el conjunto de entrada X y el conjunto de salida Y

    dos_salidas: True si el modelo a entrenar tiene 2 salidas, False si tiene 1 salida
    '''

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    if dos_salidas:
        Y_tensor = torch.tensor([[1, 0] if y == 0 else [0, 1] for y in Y], dtype=torch.float32).to(device)
    else:
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, Y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# ---- Gráficas ----


def graficar_metricas(evolucion_loss_train, evolucion_loss_validacion, evolucion_accuracy_train, evolucion_accuracy_validacion, tasa_aprendizaje=None):
    # Graficar Loss
    plt.figure(figsize=(14, 5))
    
    if tasa_aprendizaje is not None:
        plt.suptitle(f"Tasa de aprendizaje: {tasa_aprendizaje}", fontsize=16)  # Título general

    plt.subplot(1, 2, 1)  
    plt.plot(evolucion_loss_train, label='Pérdida entrenamiento', color='blue')
    plt.plot(evolucion_loss_validacion, label='Pérdida validación', color='orange')
    plt.title('Evolución de pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()

    # Graficar Accuracy
    plt.subplot(1, 2, 2) 
    plt.plot(evolucion_accuracy_train, label='Precisión entrenamiento', color='green')
    plt.plot(evolucion_accuracy_validacion, label='Precisión validación', color='red')
    plt.title('Evolución de la precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(Y_real, Y_predicho):
    plt.figure(figsize=(7, 5))

    sns.heatmap(confusion_matrix(Y_real, Y_predicho), annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0, 1])

    plt.ylabel('Clase verdadera')
    plt.xlabel('Clase predicha')
    plt.title('Matriz de confusión')

    plt.show()
