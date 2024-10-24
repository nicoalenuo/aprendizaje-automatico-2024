import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---- Data Loaders ----

def crear_data_loader_singular(X, Y, batch_size, device):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float32).to(device)

    joint_dataset = TensorDataset(X_tensor, Y_tensor)
    return DataLoader(joint_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

def crear_data_loader_dual(X, Y, batch_size, device):
    X_tensor = torch.as_tensor(X, dtype=torch.float32).to(device)
    Y_par = [[1, 0] if y == 0 else [0, 1] for y in Y]
    Y_tensor = torch.tensor(Y_par, dtype=torch.float32).to(device)

    joint_dataset = TensorDataset(X_tensor, Y_tensor)
    return DataLoader(joint_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# ---- Gráficas ----


def graficar_metricas(evolucion_loss_train, evolucion_loss_validacion, evolucion_accuracy_train, evolucion_accuracy_validacion):
    # Graficar Loss
    plt.figure(figsize=(14, 6))

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

