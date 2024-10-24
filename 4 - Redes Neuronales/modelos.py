import torch.nn as nn

class Modelo_2(nn.Module):

    def __init__(self, cant_caracteristicas):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(cant_caracteristicas, 2),
        )

    def forward(self, x):
        return self.red(x)

class Modelo_3(nn.Module):

    def __init__(self, cant_caracteristicas):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(cant_caracteristicas, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.red(x)

class Modelo_4(nn.Module):
    
    def __init__(self, input_size, hidden_size=16):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.red(x)
    
# ---- Parte 5 ----

class Modelo_5_1(nn.Module):
    def __init__(self, tamanio_entrada):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(tamanio_entrada, 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.red(x)

