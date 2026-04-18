import os
import time
import copy
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils import ETIQUETAS_CLASES, DESCRIPCIONES_CLASES, NOMBRES_CLASES

registrador = logging.getLogger(__name__)
NUM_CLASES  = len(ETIQUETAS_CLASES)


class RedCancer(nn.Module):
    def __init__(self, num_clases: int = NUM_CLASES, abandono: float = 0.4, preentrenado: bool = True):
        super().__init__()
        pesos    = models.EfficientNet_B0_Weights.DEFAULT if preentrenado else None
        esqueleto = models.efficientnet_b0(weights=pesos)

        for _, parametro in esqueleto.named_parameters():
            parametro.requires_grad = False
        for nombre, parametro in esqueleto.named_parameters():
            if any(f"features.{i}" in nombre for i in [6, 7, 8]):
                parametro.requires_grad = True

        caracteristicas_entrada = esqueleto.classifier[1].in_features
        esqueleto.classifier    = nn.Identity()

        self.esqueleto    = esqueleto
        self.clasificador = nn.Sequential(
            nn.BatchNorm1d(caracteristicas_entrada),
            nn.Dropout(abandono),
            nn.Linear(caracteristicas_entrada, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(abandono / 2),
            nn.Linear(512, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        caracteristicas = self.esqueleto(x)
        return self.clasificador(caracteristicas)


def obtener_modelo(dispositivo: torch.device, **kwargs) -> RedCancer:
    modelo = RedCancer(**kwargs)
    return modelo.to(dispositivo)


def entrenar_epoca(
    modelo:      nn.Module,
    cargador:    DataLoader,
    criterio:    nn.Module,
    optimizador: optim.Optimizer,
    dispositivo: torch.device,
    escalador,
) -> Tuple[float, float]:
    modelo.train()
    perdida_acumulada = 0.0
    correctos = 0
    total     = 0

    for imagenes, etiquetas in tqdm(cargador, desc="  Entrenando", leave=False):
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        optimizador.zero_grad()

        with torch.cuda.amp.autocast(enabled=dispositivo.type == "cuda"):
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)

        escalador.scale(perdida).backward()
        escalador.step(optimizador)
        escalador.update()

        perdida_acumulada += perdida.item() * imagenes.size(0)
        predicciones = salidas.argmax(dim=1)
        correctos   += (predicciones == etiquetas).sum().item()
        total       += imagenes.size(0)

    return perdida_acumulada / total, correctos / total


@torch.no_grad()
def evaluar(
    modelo:      nn.Module,
    cargador:    DataLoader,
    criterio:    nn.Module,
    dispositivo: torch.device,
) -> Tuple[float, float]:
    modelo.eval()
    perdida_acumulada = 0.0
    correctos = 0
    total     = 0

    for imagenes, etiquetas in tqdm(cargador, desc="  Validando", leave=False):
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)

        perdida_acumulada += perdida.item() * imagenes.size(0)
        predicciones = salidas.argmax(dim=1)
        correctos   += (predicciones == etiquetas).sum().item()
        total       += imagenes.size(0)

    return perdida_acumulada / total, correctos / total


def entrenar(
    modelo:                 nn.Module,
    cargador_entrenamiento: DataLoader,
    cargador_validacion:    DataLoader,
    dispositivo:            torch.device,
    epocas:                 int   = 20,
    tasa_aprendizaje:       float = 1e-3,
    decaimiento_peso:       float = 1e-4,
    paciencia:              int   = 5,
    ruta_guardado:          str   = "model/lung_colon_cnn.pth",
) -> Dict:
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)

    criterio    = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizador = optim.AdamW(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=tasa_aprendizaje, weight_decay=decaimiento_peso
    )
    programador = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas, eta_min=1e-6)
    escalador   = torch.cuda.amp.GradScaler(enabled=dispositivo.type == "cuda")

    historial = {
        "perdida_entrenamiento":   [],
        "perdida_validacion":      [],
        "precision_entrenamiento": [],
        "precision_validacion":    [],
    }
    mejor_precision_val = 0.0
    mejor_estado        = None
    epocas_sin_mejora   = 0

    registrador.info("Iniciando entrenamiento — %d epocas | dispositivo: %s", epocas, dispositivo)

    for epoca in range(1, epocas + 1):
        t0 = time.time()
        perdida_entr, precision_entr = entrenar_epoca(
            modelo, cargador_entrenamiento, criterio, optimizador, dispositivo, escalador)
        perdida_val, precision_val   = evaluar(modelo, cargador_validacion, criterio, dispositivo)
        programador.step()

        historial["perdida_entrenamiento"].append(perdida_entr)
        historial["perdida_validacion"].append(perdida_val)
        historial["precision_entrenamiento"].append(precision_entr)
        historial["precision_validacion"].append(precision_val)

        transcurrido = time.time() - t0
        registrador.info(
            "Epoch %3d/%d | Train Loss %.4f Acc %.4f | Val Loss %.4f Acc %.4f | LR %.2e | %.1fs",
            epoca, epocas, perdida_entr, precision_entr, perdida_val, precision_val,
            optimizador.param_groups[0]["lr"], transcurrido
        )

        if precision_val > mejor_precision_val:
            mejor_precision_val = precision_val
            mejor_estado        = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora   = 0
            torch.save({"model_state": mejor_estado, "epoch": epoca, "val_acc": precision_val}, ruta_guardado)
            registrador.info("   Nuevo mejor modelo guardado (val_acc=%.4f)", precision_val)
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= paciencia:
                registrador.info("   Parada temprana en época %d", epoca)
                break

    if mejor_estado:
        modelo.load_state_dict(mejor_estado)

    return historial


@torch.no_grad()
def evaluacion_completa(modelo: nn.Module, cargador_prueba: DataLoader, dispositivo: torch.device) -> Dict:
    modelo.eval()
    todas_predicciones   = []
    todas_etiquetas      = []
    todas_probabilidades = []

    for imagenes, etiquetas in tqdm(cargador_prueba, desc="Evaluando"):
        imagenes = imagenes.to(dispositivo)
        logits   = modelo(imagenes)
        probs    = torch.softmax(logits, dim=1).cpu().numpy()
        preds    = logits.argmax(dim=1).cpu().numpy()
        todas_predicciones.extend(preds)
        todas_etiquetas.extend(etiquetas.numpy())
        todas_probabilidades.extend(probs)

    todas_predicciones   = np.array(todas_predicciones)
    todas_etiquetas      = np.array(todas_etiquetas)
    todas_probabilidades = np.array(todas_probabilidades)

    nombres_objetivo = [DESCRIPCIONES_CLASES[NOMBRES_CLASES[i]] for i in range(NUM_CLASES)]
    reporte = classification_report(todas_etiquetas, todas_predicciones,
                                    target_names=nombres_objetivo, output_dict=True)
    matriz  = confusion_matrix(todas_etiquetas, todas_predicciones)

    try:
        auc = roc_auc_score(
            np.eye(NUM_CLASES)[todas_etiquetas], todas_probabilidades,
            multi_class="ovr", average="macro"
        )
    except Exception:
        auc = None

    precision = (todas_predicciones == todas_etiquetas).mean()
    registrador.info("Test Accuracy: %.4f | Macro AUC: %s", precision, f"{auc:.4f}" if auc else "N/A")

    return {
        "accuracy":  precision,
        "auc":       auc,
        "report":    reporte,
        "confusion": matriz,
        "preds":     todas_predicciones,
        "labels":    todas_etiquetas,
        "probs":     todas_probabilidades,
    }


def graficar_historial_entrenamiento(historial: Dict, ruta_guardado: Optional[str] = None):
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4))
    epocas_rango = range(1, len(historial["perdida_entrenamiento"]) + 1)

    ejes[0].plot(epocas_rango, historial["perdida_entrenamiento"], label="Entrenamiento", linewidth=2)
    ejes[0].plot(epocas_rango, historial["perdida_validacion"],    label="Validación",    linewidth=2, linestyle="--")
    ejes[0].set_title("Pérdida por Época", fontweight="bold")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Pérdida")
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)

    ejes[1].plot(epocas_rango, historial["precision_entrenamiento"], label="Entrenamiento", linewidth=2)
    ejes[1].plot(epocas_rango, historial["precision_validacion"],    label="Validación",    linewidth=2, linestyle="--")
    ejes[1].set_title("Precisión por Época", fontweight="bold")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Precisión")
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)

    plt.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=120, bbox_inches="tight")
    return figura


def graficar_matriz_confusion(matriz: np.ndarray, ruta_guardado: Optional[str] = None):
    etiquetas = [DESCRIPCIONES_CLASES[NOMBRES_CLASES[i]] for i in range(NUM_CLASES)]
    cortas    = [e.split(" (")[0] for e in etiquetas]

    figura, eje = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues",
        xticklabels=cortas, yticklabels=cortas, ax=eje
    )
    eje.set_title("Matriz de Confusión — Conjunto de Prueba", fontsize=14, fontweight="bold")
    eje.set_ylabel("Etiqueta Real")
    eje.set_xlabel("Predicción")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=120, bbox_inches="tight")
    return figura
