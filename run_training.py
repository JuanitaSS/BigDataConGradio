import sys
import os
import logging
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import ejecutar_pipeline_etl
from src.train import obtener_modelo, entrenar, evaluacion_completa, graficar_historial_entrenamiento, graficar_matriz_confusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
registrador = logging.getLogger(__name__)

RUTA_DATOS   = "data/lung_colon_image_set"
RUTA_MODELO  = "model/lung_colon_cnn.pth"
EPOCAS       = 20
TAMANO_LOTE  = 32
TASA_APREND  = 1e-3


def principal():
    if not Path(RUTA_DATOS).exists():
        registrador.error("Dataset no encontrado en: %s", RUTA_DATOS)
        sys.exit(1)

    registrador.info("=" * 60)
    registrador.info("PASO 1/3 — ETL")
    registrador.info("=" * 60)
    cargadores = ejecutar_pipeline_etl(RUTA_DATOS, tamano_lote=TAMANO_LOTE, num_workers=0)
    cargador_entrenamiento, cargador_validacion, cargador_prueba, \
        marco_entrenamiento, marco_validacion, marco_prueba, marco_limpio = cargadores
    registrador.info("Dataset: %d entrenamiento | %d validación | %d prueba",
                     len(marco_entrenamiento), len(marco_validacion), len(marco_prueba))

    registrador.info("=" * 60)
    registrador.info("PASO 2/3 — ENTRENAMIENTO")
    registrador.info("=" * 60)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registrador.info("Dispositivo: %s", dispositivo)

    modelo = obtener_modelo(dispositivo, preentrenado=True)
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    registrador.info("Parámetros entrenables: %d", entrenables)

    historial = entrenar(
        modelo, cargador_entrenamiento, cargador_validacion, dispositivo,
        epocas=EPOCAS, tasa_aprendizaje=TASA_APREND, ruta_guardado=RUTA_MODELO,
    )

    registrador.info("=" * 60)
    registrador.info("PASO 3/3 — EVALUACIÓN")
    registrador.info("=" * 60)
    resultado_evaluacion = evaluacion_completa(modelo, cargador_prueba, dispositivo)
    registrador.info("Precisión de prueba: %.4f", resultado_evaluacion["accuracy"])
    if resultado_evaluacion["auc"]:
        registrador.info("Macro AUC: %.4f", resultado_evaluacion["auc"])

    os.makedirs("model", exist_ok=True)
    graficar_historial_entrenamiento(historial, "model/historial_entrenamiento.png")
    graficar_matriz_confusion(resultado_evaluacion["confusion"], "model/matriz_confusion.png")
    registrador.info("Gráficas guardadas en model/")
    registrador.info("Modelo guardado en: %s", RUTA_MODELO)


if __name__ == "__main__":
    principal()
