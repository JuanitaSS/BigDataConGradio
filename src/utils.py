import os
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
registrador = logging.getLogger(__name__)

ETIQUETAS_CLASES = {
    "colon_aca": 0,
    "colon_n":   1,
    "lung_aca":  2,
    "lung_n":    3,
    "lung_scc":  4,
}

NOMBRES_CLASES = {v: k for k, v in ETIQUETAS_CLASES.items()}

DESCRIPCIONES_CLASES = {
    "colon_aca": "Adenocarcinoma de Colon (Maligno)",
    "colon_n":   "Tejido Benigno de Colon",
    "lung_aca":  "Adenocarcinoma de Pulmón (Maligno)",
    "lung_n":    "Tejido Benigno de Pulmón",
    "lung_scc":  "Carcinoma de Células Escamosas de Pulmón (Maligno)",
}

CLASES_MALIGNAS = {"colon_aca", "lung_aca", "lung_scc"}

TAMANO_IMAGEN = 224
MEDIA         = [0.485, 0.456, 0.406]
DESVIACION    = [0.229, 0.224, 0.225]

transformacion_entrenamiento = transforms.Compose([
    transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIACION),
])

transformacion_validacion = transforms.Compose([
    transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIACION),
])

transformacion_inferencia = transformacion_validacion


class DatasetHistopatologico(Dataset):
    def __init__(self, marco: pd.DataFrame, transformacion=None):
        self.marco          = marco.reset_index(drop=True)
        self.transformacion = transformacion

    def __len__(self):
        return len(self.marco)

    def __getitem__(self, indice):
        fila   = self.marco.iloc[indice]
        imagen = Image.open(fila["path"]).convert("RGB")
        if self.transformacion:
            imagen = self.transformacion(imagen)
        return imagen, int(fila["label_idx"])


def extraer_dataset(raiz_datos: str) -> pd.DataFrame:
    registrador.info("EXTRACT — escaneando dataset en: %s", raiz_datos)
    registros  = []
    ruta_datos = Path(raiz_datos)
    if not ruta_datos.exists():
        raise FileNotFoundError(f"Raíz de datos no encontrada: {raiz_datos}")

    for extension in ("*.jpeg", "*.jpg"):
        for ruta_imagen in ruta_datos.rglob(extension):
            etiqueta = ruta_imagen.parent.name
            registros.append({
                "path":     str(ruta_imagen),
                "filename": ruta_imagen.name,
                "label":    etiqueta,
                "organ":    "lung" if "lung" in etiqueta else "colon",
            })

    marco = pd.DataFrame(registros)
    registrador.info("   Encontradas %d imágenes en %d clases", len(marco), marco["label"].nunique())
    return marco


def _calcular_hash(ruta: str) -> Optional[str]:
    try:
        with open(ruta, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def _es_imagen_valida(ruta: str, tamano_minimo: int = 32) -> Tuple[bool, str]:
    try:
        imagen = Image.open(ruta)
        imagen.verify()
        imagen = Image.open(ruta)
        ancho, alto = imagen.size
        if ancho < tamano_minimo or alto < tamano_minimo:
            return False, f"demasiado pequeña ({ancho}x{alto})"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def limpiar_dataset(marco: pd.DataFrame, tamano_minimo: int = 32) -> Tuple[pd.DataFrame, pd.DataFrame]:
    registrador.info("CLEAN — iniciando con %d registros", len(marco))
    registros_eliminados = []

    conocidas            = set(ETIQUETAS_CLASES.keys())
    mascara_desconocidas = ~marco["label"].isin(conocidas)
    if mascara_desconocidas.any():
        registros_eliminados.append(marco[mascara_desconocidas].assign(razon="etiqueta_desconocida"))
        marco = marco[~mascara_desconocidas].copy()

    validez = [_es_imagen_valida(p, tamano_minimo) for p in tqdm(marco["path"], desc="Validando imágenes")]
    banderas_validas, razones = zip(*validez) if validez else ([], [])
    marco["_valida"] = banderas_validas
    marco["_razon"]  = razones
    mascara_invalidas = ~marco["_valida"]
    if mascara_invalidas.any():
        registros_eliminados.append(marco[mascara_invalidas].assign(razon=marco.loc[mascara_invalidas, "_razon"]))
        marco = marco[marco["_valida"]].copy()
    marco.drop(columns=["_valida", "_razon"], inplace=True)

    registrador.info("   Calculando hashes para deduplicación...")
    marco["hash"]      = [_calcular_hash(p) for p in tqdm(marco["path"], desc="Hasheando")]
    mascara_duplicados = marco.duplicated(subset="hash", keep="first")
    if mascara_duplicados.any():
        registros_eliminados.append(marco[mascara_duplicados].assign(razon="duplicado"))
        marco = marco[~mascara_duplicados].copy()
    marco.drop(columns=["hash"], inplace=True)

    marco["label_idx"]    = marco["label"].map(ETIQUETAS_CLASES)
    marco["description"]  = marco["label"].map(DESCRIPCIONES_CLASES)
    marco["is_malignant"] = marco["label"].isin(CLASES_MALIGNAS)

    marco_eliminado = pd.concat(registros_eliminados, ignore_index=True) if registros_eliminados else pd.DataFrame()
    registrador.info("CLEAN completo — %d imágenes útiles, %d eliminadas", len(marco), len(marco_eliminado))
    return marco.reset_index(drop=True), marco_eliminado


def crear_cargadores(
    marco: pd.DataFrame,
    tamano_lote:       int   = 32,
    num_workers:       int   = 4,
    proporcion_val:    float = 0.15,
    proporcion_prueba: float = 0.15,
    semilla:           int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registrador.info("LOAD — creando splits estratificados train/val/test")

    marco_entrval, marco_prueba = train_test_split(
        marco, test_size=proporcion_prueba, stratify=marco["label_idx"], random_state=semilla
    )
    proporcion_val_relativa = proporcion_val / (1 - proporcion_prueba)
    marco_entrenamiento, marco_validacion = train_test_split(
        marco_entrval, test_size=proporcion_val_relativa,
        stratify=marco_entrval["label_idx"], random_state=semilla
    )

    registrador.info("   Entrenamiento: %d | Val: %d | Prueba: %d",
                     len(marco_entrenamiento), len(marco_validacion), len(marco_prueba))

    cargador_entrenamiento = DataLoader(
        DatasetHistopatologico(marco_entrenamiento, transformacion_entrenamiento),
        batch_size=tamano_lote, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    cargador_validacion = DataLoader(
        DatasetHistopatologico(marco_validacion, transformacion_validacion),
        batch_size=tamano_lote, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    cargador_prueba = DataLoader(
        DatasetHistopatologico(marco_prueba, transformacion_validacion),
        batch_size=tamano_lote, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return (cargador_entrenamiento, cargador_validacion, cargador_prueba,
            marco_entrenamiento, marco_validacion, marco_prueba)


def ejecutar_pipeline_etl(raiz_datos: str, tamano_lote: int = 32, num_workers: int = 4):
    registrador.info("=" * 60)
    registrador.info(" INICIANDO PIPELINE ETL")
    registrador.info("=" * 60)

    marco_crudo              = extraer_dataset(raiz_datos)
    marco_limpio, marco_eliminado = limpiar_dataset(marco_crudo)

    if not marco_eliminado.empty:
        ruta_eliminados = Path(raiz_datos).parent / "etl_registros_eliminados.csv"
        marco_eliminado.to_csv(ruta_eliminados, index=False)

    ruta_resumen = Path(raiz_datos).parent / "etl_resumen_limpio.csv"
    marco_limpio.to_csv(ruta_resumen, index=False)

    (cargador_entrenamiento, cargador_validacion, cargador_prueba,
     marco_entrenamiento, marco_validacion, marco_prueba) = crear_cargadores(
        marco_limpio, tamano_lote=tamano_lote, num_workers=num_workers
    )

    registrador.info("=" * 60)
    registrador.info(" PIPELINE ETL COMPLETO")
    registrador.info("=" * 60)

    return (cargador_entrenamiento, cargador_validacion, cargador_prueba,
            marco_entrenamiento, marco_validacion, marco_prueba, marco_limpio)


def graficar_distribucion(marco: pd.DataFrame, ruta_guardado: Optional[str] = None):
    conteos = marco["label"].value_counts().rename(index=DESCRIPCIONES_CLASES)
    figura, eje = plt.subplots(figsize=(10, 5))
    sns.barplot(x=conteos.values, y=conteos.index, palette="viridis", ax=eje)
    eje.set_title("Distribución de Clases — Dataset Limpio", fontsize=14, fontweight="bold")
    eje.set_xlabel("Número de Imágenes")
    eje.set_ylabel("")
    for barra, valor in zip(eje.patches, conteos.values):
        eje.text(barra.get_width() + 50, barra.get_y() + barra.get_height() / 2,
                 f"{valor:,}", va="center", fontsize=10)
    plt.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=120, bbox_inches="tight")
    return figura


def generar_reporte_etl(marco_limpio: pd.DataFrame, marco_eliminado: pd.DataFrame) -> str:
    total_crudo = len(marco_limpio) + len(marco_eliminado)
    reporte  = "# Reporte ETL — Lung & Colon Cancer Dataset\n\n"
    reporte += "| Métrica | Valor |\n|---|---|\n"
    reporte += f"| Imágenes brutas | {total_crudo:,} |\n"
    reporte += f"| Eliminadas | {len(marco_eliminado):,} |\n"
    reporte += f"| Limpias | {len(marco_limpio):,} |\n"
    reporte += f"| Retención | {len(marco_limpio)/total_crudo*100:.1f}% |\n"
    reporte += f"| Clases | {marco_limpio['label'].nunique()} |\n\n"
    reporte += "## Distribución por Clase\n\n| Clase | Descripción | N | % |\n|---|---|---|---|\n"
    for etiqueta, conteo in marco_limpio["label"].value_counts().items():
        descripcion = DESCRIPCIONES_CLASES.get(etiqueta, etiqueta)
        porcentaje  = conteo / len(marco_limpio) * 100
        reporte += f"| `{etiqueta}` | {descripcion} | {conteo:,} | {porcentaje:.1f}% |\n"
    return reporte
