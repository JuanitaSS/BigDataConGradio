import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from src.utils import NOMBRES_CLASES, DESCRIPCIONES_CLASES, CLASES_MALIGNAS, transformacion_inferencia
from src.train import RedCancer, NUM_CLASES

registrador = logging.getLogger(__name__)

_cache_modelo:      Optional[RedCancer]     = None
_cache_dispositivo: Optional[torch.device] = None


def obtener_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cargar_modelo(ruta_modelo: str, forzar_recarga: bool = False) -> Tuple[RedCancer, torch.device]:
    global _cache_modelo, _cache_dispositivo

    if _cache_modelo is not None and not forzar_recarga:
        return _cache_modelo, _cache_dispositivo

    dispositivo   = obtener_dispositivo()
    modelo        = RedCancer(num_clases=NUM_CLASES, preentrenado=False)
    punto_control = torch.load(ruta_modelo, map_location=dispositivo)
    estado        = punto_control.get("model_state", punto_control)

    estado_remapeado = {
        clave.replace("backbone.", "esqueleto.").replace("classifier.", "clasificador."): valor
        for clave, valor in estado.items()
    }

    modelo.load_state_dict(estado_remapeado)
    modelo.to(dispositivo)
    modelo.eval()
    _cache_modelo      = modelo
    _cache_dispositivo = dispositivo
    registrador.info("Modelo cargado desde %s en %s", ruta_modelo, dispositivo)
    return modelo, dispositivo


class GradCAM:
    def __init__(self, modelo: RedCancer):
        self.modelo       = modelo
        self.gradientes   = None
        self.activaciones = None
        self._ganchos     = []
        self._registrar_ganchos()

    def _registrar_ganchos(self):
        capa_objetivo = self.modelo.esqueleto.features[-1]

        def gancho_adelante(*args):
            self.activaciones = args[-1].detach()

        def gancho_atras(*args):
            self.gradientes = args[-1][0].detach()

        self._ganchos.append(capa_objetivo.register_forward_hook(gancho_adelante))
        self._ganchos.append(capa_objetivo.register_full_backward_hook(gancho_atras))

    def generar(self, tensor_entrada: torch.Tensor, indice_clase: int) -> np.ndarray:
        self.modelo.zero_grad()
        salida  = self.modelo(tensor_entrada)
        perdida = salida[0, indice_clase]
        perdida.backward()

        pesos = self.gradientes.mean(dim=[2, 3], keepdim=True)
        mapa  = (pesos * self.activaciones).sum(dim=1, keepdim=True)
        mapa  = F.relu(mapa)
        mapa  = F.interpolate(mapa, size=(224, 224), mode="bilinear", align_corners=False)
        mapa  = mapa.squeeze().cpu().numpy()
        mapa  = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)
        return mapa

    def eliminar_ganchos(self):
        for gancho in self._ganchos:
            gancho.remove()


def superponer_gradcam(imagen_original: np.ndarray, mapa: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    mapa_calor    = cv2.applyColorMap(np.uint8(255 * mapa), cv2.COLORMAP_JET)
    mapa_calor    = cv2.cvtColor(mapa_calor, cv2.COLOR_BGR2RGB)
    superposicion = (alpha * mapa_calor + (1 - alpha) * imagen_original).astype(np.uint8)
    return superposicion


def predecir_imagen(imagen: Image.Image, ruta_modelo: str) -> Dict:
    modelo, dispositivo = cargar_modelo(ruta_modelo)

    imagen_rgb    = imagen.convert("RGB")
    tensor_imagen = transformacion_inferencia(imagen_rgb).unsqueeze(0).to(dispositivo)
    imagen_np     = np.array(imagen_rgb.resize((224, 224)))

    with torch.no_grad():
        logits = modelo(tensor_imagen)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    indice_predicho    = int(probs.argmax())
    etiqueta_predicha  = NOMBRES_CLASES[indice_predicho]
    confianza_predicha = float(probs[indice_predicho])
    descripcion        = DESCRIPCIONES_CLASES[etiqueta_predicha]
    es_maligno         = etiqueta_predicha in CLASES_MALIGNAS

    gradcam = GradCAM(modelo)
    tensor_imagen.requires_grad_(True)
    mapa          = gradcam.generar(tensor_imagen, indice_predicho)
    gradcam.eliminar_ganchos()
    superposicion = superponer_gradcam(imagen_np, mapa)

    figura, ejes = plt.subplots(1, 3, figsize=(12, 4))
    figura.patch.set_facecolor("#0d1117")
    for eje in ejes:
        eje.set_facecolor("#0d1117")

    ejes[0].imshow(imagen_np)
    ejes[0].set_title("Imagen Original", color="white", fontsize=11)
    ejes[0].axis("off")

    ejes[1].imshow(mapa, cmap="jet")
    ejes[1].set_title("Grad-CAM", color="white", fontsize=11)
    ejes[1].axis("off")

    ejes[2].imshow(superposicion)
    ejes[2].set_title("Superposición", color="white", fontsize=11)
    ejes[2].axis("off")

    color    = "#ff4757" if es_maligno else "#2ed573"
    etiqueta = "MALIGNO" if es_maligno else "BENIGNO"
    figura.suptitle(
        f"{etiqueta} — {descripcion}  ({confianza_predicha*100:.1f}%)",
        color=color, fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    dict_probabilidades = {DESCRIPCIONES_CLASES[NOMBRES_CLASES[i]]: float(probs[i]) for i in range(NUM_CLASES)}

    texto_reporte = (
        f"RESULTADO DEL DIAGNÓSTICO\n"
        f"{'='*40}\n"
        f"Clasificación:   {descripcion}\n"
        f"Malignidad:      {'MALIGNO' if es_maligno else 'BENIGNO'}\n"
        f"Confianza:       {confianza_predicha*100:.2f}%\n\n"
        f"Probabilidades por clase:\n"
    )
    for clase, prob in sorted(dict_probabilidades.items(), key=lambda x: -x[1]):
        barra = "█" * int(prob * 20)
        texto_reporte += f"  {clase:<45} {prob*100:5.1f}%  {barra}\n"

    return {
        "label":         etiqueta_predicha,
        "description":   descripcion,
        "confidence":    confianza_predicha,
        "is_malignant":  es_maligno,
        "probabilities": dict_probabilidades,
        "gradcam_fig":   figura,
        "report_text":   texto_reporte,
    }
