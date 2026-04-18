import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.predict import predecir_imagen
from app.sms import enviar_sms_diagnostico, formatear_vista_previa

registrador = logging.getLogger(__name__)

RUTA_MODELO = os.getenv("MODEL_PATH", "model/lung_colon_cnn.pth")
RUTA_DATOS  = os.getenv("DATA_PATH",  "data/lung_colon_image_set")

TEMA = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#050a0e",
    body_text_color="#c9d1d9",
    block_background_fill="#0d1117",
    block_border_color="#21262d",
    block_label_text_color="#8b949e",
    input_background_fill="#161b22",
    button_primary_background_fill="#00b4d8",
    button_primary_background_fill_hover="#0096c7",
    button_primary_text_color="#050a0e",
)

CSS_PERSONALIZADO = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

.gradio-container { max-width: 1280px !important; margin: 0 auto; }

#cabecera-app {
  background: linear-gradient(135deg, #050a0e 0%, #0d2137 50%, #050a0e 100%);
  border-bottom: 1px solid #00b4d8;
  padding: 28px 32px 18px;
  margin-bottom: 24px;
}
#cabecera-app h1 {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.8rem;
  color: #00b4d8;
  letter-spacing: 0.04em;
  margin: 0 0 4px;
  text-shadow: 0 0 24px #00b4d860;
}
#cabecera-app p {
  color: #8b949e;
  font-size: 0.85rem;
  margin: 0;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.tarjeta-resultado {
  border-radius: 10px;
  padding: 18px 22px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.92rem;
  line-height: 1.7;
  margin-top: 10px;
}
.maligno { background: #1a0505; border: 1px solid #ff4757; color: #ff6b81; }
.benigno { background: #051a0a; border: 1px solid #2ed573; color: #7bed9f; }

.contenedor-barra-prob { margin-bottom: 6px; }
.etiqueta-prob { font-size: 0.78rem; color: #8b949e; margin-bottom: 2px; }
.pista-barra-prob {
  background: #21262d;
  border-radius: 4px;
  height: 12px;
  position: relative;
  overflow: hidden;
}
.relleno-barra-prob {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

.tab-nav button {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

.seccion-sms {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px;
}

.aviso {
  background: #1a1200;
  border: 1px solid #d4a017;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 0.78rem;
  color: #d4a017;
  margin-top: 16px;
}

/* ── Acerca de: página profesional ─────────── */
.about-page {
  font-family: 'IBM Plex Sans', sans-serif;
  color: #c9d1d9;
  padding: 8px 0 32px;
}

/* Hero strip */
.about-hero {
  background: linear-gradient(120deg, #020d18 0%, #071d30 40%, #020d18 100%);
  border: 1px solid #1a3a52;
  border-radius: 12px;
  padding: 36px 40px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.about-hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 220px; height: 220px;
  border-radius: 50%;
  background: radial-gradient(circle, #00b4d820 0%, transparent 70%);
  pointer-events: none;
}
.about-hero-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #00b4d8;
  background: #00b4d812;
  border: 1px solid #00b4d840;
  border-radius: 4px;
  padding: 3px 10px;
  margin-bottom: 14px;
}
.about-hero h2 {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.55rem;
  font-weight: 600;
  color: #e6edf3;
  margin: 0 0 10px;
  letter-spacing: 0.02em;
}
.about-hero p {
  font-size: 0.9rem;
  color: #8b949e;
  line-height: 1.7;
  max-width: 680px;
  margin: 0;
}

/* Section label */
.about-section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #00b4d8;
  margin: 28px 0 12px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.about-section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, #1a3a52, transparent);
}

/* Grid cards */
.about-grid {
  display: grid;
  gap: 14px;
}
.about-grid-2 { grid-template-columns: 1fr 1fr; }
.about-grid-3 { grid-template-columns: 1fr 1fr 1fr; }

.about-card {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 10px;
  padding: 20px 22px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.about-card:hover {
  border-color: #00b4d840;
  box-shadow: 0 0 18px #00b4d808;
}
.about-card-icon {
  font-size: 1.4rem;
  margin-bottom: 10px;
  display: block;
}
.about-card-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #58a6ff;
  margin-bottom: 8px;
}
.about-card-body {
  font-size: 0.84rem;
  color: #8b949e;
  line-height: 1.65;
}

/* ETL pipeline row */
.etl-row {
  display: flex;
  align-items: stretch;
  gap: 0;
  border: 1px solid #21262d;
  border-radius: 10px;
  overflow: hidden;
}
.etl-step {
  flex: 1;
  padding: 18px 16px;
  background: #0d1117;
  border-right: 1px solid #21262d;
  position: relative;
}
.etl-step:last-child { border-right: none; }
.etl-step::after {
  content: '›';
  position: absolute;
  right: -10px; top: 50%;
  transform: translateY(-50%);
  color: #00b4d8;
  font-size: 1.2rem;
  z-index: 2;
}
.etl-step:last-child::after { display: none; }
.etl-step-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  color: #00b4d8;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.etl-step-name {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.88rem;
  font-weight: 600;
  color: #e6edf3;
  margin-bottom: 6px;
}
.etl-step-desc {
  font-size: 0.78rem;
  color: #6e7681;
  line-height: 1.5;
}

/* Classes table */
.class-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.83rem;
}
.class-table th {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #8b949e;
  padding: 8px 14px;
  border-bottom: 1px solid #21262d;
  text-align: left;
}
.class-table td {
  padding: 10px 14px;
  border-bottom: 1px solid #161b22;
  color: #c9d1d9;
  vertical-align: middle;
}
.class-table tr:last-child td { border-bottom: none; }
.class-table tr:hover td { background: #161b22; }
.badge-mal {
  display: inline-block;
  background: #2d0a0a;
  border: 1px solid #ff475750;
  color: #ff6b81;
  border-radius: 4px;
  padding: 1px 8px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
}
.badge-ben {
  display: inline-block;
  background: #0a2d12;
  border: 1px solid #2ed57350;
  color: #7bed9f;
  border-radius: 4px;
  padding: 1px 8px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
}
.mono-tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 3px;
  padding: 1px 6px;
  color: #79c0ff;
}

/* Team cards */
.team-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.team-card {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 12px;
  padding: 24px 26px;
  display: flex;
  align-items: center;
  gap: 20px;
  transition: border-color 0.25s, box-shadow 0.25s;
}
.team-card:hover {
  border-color: #00b4d860;
  box-shadow: 0 0 24px #00b4d810;
}
.team-avatar {
  width: 54px; height: 54px;
  border-radius: 50%;
  background: linear-gradient(135deg, #0d2137, #00b4d830);
  border: 2px solid #00b4d840;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem;
  flex-shrink: 0;
}
.team-info { flex: 1; }
.team-name {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.98rem;
  font-weight: 600;
  color: #e6edf3;
  margin-bottom: 4px;
}
.team-role {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #00b4d8;
  margin-bottom: 6px;
}
.team-desc {
  font-size: 0.78rem;
  color: #6e7681;
  line-height: 1.5;
}

/* Architecture box */
.arch-box {
  background: #070d14;
  border: 1px solid #1a3a52;
  border-radius: 8px;
  padding: 18px 22px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  color: #79c0ff;
  line-height: 2;
}
.arch-dim { color: #6e7681; font-size: 0.72rem; }

/* Disclaimer bar */
.disclaimer-bar {
  margin-top: 32px;
  background: #0d1008;
  border: 1px solid #2ea04326;
  border-left: 3px solid #d4a017;
  border-radius: 6px;
  padding: 14px 20px;
  display: flex;
  align-items: flex-start;
  gap: 12px;
  font-size: 0.82rem;
  color: #8b949e;
  line-height: 1.6;
}
.disclaimer-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
"""


def _html_barras_probabilidad(dict_probabilidades: dict) -> str:
    barras = ""
    for etiqueta, prob in sorted(dict_probabilidades.items(), key=lambda x: -x[1]):
        porcentaje = prob * 100
        color = "#ff4757" if any(k in etiqueta for k in ["Adenocarcinoma", "Células Escamosas"]) else "#2ed573"
        barras += f"""
        <div class="contenedor-barra-prob">
          <div class="etiqueta-prob">{etiqueta} — {porcentaje:.1f}%</div>
          <div class="pista-barra-prob">
            <div class="relleno-barra-prob" style="width:{porcentaje:.1f}%; background:{color};"></div>
          </div>
        </div>"""
    return f'<div style="padding:4px 0">{barras}</div>'


def ejecutar_diagnostico(imagen: Optional[Image.Image]) -> Tuple:
    if imagen is None:
        return (
            None,
            "<p style='color:#ff4757'>Por favor, sube una imagen histopatológica.</p>",
            "",
            "Sin diagnóstico aún.",
            gr.update(visible=False),
        )

    if not Path(RUTA_MODELO).exists():
        return (
            None,
            "<p style='color:#ff4757'>Modelo no encontrado. Entrena primero el modelo.</p>",
            "",
            "",
            gr.update(visible=False),
        )

    try:
        resultado = ejecutar_diagnostico._ultimo_resultado = predecir_imagen(imagen, RUTA_MODELO)
    except Exception as e:
        registrador.exception("Fallo en predicción")
        return (
            None,
            f"<p style='color:#ff4757'>Error en el modelo: {e}</p>",
            "",
            "",
            gr.update(visible=False),
        )

    clase_tarjeta = "maligno" if resultado["is_malignant"] else "benigno"
    etiqueta      = "MALIGNO" if resultado["is_malignant"] else "BENIGNO"
    color         = "#ff4757" if resultado["is_malignant"] else "#2ed573"
    urgencia      = (
        "Consulte a su oncólogo de manera urgente."
        if resultado["is_malignant"]
        else "No se detectó malignidad. Continúe sus controles."
    )

    html_resultado = f"""
    <div class="tarjeta-resultado {clase_tarjeta}">
      <div style="font-size:1.4rem;font-weight:700;color:{color};margin-bottom:8px">{etiqueta}</div>
      <div><b>Diagnóstico:</b> {resultado['description']}</div>
      <div><b>Confianza:</b>   {resultado['confidence']*100:.2f}%</div>
      <div style="margin-top:8px;color:{color}">{urgencia}</div>
    </div>
    <div class="aviso">
      Este sistema es experimental. No reemplaza el diagnóstico de un profesional médico calificado.
    </div>
    """

    html_probabilidades = _html_barras_probabilidad(resultado["probabilities"])
    texto_reporte       = resultado["report_text"]
    figura              = resultado["gradcam_fig"]

    return figura, html_resultado, html_probabilidades, texto_reporte, gr.update(visible=True)


ejecutar_diagnostico._ultimo_resultado = None


def manejar_envio_sms(nombre_paciente: str, numero_telefono: str) -> str:
    resultado = ejecutar_diagnostico._ultimo_resultado
    if resultado is None:
        return "Realiza un diagnóstico primero antes de enviar el SMS."
    if not numero_telefono.strip():
        return "Ingresa un número de teléfono válido (ej: +573001234567)."

    nombre   = nombre_paciente.strip() or "Paciente"
    respuesta = enviar_sms_diagnostico(
        numero_destino=numero_telefono.strip(),
        resultado=resultado,
        nombre_paciente=nombre,
    )
    if respuesta["success"]:
        return f"SMS enviado exitosamente.\nSID: {respuesta['message_sid']}"
    return f"Error al enviar SMS:\n{respuesta['error']}"


def manejar_vista_previa_sms(nombre_paciente: str) -> str:
    resultado = ejecutar_diagnostico._ultimo_resultado
    if resultado is None:
        return "(Realiza un diagnóstico para ver la vista previa del SMS)"
    return formatear_vista_previa(resultado, nombre_paciente or "Paciente")


# ─────────────────────────────────────────────
#  HTML: PESTAÑA "ACERCA DE"
# ─────────────────────────────────────────────

ABOUT_HTML = """
<div class="about-page">

  <!-- HERO -->
  <div class="about-hero">
    <div class="about-hero-badge">Proyecto Académico · Big Data &amp; Deep Learning</div>
    <h2>CancerDetect AI</h2>
    <p>
      Sistema de diagnóstico asistido por inteligencia artificial para la detección de cáncer
      de pulmón y colon a partir de imágenes histopatológicas. Integra un pipeline ETL completo,
      una red neuronal convolucional con transfer learning, visualización Grad-CAM y notificación
      SMS mediante Twilio.
    </p>
  </div>

  <!-- DATASET -->
  <div class="about-section-label">Dataset</div>
  <div class="about-grid about-grid-2">
    <div class="about-card">
      <span class="about-card-icon">🧫</span>
      <div class="about-card-title">LC25000 — Kaggle</div>
      <div class="about-card-body">
        <b>25 000</b> imágenes histopatológicas de resolución <b>224 × 224 px</b> en formato JPEG.
        Divididas en 5 clases balanceadas (5 000 imágenes por clase) que abarcan tejidos
        de pulmón y colon, tanto benignos como malignos.
      </div>
    </div>
    <div class="about-card">
      <span class="about-card-icon">🔬</span>
      <div class="about-card-title">Clases Diagnósticas</div>
      <div class="about-card-body">
        <table class="class-table" style="margin-top:4px">
          <thead><tr><th>Código</th><th>Descripción</th><th>Tipo</th></tr></thead>
          <tbody>
            <tr><td><span class="mono-tag">colon_aca</span></td><td>Adenocarcinoma de Colon</td><td><span class="badge-mal">Maligno</span></td></tr>
            <tr><td><span class="mono-tag">colon_n</span></td><td>Tejido Benigno de Colon</td><td><span class="badge-ben">Benigno</span></td></tr>
            <tr><td><span class="mono-tag">lung_aca</span></td><td>Adenocarcinoma de Pulmón</td><td><span class="badge-mal">Maligno</span></td></tr>
            <tr><td><span class="mono-tag">lung_n</span></td><td>Tejido Benigno de Pulmón</td><td><span class="badge-ben">Benigno</span></td></tr>
            <tr><td><span class="mono-tag">lung_scc</span></td><td>Carcinoma de Células Escamosas</td><td><span class="badge-mal">Maligno</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ETL PIPELINE -->
  <div class="about-section-label">Pipeline ETL</div>
  <div class="etl-row">
    <div class="etl-step">
      <div class="etl-step-num">Paso 01</div>
      <div class="etl-step-name">Extract</div>
      <div class="etl-step-desc">Escaneo recursivo del árbol de directorios. Inventario completo en DataFrame con rutas, etiquetas y órgano.</div>
    </div>
    <div class="etl-step">
      <div class="etl-step-num">Paso 02</div>
      <div class="etl-step-name">Clean</div>
      <div class="etl-step-desc">Validación de imágenes corruptas o truncadas con PIL. Deduplicación por hash MD5. Filtro de tamaño mínimo.</div>
    </div>
    <div class="etl-step">
      <div class="etl-step-num">Paso 03</div>
      <div class="etl-step-name">Transform</div>
      <div class="etl-step-desc">Codificación numérica de etiquetas, flags de malignidad, splits estratificados train / val / test.</div>
    </div>
    <div class="etl-step">
      <div class="etl-step-num">Paso 04</div>
      <div class="etl-step-name">Load</div>
      <div class="etl-step-desc">DataLoaders con augmentation (flip, rotación, jitter) para entrenamiento y normalización ImageNet para inferencia.</div>
    </div>
  </div>

  <!-- ARQUITECTURA CNN -->
  <div class="about-section-label">Arquitectura CNN</div>
  <div class="about-grid about-grid-2">
    <div class="about-card">
      <span class="about-card-icon">🧠</span>
      <div class="about-card-title">EfficientNet-B0 + Cabeza Personalizada</div>
      <div class="about-card-body">
        <div class="arch-box">
Input  <span class="arch-dim">3 × 224 × 224</span>
  └─ EfficientNet-B0 backbone
       <span class="arch-dim">bloques 6–8 fine-tuned</span>
       └─ BatchNorm1d(1280)
            └─ Dropout(0.4)
                 └─ Linear 1280 → 512
                      └─ SiLU
                           └─ BatchNorm1d(512)
                                └─ Dropout(0.2)
                                     └─ Linear 512 → 5
                                          └─ Logits
        </div>
      </div>
    </div>
    <div class="about-card">
      <span class="about-card-icon">⚙️</span>
      <div class="about-card-title">Configuración de Entrenamiento</div>
      <div class="about-card-body">
        <table class="class-table">
          <tbody>
            <tr><td style="color:#8b949e">Optimizador</td><td><span class="mono-tag">AdamW</span></td></tr>
            <tr><td style="color:#8b949e">Scheduler</td><td><span class="mono-tag">CosineAnnealing</span></td></tr>
            <tr><td style="color:#8b949e">Loss</td><td><span class="mono-tag">CrossEntropy + Label Smoothing</span></td></tr>
            <tr><td style="color:#8b949e">Precisión</td><td><span class="mono-tag">Mixed Precision (AMP)</span></td></tr>
            <tr><td style="color:#8b949e">Parada</td><td><span class="mono-tag">Early Stopping (patience=5)</span></td></tr>
            <tr><td style="color:#8b949e">Augmentation</td><td><span class="mono-tag">Flip H/V · Rotación · Jitter</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TECNOLOGÍAS -->
  <div class="about-section-label">Stack Tecnológico</div>
  <div class="about-grid about-grid-3">
    <div class="about-card">
      <span class="about-card-icon">🔥</span>
      <div class="about-card-title">Deep Learning</div>
      <div class="about-card-body">PyTorch · TorchVision · EfficientNet-B0 · Grad-CAM para interpretabilidad visual del modelo.</div>
    </div>
    <div class="about-card">
      <span class="about-card-icon">🖥️</span>
      <div class="about-card-title">Interfaz Web</div>
      <div class="about-card-body">Gradio con tema personalizado oscuro, barras de probabilidad en HTML/CSS y visualización Grad-CAM en tiempo real.</div>
    </div>
    <div class="about-card">
      <span class="about-card-icon">📱</span>
      <div class="about-card-title">Notificaciones SMS</div>
      <div class="about-card-body">Twilio REST API para envío de resultados diagnósticos al número del paciente en formato E.164.</div>
    </div>
    <div class="about-card">
      <span class="about-card-icon">🧹</span>
      <div class="about-card-title">ETL &amp; Datos</div>
      <div class="about-card-body">Pandas · scikit-learn · OpenCV · PIL. Pipeline con validación MD5, deduplicación y splits estratificados.</div>
    </div>
    <div class="about-card">
      <span class="about-card-icon">📊</span>
      <div class="about-card-title">Visualización</div>
      <div class="about-card-body">Matplotlib · Seaborn. Curvas de entrenamiento, matrices de confusión y distribución de clases post-ETL.</div>
    </div>
    <div class="about-card">
      <span class="about-card-icon">✅</span>
      <div class="about-card-title">Testing</div>
      <div class="about-card-body">pytest con cobertura de transforms, arquitectura del modelo, validación de imágenes y módulo SMS.</div>
    </div>
  </div>

  <!-- INTEGRANTES -->
  <div class="about-section-label">Equipo de Desarrollo</div>
  <div class="team-grid">
    <div class="team-card">
      <div class="team-avatar">👩‍💻</div>
      <div class="team-info">
        <div class="team-name">Juanita Solórzano Salazar</div>
        <div class="team-role">Investigadora · Desarrolladora</div>
        
      </div>
    </div>
    <div class="team-card">
      <div class="team-avatar">👨‍💻</div>
      <div class="team-info">
        <div class="team-name">Cristian Ocampo</div>
        <div class="team-role">Investigador · Desarrollador</div>

      </div>
    </div>
  </div>

  <!-- DISCLAIMER -->
  <div class="disclaimer-bar">
    <span class="disclaimer-icon">⚕️</span>
    <span>
      <b style="color:#d4a017">Aviso Médico — Uso Académico.</b>
      Este sistema es una herramienta experimental desarrollada con fines educativos en el marco
      de un curso de Big Data y Deep Learning. Los resultados generados <b>no constituyen un
      diagnóstico médico</b> y no deben utilizarse como sustituto de la evaluación de un patólogo
      o médico certificado. Consulte siempre a un profesional de la salud calificado.
    </span>
  </div>

</div>
"""


def construir_interfaz() -> gr.Blocks:
    with gr.Blocks(theme=TEMA, css=CSS_PERSONALIZADO, title="CancerDetect AI") as aplicacion:

        gr.HTML("""
        <div id="cabecera-app">
          <h1>CancerDetect AI</h1>
          <p>Detección de Cáncer de Pulmón &amp; Colon mediante Deep Learning · CNN EfficientNet-B0 · Grad-CAM</p>
        </div>
        """)

        with gr.Tabs(elem_classes="tab-nav"):

            with gr.TabItem("Diagnóstico"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=280):
                        entrada_imagen = gr.Image(
                            type="pil",
                            label="Imagen Histopatológica",
                            height=280,
                        )
                        boton_analizar = gr.Button("Analizar Imagen", variant="primary", size="lg")
                        gr.Markdown(
                            "_Sube una imagen `.jpeg/.jpg` de tejido histopatológico "
                            "de pulmón o colon (224×224 px recomendado)._"
                        )

                    with gr.Column(scale=2):
                        html_resultado      = gr.HTML(
                            "<p style='color:#4a5568;font-family:monospace'>"
                            "El resultado aparecerá aquí...</p>"
                        )
                        html_probabilidades = gr.HTML("")
                        salida_gradcam      = gr.Plot(label="Visualización Grad-CAM")

                with gr.Accordion("Reporte Detallado", open=False):
                    texto_reporte = gr.Textbox(
                        label="", lines=14, interactive=False,
                        elem_id="caja-reporte"
                    )

                with gr.Group(visible=False, elem_classes="seccion-sms") as grupo_sms:
                    gr.Markdown("### Enviar Resultados por SMS (Twilio)")
                    with gr.Row():
                        nombre_paciente = gr.Textbox(label="Nombre del Paciente", placeholder="Ej: Juan García")
                        numero_telefono = gr.Textbox(label="Teléfono (E.164)", placeholder="+573001234567")
                    with gr.Row():
                        boton_vista_previa = gr.Button("Vista Previa SMS", variant="secondary")
                        boton_enviar       = gr.Button("Enviar SMS", variant="primary")
                    vista_previa_sms = gr.Textbox(label="Vista Previa", lines=8, interactive=False)
                    estado_sms       = gr.Textbox(label="Estado del Envío", interactive=False)

                boton_analizar.click(
                    ejecutar_diagnostico,
                    inputs=[entrada_imagen],
                    outputs=[salida_gradcam, html_resultado, html_probabilidades, texto_reporte, grupo_sms],
                )
                boton_vista_previa.click(
                    manejar_vista_previa_sms,
                    inputs=[nombre_paciente],
                    outputs=[vista_previa_sms],
                )
                boton_enviar.click(
                    manejar_envio_sms,
                    inputs=[nombre_paciente, numero_telefono],
                    outputs=[estado_sms],
                )

            with gr.TabItem("Acerca de"):
                gr.HTML(ABOUT_HTML)

    return aplicacion


def lanzar_app():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    aplicacion = construir_interfaz()
    aplicacion.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    lanzar_app()