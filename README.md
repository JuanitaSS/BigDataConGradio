# CancerDetect AI — Detección de Cáncer de Pulmón y Colon

Sistema de diagnóstico asistido por inteligencia artificial basado en imágenes histopatológicas, con interfaz web en Gradio y notificaciones por SMS.

---

## Integrantes

| Nombre | GitHub |
|--------|--------|
| Juanita Solórzano Salazar | [@JuanitaSS](https://github.com/JuanitaSS) |
| Cristian Ocampo | [@Susanassstaion](https://github.com/Susanassstaion) |

---

## Descripción

El sistema entrena una red neuronal convolucional (EfficientNet-B0) sobre el dataset **LC25000** de imágenes histopatológicas de pulmón y colon, permitiendo clasificar 5 tipos de tejido con alta precisión. Los resultados se visualizan en una interfaz Gradio con mapas de calor Grad-CAM y pueden enviarse al paciente por SMS vía Twilio.

---

## Dataset

**Lung and Colon Cancer Histopathological Images (LC25000)**
- 25 000 imágenes de 224×224 px
- 5 clases: adenocarcinoma de colon, tejido benigno de colon, adenocarcinoma de pulmón, tejido benigno de pulmón, carcinoma de células escamosas de pulmón
- Fuente: [Kaggle — andrewmvd](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

---

## Arquitectura del Modelo

- **Backbone**: EfficientNet-B0 (transfer learning, ImageNet)
- **Cabeza clasificadora**: BatchNorm → Dropout(0.4) → Linear(1280→512) → SiLU → BN → Dropout → Linear(512→5)
- **Fine-tuning**: bloques 6–8 descongelados
- **Optimizador**: AdamW + CosineAnnealing LR + Label Smoothing
- **Interpretabilidad**: Grad-CAM

### Resultados del entrenamiento

| Métrica | Valor |
|---------|-------|
| Val Accuracy (mejor época) | 100.00% |
| Test Accuracy | 99.97% |
| Macro AUC | 1.0000 |

---

## Pipeline ETL

| Paso | Descripción |
|------|-------------|
| Extraer | Escaneo recursivo del dataset por extensión `.jpeg`/`.jpg` |
| Limpiar | Validación de imágenes corruptas, deduplicación MD5, filtro de tamaño |
| Transformar | Codificación de etiquetas y flags de malignidad |
| Cargar | Splits estratificados train/val/test + DataLoaders con augmentation |

---

## Instalación y uso

### 1. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r Requirements.txt
```

### 2. Descargar el dataset

```bash
python data\download_dataset.py
```

> Requiere autenticación de Kaggle: archivo `~/.kaggle/kaggle.json` o variables de entorno `KAGGLE_USERNAME` y `KAGGLE_KEY`.

### 3. Entrenar el modelo

```bash
python run_training.py
```

El modelo se guarda en `model/lung_colon_cnn.pth`.

### 4. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
TWILIO_ACCOUNT_SID=tu_account_sid
TWILIO_AUTH_TOKEN=tu_auth_token
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
```

### 5. Lanzar la aplicación

```bash
python app\gradioApp.py
```

La app estará disponible en `http://localhost:7860`.

---

## Estructura del proyecto

```
BigDataConGradio/
├── app/
│   ├── gradioApp.py       # Interfaz Gradio
│   ├── predict.py         # Inferencia + Grad-CAM
│   └── sms.py             # Envío de SMS con Twilio
├── src/
│   ├── utils.py           # Pipeline ETL
│   └── train.py           # Arquitectura CNN y entrenamiento
├── data/
│   └── download_dataset.py
├── model/                 # Pesos del modelo entrenado
├── run_training.py        # Script principal de entrenamiento
├── Requirements.txt
└── .env                   # Credenciales (no incluido en git)
```

---

## Tecnologías

`Python` · `PyTorch` · `EfficientNet-B0` · `Gradio` · `Twilio` · `OpenCV` · `scikit-learn` · `pandas` · `kagglehub`

---

> **Aviso**: Este sistema es una herramienta experimental de apoyo académico. No reemplaza el diagnóstico de un patólogo certificado.
