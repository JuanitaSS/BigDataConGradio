import os
import logging
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv(override=True)
registrador = logging.getLogger(__name__)


def enviar_sms_diagnostico(
    numero_destino: str,
    resultado:      Dict,
    nombre_paciente: str = "Paciente",
    id_cuenta:       Optional[str] = None,
    token_auth:      Optional[str] = None,
    numero_origen:   Optional[str] = None,
) -> Dict:
    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
    except ImportError:
        return {"success": False, "message_sid": None, "error": "twilio no instalado"}

    sid    = id_cuenta     or os.getenv("TWILIO_ACCOUNT_SID")
    token  = token_auth    or os.getenv("TWILIO_AUTH_TOKEN")
    origen = numero_origen or os.getenv("TWILIO_PHONE_NUMBER")

    if not all([sid, token, origen]):
        error = "Faltan credenciales de Twilio (ACCOUNT_SID, AUTH_TOKEN, PHONE_NUMBER)"
        registrador.error(error)
        return {"success": False, "message_sid": None, "error": error}

    etiqueta = "MALIGNO" if resultado["is_malignant"] else "BENIGNO"
    clase    = resultado["description"]
    confianza = resultado["confidence"] * 100
    urgencia  = ("URGENTE — Consulte a su oncólogo de inmediato."
                 if resultado["is_malignant"]
                 else "No se detectó malignidad. Mantenga sus controles rutinarios.")

    cuerpo = (
        f"DIAGNÓSTICO IA — {nombre_paciente}\n"
        f"Resultado: {etiqueta}\n"
        f"Clase: {clase}\n"
        f"Confianza: {confianza:.1f}%\n"
        f"{urgencia}\n"
        f"—\n"
        f"Sistema de Detección de Cáncer de Pulmón y Colon\n"
        f"Este mensaje es informativo. Consulte siempre a un médico."
    )

    try:
        cliente = Client(sid, token)
        mensaje = cliente.messages.create(body=cuerpo, from_=origen, to=numero_destino)
        registrador.info("SMS enviado: %s -> %s", mensaje.sid, numero_destino)
        return {"success": True, "message_sid": mensaje.sid, "error": None}
    except TwilioRestException as e:
        registrador.error("Error de Twilio: %s", e)
        return {"success": False, "message_sid": None, "error": str(e)}


def formatear_vista_previa(resultado: Dict, nombre_paciente: str = "Paciente") -> str:
    etiqueta  = "MALIGNO" if resultado["is_malignant"] else "BENIGNO"
    clase     = resultado["description"]
    confianza = resultado["confidence"] * 100
    urgencia  = ("URGENTE — Consulte a su oncólogo de inmediato."
                 if resultado["is_malignant"]
                 else "No se detectó malignidad. Mantenga sus controles rutinarios.")

    return (
        f"DIAGNÓSTICO IA — {nombre_paciente}\n"
        f"Resultado: {etiqueta}\n"
        f"Clase: {clase}\n"
        f"Confianza: {confianza:.1f}%\n"
        f"{urgencia}\n"
        f"—\n"
        f"Sistema de Detección de Cáncer de Pulmón y Colon\n"
        f"Este mensaje es informativo. Consulte siempre a un médico."
    )
