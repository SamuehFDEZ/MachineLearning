import email
from bs4 import BeautifulSoup

class Email(object):

    CLRF = "\r\n\r\n"

    def __init__(self, archivo, categoria=None):
        self.categoria = categoria
        self.mail = email.message_from_binary_file(archivo)

    def subject(self):
        return self.mail.get("Subject")

    def body(self):
        payload = self.mail.get_payload()
        if self.mail.is_multipart():
            partes = [self._body_unico(parte) for parte in list(payload)]
        else:
            partes = [self._body_unico(self.mail)]
        partes_decodificadas = []
        for parte in partes:
            if len(parte) == 0:
                continue
            if isinstance(parte, bytes):
                partes_decodificadas.append(parte.decode("utf-8", errors="ignore"))
            else:
                partes_decodificadas.append(parte)
        return self.CLRF.join(partes_decodificadas)

    @staticmethod
    def _body_unico(parte):
        tipo_de_contenido = parte.get_content_type()
        try:
            body = parte.get_payload(decode=True)
        except Exception:
            return ""
        if tipo_de_contenido == "text/html":
            return BeautifulSoup(body, "html.parser").text
        elif tipo_de_contenido == "text/plain":
            return body
        return ""