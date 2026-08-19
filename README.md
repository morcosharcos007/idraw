# iDraw – kézírás-feldolgozó

Flask alkalmazás, amely fotózott kézírásból tisztított képet és SVG vonalpályát készít.

## Indítás

```bash
pip install -r requirements.txt
python app.py
```

A webalkalmazás alapértelmezés szerint a `PORT` környezeti változót használja, ennek hiányában a 8080-as porton indul.

## Render

A projekt tartalmaz `render.yaml` konfigurációt. A szolgáltatás Gunicornnal indul, és a `/health` útvonal használható health checkként.

## Folyamat

1. Kép feltöltése → `POST /process`
2. Feldolgozott állapot tárolása tokennel
3. SVG generálás → `POST /generate-svg`
4. SVG letöltés → `GET /download-svg?state_token=...`
