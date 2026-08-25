# iDraw Ultimate

Fotózott kézírásból UUNA TEK/iDraw A4 pen plotterhez előkészített, centerline-alapú és geometriailag valószínűsített tollpálya.

## Új pipeline
- konzervatív tinta-kiemelés
- skeleton / centerline
- junction stroke-párosítás
- raster-gap javítás
- resampling + Savitzky–Golay simítás
- geometriai egyszerűsítés
- valószínű stroke-sorrend és irány
- pen-up útvonal optimalizálás + bounded 2-opt
- UUNA TEK A4 safe-zone ellenőrzés
- automatikus aránytartó A4 safe-fit
- zero-motion plot plan: távolságok, becsült idő, figyelmeztetések
- gép-előkészített SVG + JSON plot plan

## Fontos korlát
A lapos fotó nem tartalmazza biztosan az eredeti toll sebességét, pen-up/pen-down telemetriáját és teljes történeti stroke-sorrendjét. Ezeket a rendszer valószínűsíti, nem állítja vissza bizonyítottan.

## Futtatás
```bash
pip install -r requirements.txt
python app.py
```
Render belépési pont: `wsgi:app`.

## Teszt
```bash
python -m unittest discover -s tests -v
```

A fizikai UUNA TEK küldőréteg szándékosan külön marad, hogy a webes feldolgozás és a helyi USB/GRBL vezérlés ne legyen összekeverve.

A donor `WelcomePastToday/pen-plotter` MIT licencű referencia; a részleteket a `THIRD_PARTY_NOTICES.md` dokumentálja.
