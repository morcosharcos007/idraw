# iDraw – kézírás-feldolgozó

Flask alkalmazás, amely fotózott kézírásból tisztított, középvonal-alapú SVG vonalpályát készít iDraw/UUNA TEK pen plotterhez.

## Feldolgozási elv

1. háttér- és lokális kontraszt alapú tinta-kiemelés
2. konzervatív zaj- és komponensszűrés
3. skeleton / centerline képzés
4. rövid skeleton-spur ágak eltávolítása
5. junctionökben tangens-alapú, pontos stroke-párosítás
6. apró raster-szakadások óvatos javítása
7. ívhossz szerinti újramintavételezés
8. enyhe Savitzky–Golay simítás
9. folyamatos cubic Bézier SVG pathok
10. XML és path-validáció letöltés előtt

A cél nem a bitmap körberajzolása, hanem a kézírás középvonalának rekonstruálása, hogy a plotter ne kapjon fűrészfogas pixelpályát.

## UUNA TEK / iDraw kompatibilitás

Az alkalmazás szándékosan nyílt, `fill="none"` SVG pathokat készít, `round` linecap és linejoin beállítással. A kimenet Inkscape-ben és az iDraw/UUNA TEK SVG workflow-ban tovább szerkeszthető és méretezhető.

A projektben nem keverjük a gépspecifikus UUNA TEK firmware-t vagy Inkscape extensiont az SVG-generátorral: az alkalmazás feladata a jó vektoros pálya előállítása, a plottervezérlést az UUNA TEK/iDraw szoftver végzi.

## Indítás

```bash
pip install -r requirements.txt
python app.py
```

A Render deployment a `PORT` környezeti változót használja.
