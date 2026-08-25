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


## Stroke reconstruction és minőségi módok

A `stroke_reconstruction.py` réteg az alkalmazás meglévő centerline/skeleton gráfjára épül. A minőség 1–5 nem csak vizuális címke: egyszerre változtatja a mintavételezési sűrűséget, a simítást, a geometriai egyszerűsítést és a minimális stroke-hosszt.

A statikus fotó a vonal geometriáját őrzi meg, de az eredeti toll-telemetriát (sebesség, pen-up/pen-down időpont, pontos irány) nem. Ezért a stroke-sorrend **geometriai valószínűsítés**: junction-folytonosság, olvasási irány, végpontok közötti utazási távolság és irányfolytonosság alapján próbálja rekonstruálni a legvalószínűbb tollmozgást. Ezt nem kezeljük biztos történeti sorrendként.

A Render belépési pontja `wsgi:app`, amely a rekonstrukciós réteget a meglévő Flask alkalmazásba injektálja.
