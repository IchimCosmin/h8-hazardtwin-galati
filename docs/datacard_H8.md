# Data card: H8 HazardTwin, Galați (Dunăre)

**Autor:** Ichim Cosmin · IC3 SIVA, an II · CP1 (Observații) · noiembrie 2026
**Repo:** https://github.com/IchimCosmin/h8-hazardtwin-galati · cifrele de mai jos se regenerează cu
`python -m h8twin.cli ingest --root <SEN12FLOOD> --out outputs` (`outputs/stats.json`).

Twin-ul are două surse de observații. **Sursa A (SEN12FLOOD)** servește la antrenare și la testarea metodei.
**Sursa B (Sentinel-1 pe Galați)** este cazul de transfer pe care se dă alerta. La CP1, A este ingerată
complet la nivel de metadate și verificată pe pixeli pe un eșantion. B are scriptul de export scris, dar nerulat
(lipsește contul GEE).

## A. SEN12FLOOD

| Câmp | Valoare |
|---|---|
| **Sursă** | Rambour et al., *SEN12-FLOOD: a SAR and Multispectral Dataset for Flood Detection*, IEEE DataPort, doi:[10.21227/w6xz-s898](https://dx.doi.org/10.21227/w6xz-s898); articol: ISPRS Archives XLIII-B2-2020, 1343–1346 |
| **Versiune descărcată** | `SEN12FLOOD.zip`, 12,2 GB, descărcat pe 2 nov. 2026 (fișiere datate 14 sep. 2020) |
| **Acoperire** | 335 de secvențe (tile-uri de circa 5 × 5 km în jurul unor orașe): Africa 221, Orientul Mijlociu 72, Australia 42 |
| **Perioadă** | S1: 15.12.2018 – 17.05.2019; S2: 13.12.2018 – 20.05.2019 |
| **Imagini** | S1: 3957 (2353 orbite ascendente, 1604 descendente); S2: 3131 |
| **Imagini pe secvență** | S1: minim 2, median 10, maxim 31; S2: minim 1, median 7, maxim 24 |
| **Revizitare (zile)** | S1: mediană 5, P90 11, maxim 12; S2: mediană 5, P90 15, maxim 35 |
| **Rezoluție spațială** | S1: 8,98·10⁻⁵ ° ≈ 10 m, EPSG:4326, circa 547 × 520 px; S2: 10 / 20 / 60 m după bandă, UTM |
| **Benzi și unități** | S1: VV și VH, σ⁰ în putere **liniară**, float32, fără valoare nodata. Pipeline-ul convertește în **dB** (10·log₁₀); mediana pe imagine e circa −10 dB în VV și −16 dB în VH. S2: 12 benzi L2A (B01–B12, B8A, fără B10), reflectanță |
| **Etichetă** | `FLOODING` (boolean) **pe imagine**, nu pe pixel. S1: 34,7 % dintre imagini sunt inundate; 40,3 % dintre secvențe conțin cel puțin o imagine inundată. S2: 38,3 % |
| **Câmpuri JSON** | `S1list.json` / `S2list.json`: `count`, `folder`, `geo` (poligon GeoJSON), `urls` (doar S1, produsele ASF), apoi pe fiecare imagine: `date`, `FLOODING`, `filename`, plus `orbit` (S1) sau `FULL-DATA-COVERAGE` (S2, 85,3 % true) |
| **Fișiere** | `<secv>/<produs S1>_corrected_VV.tif` și `_VH.tif`; `<secv>/S2_<dată>_<bandă>.tif` |
| **Licență** | Conține date Copernicus Sentinel modificate (2018–2019), sub Termenii și condițiile Sentinel (utilizare, modificare și redistribuire libere, cu atribuire). DataPort nu declară o licență separată |

### Goluri și probleme găsite la ingestie

1. **Fișiere lipsă pe disc.** JSON-ul listează imagini care nu există: în secvența `1`, 1 din 11 imagini S1
   (`…20190330T162405…`) lipsește. Index-ul marchează fiecare imagine cu `available`, iar loader-ul le sare.
   Numărătoarea completă vine din rularea `ingest` pe tot folderul.
2. **Eticheta este la nivel de eveniment.** În toate cele 335 de secvențe eticheta e monotonă: odată „inundat”, rămâne
   „inundat”. Pe eșantionul verificat pe pixeli (4 secvențe, 43 de imagini S1), fracția de pixeli VV < −18 dB crește clar
   după etichetă într-o singură secvență (secvența `1`: de la 1,3 % la 6,9 %). În celelalte trei, tile-ul nu se schimbă vizibil.
   Asta înseamnă **zgomot de etichetă**: imaginea e „inundată” pentru că zona e într-o activare EMS, nu pentru că apa se vede în tile.
3. **Split-ul oficial are scurgere.** Folderele cu 4 cifre (`0001`…) sunt train (267), iar cele fără zerouri (`0`…`67`) sunt test (68).
   67 din 68 de secvențe de test sunt la mai puțin de 10 km de o secvență de train (mediana 5,1 km), iar 249 din cele 250 de
   scene S1 din test apar și în train, decupate alături. **Nu folosesc split-ul oficial.**
4. **Diferențe față de articol.** Articolul raportează 412 serii și 47 % (S1) / 40 % (S2) imagini inundate. Release-ul are
   335 de serii și 34,7 % / 38,3 %. Raportez cifrele release-ului.
5. **Geografie.** Nu există nicio secvență din Europa sau din zone temperate cu luncă îndiguită, ca Galați.
   Transferul pe Dunăre e un test, nu o presupunere.

### Împărțirea train/test folosită

Split **spațial pe secvențe**: secvențele se grupează pe celule de 2° × 2° după centroid (21 de grupuri), iar grupuri întregi
merg în test (seed 42, țintă 20 %). Rezultat: **train 258 de secvențe / 2870 de imagini S1 (31,7 % inundate); test 77 de secvențe /
1087 de imagini (42,8 % inundate)**. Pentru CP2 adaug un split *leave-one-region-out*, ca intervalele să nu depindă de o singură împărțire.

## B. Galați: surse pentru transfer și decizie

| Sursă | Ce aduce | Perioadă / rezoluție | Stare la CP1 |
|---|---|---|---|
| Sentinel-1 GRD IW (GEE `COPERNICUS/S1_GRD`), AOI 27,90–28,25 E, 45,33–45,53 N | seria VV/VH pe luncă, % „apă nouă” (VV < prag, în afara apei permanente JRC) | 2015 – prezent · 10 m · 6–12 zile (12 zile în 2022–2025, după pierderea S1B) | script `gee/galati_s1_timeseries.js` scris, **nerulat** (fără cont GEE) |
| JRC Global Surface Water v1.4 | masca apei permanente (ocurență ≥ 80 %) | 1984–2021 · 30 m | inclus în script |
| Copernicus EMS EMSR758 (Galați și Vaslui, 14.09.2024) | hartă de referință pentru un eveniment real | un eveniment | de descărcat la CP2 |
| Cotele Dunării la Galați (INHGA / AFDJ) | etichete slabe: zilele peste cota de atenție sau de inundație | zilnic | arhiva istorică încă necerută; rezervă: debitul GloFAS |

## Limitări care intră în articol

- Etichetă pe imagine, cu zgomot: metrica de bază rămâne acuratețea pe imagine, ca în Rambour et al. (2020). Măștile pe pixel vin din Sen1Floods11.
- Radarul nu vede bine apa între clădiri (dublă reflexie), așa că alerta e definită pe lunca deschisă, nu pe străzi.
- Dunărea la Galați a avut puține inundații reale de când există S1 (din 2015), deci evaluarea pe Galați va fi pe puține evenimente.
