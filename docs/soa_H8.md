# Tabel SOA: inundații din Sentinel-1 (10 articole)

Cifrele sunt cele raportate de autori; metricile nu se compară direct între rânduri, pentru că setul de date și sarcina diferă.

| # | Articol | Metodă | Date | Metrică | Rezultat |
|---|---|---|---|---|---|
| 1 | Rambour et al., *Flood detection in time series of optical and SAR images*, ISPRS Archives XLIII-B2, 2020 | ResNet-50 pe imagine vs. ResNet-50 + GRU pe serie | SEN12FLOOD (S1 + S2, serii) | acuratețe pe imagine | doar SAR: 0,753 pe imagine → 0,875 pe serie; SAR + RGB pe serie: 0,957 |
| 2 | Bonafilia et al., *Sen1Floods11*, CVPR Workshops, 2020 | prag Otsu pe VH vs. FCNN ResNet-50 | Sen1Floods11 (446 cipuri etichetate manual, 11 evenimente) | IoU, toată apa | Otsu 0,359; FCNN cu etichete manuale 0,313; FCNN cu etichete slabe S2 0,408 |
| 3 | Konapala, Kumar, Ahmad, ISPRS J. Photogramm. 180, 2021 | U-Net, 32 de combinații de intrări | Sen1Floods11 + DEM | F1 median | S1: 0,62; S1 + DEM: 0,73; S2 în HSV: 0,90 |
| 4 | Bai et al., *Remote Sensing* 13(11), 2220, 2021 | BASNet (ResNet-34), fuziune S1 + S2 | Sen1Floods11 | IoU / OA | test: IoU 52,3 %, OA 92,8 %; Bolivia (eveniment nevăzut): IoU 76,7 % |
| 5 | DeVries et al., *Remote Sens. Environ.* 240, 2020 | detecție de schimbare multi-temporală S1 în GEE + istoric Landsat | Houston 2017, Tesalia 2018, Madagascar 2018 | acuratețe / acord cu EMS | Houston 89,8 ± 2,8 % (cu 1,6–9,8 pp peste prag simplu); acord cu EMS 98,5 % |
| 6 | Bauer-Marschallinger et al., *Remote Sensing* 14(15), 3673, 2022 | clasificator Bayes pe cub S1 (model armonic sezonier pe pixel) | S1 VV 2015–2020, Grecia (Tesalia 2018) | acuratețe globală | circa 96 %, puține false pozitive |
| 7 | Twele et al., *Int. J. Remote Sens.* 37(13), 2016 | lanț automat: prag + clasificare fuzzy | S1 IW VV/VH, granița Grecia–Turcia | OA / κ | OA 94,0–96,1 %, κ 0,88–0,91; hartă în mai puțin de 45 de minute |
| 8 | Martinis, Plank, Ćwik, *Remote Sensing* 10(4), 583, 2018 | strat de excludere a nisipului din statistici pe serie S1 | 202 scene S1 2014–2017, Somalia/Etiopia | OA / UA | OA 98,5 % (+5 pp); UA 96,0 % (+25,2 pp) |
| 9 | Bountos et al., *Kuro Siwo*, arXiv 2311.12056, 2024 | UNet / DeepLab / ViT pe tripleți S1 (2 imagini înainte + 1 după) | 43 de evenimente 2015–2022, circa jumătate în Europa | F1 inundație / mIoU | UNet-ResNet50: F1 80,1 %, mIoU 76,2 % |
| 10 | Yadav, Nascetti, Ban, IGARSS, 2022 | U-Net siamez cu atenție, înainte/după | Sen1Floods11 bi-temporal | IoU | +6 pp IoU față de cea mai bună metodă uni-temporală |

**Ce iau de aici pentru H8**

- **Baseline-ul pragului e greu de bătut** (rândul 2: Otsu bate FCNN-ul antrenat pe etichete manuale), deci Otsu pe VV/VH e primul baseline.
- **Seria temporală ajută:** rândurile 1, 5, 6, 8 și 9 câștigă din informația despre „normalul” locului. Asta e reprezentarea mea, abaterea față de sezon.
- **Contribuția posibilă:** niciunul dintre cele 10 nu testează pe Dunărea de Jos. Contextul local există doar ca studii fără metrici
  (Iurist et al., RevCAD 22, 2017, Galați, inundațiile din 2016; Cîmpianu et al., *Remote Sensing* 13(23), 4934, 2021, Prut).
