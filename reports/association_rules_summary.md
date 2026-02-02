# Association Rules Summary

**Generated:** 2026-02-02 09:32
**Method:** FP-Growth for frequent itemsets + Association Rules
**Total Clusters Analyzed:** 224

## Methodology

1. Preprocessed tags and titles for each photo (tokenization, stopword removal)
2. Grouped photos by cluster to create transaction sets
3. Applied FP-Growth algorithm to find frequent term combinations
4. Generated association rules with confidence ≥ 0.3
5. Used rules + itemsets + TF-IDF for automatic cluster naming

## Cluster Names

| Cluster | Size | Auto-Generated Name | Method |
|---------|------|---------------------|--------|
| 32 | 5054 | Ddc | tfidf |
| 33 | 3146 | Chaos | tfidf |
| 31 | 2683 | Abodeofchaos + Demeureduchaos | frequent_itemset |
| 90 | 2303 | Confluences | Confluence | tfidf |
| 143 | 1840 | Bmx | Skatepark | tfidf |
| 121 | 1700 | Unicef | Ong | tfidf |
| 169 | 1547 | Opera | Pradel | tfidf |
| 180 | 1293 | Chef | Place | tfidf |
| 221 | 1256 | Miniature | Miniature Miniature | tfidf |
| 173 | 1142 | Jacobins | Place Jacobins | tfidf |
| 225 | 1126 | Saint Jean | Jean | tfidf |
| 216 | 1072 | Bellecour | Crs Retraite | tfidf |
| 2 | 1032 | Stadium | Dion | tfidf |
| 85 | 1026 | Foursquare Venue | Foursquare | tfidf |
| 132 | 1010 | Open | Stella | tfidf |
| 99 | 937 | Biennale | Biennaledelyon | tfidf |
| 0 | 914 | Chaponost | Ballon | tfidf |
| 94 | 873 | Parc | Park | tfidf |
| 207 | 850 | Zombie | Pont Bonaparte | tfidf |
| 117 | 797 | Robot | Innovation | tfidf |
| 167 | 772 | Georges | Saint Georges | tfidf |
| 171 | 762 | Fresque Lyonnais | Fresque | tfidf |
| 145 | 757 | Nuits | Theatre | tfidf |
| 11 | 751 | Cosplay | Japan Touch | tfidf |
| 124 | 742 | Incity | Tour | tfidf |
| 59 | 724 | Europa Museu | Museum Europa | tfidf |
| 151 | 723 | Fourviere | Panoramic | tfidf |
| 191 | 719 | Revolutions | Zombie | tfidf |
| 19 | 717 | Chaos | tfidf |
| 164 | 691 | Nizier | Saint Nizier | tfidf |
| 130 | 685 | Part | Part Dieu | tfidf |
| 177 | 683 | Foursquare Venue | Foursquare | tfidf |
| 136 | 656 | Marquis | tfidf |
| 194 | 655 | Celestins | Parking | tfidf |
| 10 | 652 | Boardgame | Nuages Nuages | tfidf |
| 23 | 624 | Touch | Japan | tfidf |
| 4 | 622 | Miribel | Jonage | tfidf |
| 29 | 619 | Chaos | tfidf |
| 125 | 601 | Auditorium | Partdieu | tfidf |
| 183 | 595 | Pasted | Paper | tfidf |
| 22 | 584 | Chaos | tfidf |
| 98 | 580 | Confluence | Cube Orange | tfidf |
| 208 | 575 | Passerelle | Justice | tfidf |
| 24 | 574 | Automobile Voitures | Epoqu Auto | tfidf |
| 205 | 569 | Justice | Palais Justice | tfidf |
| 34 | 565 | Chaos | tfidf |
| 116 | 562 | Mac | Biennale | tfidf |
| 152 | 559 | Zombie | Zombie Walk | tfidf |
| 76 | 548 | Insa Villeurbanne | Cross Insa | tfidf |
| 66 | 543 | Lou | Ubb | tfidf |

## Sample Association Rules

### Cluster 32: Ddc (5054 photos)

1. `mystic + survivetheapocalypse` → `manifestation` (conf: 1.00, lift: 2.47)
2. `endoftheworld + survivetheapocalypse` → `manifestation` (conf: 1.00, lift: 2.47)
3. `survivetheapocalypse + actingperformance` → `manifestation` (conf: 1.00, lift: 2.47)
4. `eros + survivetheapocalypse` → `manifestation` (conf: 1.00, lift: 2.47)
5. `manifestation` → `event` (conf: 1.00, lift: 2.47)

### Cluster 33: Chaos (3146 photos)

1. `ehrmann + freemasonry` → `thierry` (conf: 0.93, lift: 1.78)
2. `thierry` → `ehrmann + freemasonry` (conf: 0.56, lift: 1.78)
3. `streetart + thierry` → `ehrmann` (conf: 1.00, lift: 1.70)
4. `ehrmann` → `streetart + thierry` (conf: 0.81, lift: 1.70)
5. `organmuseum + thierry` → `ehrmann` (conf: 1.00, lift: 1.70)

### Cluster 31: Abodeofchaos + Demeureduchaos (2683 photos)

1. `francmaconnerie` → `sanctuaire + freemasonry` (conf: 0.77, lift: 1.60)
2. `francmaconnerie` → `freemasonry + prophétie` (conf: 0.74, lift: 1.60)
3. `lespritdelasalamandre + freemasonry` → `francmaconnerie` (conf: 1.00, lift: 1.60)
4. `francmaconnerie` → `lespritdelasalamandre + freemasonry` (conf: 1.00, lift: 1.60)
5. `salamanderspirit + freemasonry` → `francmaconnerie` (conf: 1.00, lift: 1.60)

### Cluster 90: Confluences | Confluence (2303 photos)

1. `francie + франция` → `フランス` (conf: 1.00, lift: 16.64)
2. `フランス` → `francie + франция` (conf: 1.00, lift: 16.64)
3. `francie + лион` → `フランス` (conf: 1.00, lift: 16.64)
4. `フランス` → `francie + лион` (conf: 1.00, lift: 16.64)
5. `франция + francja` → `フランス` (conf: 1.00, lift: 16.64)

### Cluster 143: Bmx | Skatepark (1840 photos)

1. `dj` → `plateforme` (conf: 0.97, lift: 36.80)
2. `plateforme` → `dj` (conf: 0.97, lift: 36.80)
3. `europa` → `frança` (conf: 0.80, lift: 26.87)
4. `frança` → `europa` (conf: 0.71, lift: 26.87)
5. `foursquare` → `venue` (conf: 1.00, lift: 26.56)

### Cluster 121: Unicef | Ong (1700 photos)

1. `international` → `vaccination` (conf: 1.00, lift: 49.53)
2. `vaccination` → `international` (conf: 1.00, lift: 49.53)
3. `international + unicefrhône` → `vaccination` (conf: 1.00, lift: 49.53)
4. `unicefrhône + vaccination` → `international` (conf: 1.00, lift: 49.53)
5. `international` → `unicefrhône + vaccination` (conf: 1.00, lift: 49.53)

### Cluster 169: Opera | Pradel (1547 photos)

1. `december8th` → `festivaloflights + grandlyon` (conf: 0.94, lift: 28.94)
2. `december8th` → `grandlyon + 8décembre` (conf: 0.94, lift: 28.94)
3. `grandlyon` → `1er + december8th` (conf: 0.94, lift: 28.94)
4. `december8th` → `1er + grandlyon` (conf: 0.94, lift: 28.94)
5. `december8th` → `grandlyon + festival` (conf: 0.94, lift: 28.94)

### Cluster 180: Chef | Place (1293 photos)

1. `air + bellecour` → `yoann` (conf: 1.00, lift: 27.78)
2. `yoann` → `air + bellecour` (conf: 1.00, lift: 27.78)
3. `air + 8décembre` → `yoann` (conf: 1.00, lift: 27.78)
4. `yoann` → `air + 8décembre` (conf: 1.00, lift: 27.78)
5. `air + illuminations` → `yoann` (conf: 1.00, lift: 27.78)

### Cluster 221: Miniature | Miniature Miniature (1256 photos)

1. `view + dark` → `special` (conf: 1.00, lift: 31.95)
2. `view + light` → `special` (conf: 1.00, lift: 31.95)
3. `dark + light` → `special` (conf: 1.00, lift: 31.95)
4. `maisondesavocats + unesco` → `lawoffices` (conf: 1.00, lift: 31.95)
5. `lawoffices` → `maisondesavocats + unesco` (conf: 1.00, lift: 31.95)

### Cluster 173: Jacobins | Place Jacobins (1142 photos)

1. `fdl` → `n1` (conf: 1.00, lift: 45.29)
2. `n1` → `fdl` (conf: 1.00, lift: 45.29)
3. `supermimil` → `n1` (conf: 1.00, lift: 45.29)
4. `n1` → `supermimil` (conf: 1.00, lift: 45.29)
5. `supermimil` → `fdl` (conf: 1.00, lift: 45.29)

## Naming Method Distribution

| Method | Count | Percentage |
|--------|-------|------------|
| tfidf | 222 | 98.2% |
| fallback | 2 | 0.9% |
| frequent_itemset | 2 | 0.9% |

## Validation: Known Lyon Landmarks

| Expected Landmark | Matched Cluster | Auto-Name | Status |
|-------------------|-----------------|-----------|--------|
| Fourvière | 135 | Ainay | Martin | ✅ |
| Vieux Lyon | 206 | Vieuxlyon | Traboules | ✅ |
| Place Bellecour | 201 | Hommage | Marche | ✅ |
| Fête des Lumières | 115 | Brotteaux | Lumieres Fete | ✅ |
| Demeure du Chaos | 18 | Palaisideal + Vanitas | ✅ |
| Parc Tête d'Or | 105 | Zoo | Show | ✅ |
| Croix-Rousse | 122 | Canuts | Mur | ✅ |
| Presqu'île | 173 | Jacobins | Place Jacobins | ✅ |