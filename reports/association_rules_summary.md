# Association Rules Summary

**Generated:** 2026-02-02 13:03
**Method:** FP-Growth for frequent itemsets + Association Rules
**Total Clusters Analyzed:** 274

## Methodology

1. Preprocessed tags and titles for each photo (tokenization, stopword removal)
2. Grouped photos by cluster to create transaction sets
3. Applied FP-Growth algorithm to find frequent term combinations
4. Generated association rules with confidence ≥ 0.3
5. Used rules + itemsets + TF-IDF for automatic cluster naming

## Cluster Names

| Cluster | Size | Auto-Generated Name | Method |
|---------|------|---------------------|--------|
| 42 | 5054 | Ddc | tfidf |
| 43 | 3146 | Chaos | tfidf |
| 41 | 2683 | Francmaconnerie + Sanctuaire | association_rule |
| 102 | 2331 | Confluences | Confluence | tfidf |
| 173 | 1895 | Bmx | River | tfidf |
| 139 | 1701 | Unicef | Ong | tfidf |
| 194 | 1175 | Place | Republique | tfidf |
| 208 | 1152 | Jacobins | Place Jacobins | tfidf |
| 3 | 1026 | Stadium | Dion | tfidf |
| 154 | 1021 | Open | Stella | tfidf |
| 200 | 970 | Georges | Saint Georges | tfidf |
| 214 | 909 | Museedesbeauxarts | Bellecour | tfidf |
| 0 | 884 | Craponne | Moulin | tfidf |
| 167 | 880 | Fourviere | Panoramic | tfidf |
| 240 | 798 | Opera | Nouvel | tfidf |
| 141 | 797 | Robot | Conference | tfidf |
| 211 | 756 | Fresque Lyonnais | Fresque | tfidf |
| 64 | 754 | Europa Museu | Museum Europa | tfidf |
| 12 | 752 | Cosplay | Japan Touch | tfidf |
| 232 | 751 | Pasted | Paper | tfidf |
| 260 | 740 | Zombie | Pont | tfidf |
| 147 | 729 | Incity | Tour | tfidf |
| 26 | 717 | Chaos | tfidf |
| 174 | 717 | Nuits | Theatre | tfidf |
| 224 | 713 | Foursquare | Foursquare Venue | tfidf |
| 213 | 708 | Nizier | Saint Nizier | tfidf |
| 235 | 695 | Revolutions | Zombie | tfidf |
| 15 | 653 | Boardgame | Nuages Nuages | tfidf |
| 233 | 652 | Celestins | Parking | tfidf |
| 19 | 624 | Touch | Japan | tfidf |
| 118 | 621 | Parc | Park | tfidf |
| 114 | 620 | Confluence | Cube | tfidf |
| 39 | 619 | Chaos | tfidf |
| 187 | 615 | Croixrousse | Zombie | tfidf |
| 146 | 589 | Auditorium | Partdieu | tfidf |
| 27 | 583 | Chaos | tfidf |
| 20 | 574 | Automobile Voitures | Epoqu | tfidf |
| 210 | 572 | Fireworks | Artifice Feu | tfidf |
| 45 | 558 | Chaos | tfidf |
| 266 | 558 | Vieuxlyon | Vieux | tfidf |
| 82 | 548 | Insa Villeurbanne | Cross Insa | tfidf |
| 259 | 548 | Justice | Palais Justice | tfidf |
| 148 | 545 | Part Dieu | Gare | tfidf |
| 140 | 543 | Mac | Biennale | tfidf |
| 125 | 528 | Confluence | Lego Ferrari | tfidf |
| 132 | 527 | Parc | Fleurs | tfidf |
| 133 | 513 | Library | Chocolat | tfidf |
| 138 | 510 | Parc | Zoo | tfidf |
| 1 | 506 | Craponne | Mai | tfidf |
| 55 | 496 | Rieur | Inte | tfidf |

## Sample Association Rules

### Cluster 42: Ddc (5054 photos)

1. `survivetheapocalypse + mystic` → `manifestation` (conf: 1.00, lift: 2.47)
2. `endoftheworld + survivetheapocalypse` → `manifestation` (conf: 1.00, lift: 2.47)
3. `survivetheapocalypse + actingperformance` → `manifestation` (conf: 1.00, lift: 2.47)
4. `eros + survivetheapocalypse` → `manifestation` (conf: 1.00, lift: 2.47)
5. `manifestation` → `event` (conf: 1.00, lift: 2.47)

### Cluster 43: Chaos (3146 photos)

1. `ehrmann + freemasonry` → `thierry` (conf: 0.93, lift: 1.78)
2. `thierry` → `ehrmann + freemasonry` (conf: 0.56, lift: 1.78)
3. `thierry + streetart` → `ehrmann` (conf: 1.00, lift: 1.70)
4. `ehrmann` → `thierry + streetart` (conf: 0.81, lift: 1.70)
5. `thierry + organmuseum` → `ehrmann` (conf: 1.00, lift: 1.70)

### Cluster 41: Francmaconnerie + Sanctuaire (2683 photos)

1. `francmaconnerie` → `sanctuaire + freemasonry` (conf: 0.77, lift: 1.60)
2. `francmaconnerie` → `prophétie + freemasonry` (conf: 0.74, lift: 1.60)
3. `lespritdelasalamandre + freemasonry` → `francmaconnerie` (conf: 1.00, lift: 1.60)
4. `francmaconnerie` → `lespritdelasalamandre + freemasonry` (conf: 1.00, lift: 1.60)
5. `salamanderspirit + freemasonry` → `francmaconnerie` (conf: 1.00, lift: 1.60)

### Cluster 102: Confluences | Confluence (2331 photos)

1. `франция + لیون` → `ประเทศฝร` (conf: 1.00, lift: 16.86)
2. `ประเทศฝร` → `франция + لیون` (conf: 1.00, lift: 16.86)
3. `франция + ليون` → `ประเทศฝร` (conf: 1.00, lift: 16.86)
4. `ประเทศฝร` → `франция + ليون` (conf: 1.00, lift: 16.86)
5. `франция + リヨン` → `ประเทศฝร` (conf: 1.00, lift: 16.86)

### Cluster 173: Bmx | River (1895 photos)

1. `crs` → `retraite` (conf: 1.00, lift: 36.62)
2. `retraite` → `crs` (conf: 1.00, lift: 36.62)
3. `riots` → `retraite` (conf: 1.00, lift: 36.62)
4. `retraite` → `riots` (conf: 1.00, lift: 36.62)
5. `crs` → `riots` (conf: 1.00, lift: 36.62)

### Cluster 139: Unicef | Ong (1701 photos)

1. `vaccination` → `international` (conf: 1.00, lift: 49.53)
2. `international` → `vaccination` (conf: 1.00, lift: 49.53)
3. `unicefrhône + vaccination` → `international` (conf: 1.00, lift: 49.53)
4. `unicefrhône + international` → `vaccination` (conf: 1.00, lift: 49.53)
5. `vaccination` → `unicefrhône + international` (conf: 1.00, lift: 49.53)

### Cluster 194: Place | Republique (1175 photos)

1. `laplacedelart` → `parc + lyonparcauto` (conf: 0.96, lift: 28.43)
2. `lyonparcauto` → `laplacedelart + parc` (conf: 0.96, lift: 28.43)
3. `parc + place` → `lyonparcauto` (conf: 1.00, lift: 28.43)
4. `laplacedelart` → `lyonparcauto` (conf: 1.00, lift: 28.43)
5. `lyonparcauto` → `laplacedelart` (conf: 1.00, lift: 28.43)

### Cluster 208: Jacobins | Place Jacobins (1152 photos)

1. `fdl` → `n1` (conf: 1.00, lift: 45.47)
2. `n1` → `fdl` (conf: 1.00, lift: 45.47)
3. `fdl` → `supermimil` (conf: 1.00, lift: 45.47)
4. `supermimil` → `fdl` (conf: 1.00, lift: 45.47)
5. `supermimil` → `n1` (conf: 1.00, lift: 45.47)

### Cluster 3: Stadium | Dion (1026 photos)

1. `dicaire` → `véronic` (conf: 1.00, lift: 26.81)
2. `véronic` → `dicaire` (conf: 1.00, lift: 26.81)
3. `parc + dicaire` → `véronic` (conf: 1.00, lift: 26.81)
4. `parc + véronic` → `dicaire` (conf: 1.00, lift: 26.81)
5. `dicaire` → `parc + véronic` (conf: 1.00, lift: 26.81)

### Cluster 154: Open | Stella (1021 photos)

1. `piscine + piscinedurhône` → `swimmingpool` (conf: 0.81, lift: 19.33)
2. `swimmingpool` → `piscine + piscinedurhône` (conf: 0.54, lift: 19.33)
3. `déficit` → `démocratique` (conf: 1.00, lift: 15.43)
4. `démocratique` → `déficit` (conf: 1.00, lift: 15.43)
5. `déficit + union` → `démocratique` (conf: 1.00, lift: 15.43)

## Naming Method Distribution

| Method | Count | Percentage |
|--------|-------|------------|
| tfidf | 272 | 98.2% |
| fallback | 3 | 1.1% |
| association_rule | 2 | 0.7% |

## Validation: Known Lyon Landmarks

| Expected Landmark | Matched Cluster | Auto-Name | Status |
|-------------------|-----------------|-----------|--------|
| Fourvière | 156 | Basilique Basilique | Basilique | ✅ |
| Vieux Lyon | 251 | Vieuxlyon | Traboules | ✅ |
| Place Bellecour | 183 | Environs | Euro Province | ✅ |
| Fête des Lumières | 123 | Brotteaux | Lumieres Fete | ✅ |
| Demeure du Chaos | 25 | Prophétie + Temps | ✅ |
| Parc Tête d'Or | 127 | Zoo | Show | ✅ |
| Croix-Rousse | 137 | Canuts | Mur | ✅ |
| Presqu'île | 183 | Environs | Euro Province | ✅ |