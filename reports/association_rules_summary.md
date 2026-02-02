# Association Rules Summary

**Generated:** 2026-02-01 23:13
**Method:** FP-Growth for frequent itemsets + Association Rules
**Total Clusters Analyzed:** 914

## Methodology

1. Preprocessed tags and titles for each photo (tokenization, stopword removal)
2. Grouped photos by cluster to create transaction sets
3. Applied FP-Growth algorithm to find frequent term combinations
4. Generated association rules with confidence ≥ 0.3
5. Used rules + itemsets + TF-IDF for automatic cluster naming

## Cluster Names

| Cluster | Size | Auto-Generated Name | Method |
|---------|------|---------------------|--------|
| 237 | 1699 | International + Vaccination | association_rule |
| 619 | 1142 | Bellecour + Air + Saintjean | association_rule |
| 614 | 955 | Burgundy + Placeantoninponcet + Bourgogne | association_rule |
| 753 | 791 | Shooting + Photographie | association_rule |
| 431 | 788 | 프랑스 + 法国 | association_rule |
| 150 | 548 | Cross + Villeurbanne | association_rule |
| 78 | 494 | Blanc + Partipirate + Drapeau | association_rule |
| 162 | 453 | Ldoll2016 + Dd + Rin | association_rule |
| 173 | 441 | Pentaxk10D + Cimetièreloyasse | association_rule |
| 603 | 441 | Street + City + Streetphotography | association_rule |
| 212 | 410 | Bookselections + Flickrselections | association_rule |
| 478 | 359 | Europe + Zombie + Pentax | association_rule |
| 157 | 355 | Horizonexpress + Expert | association_rule |
| 30 | 348 | Championnat | Monde | Instagram Foursquare | tfidf |
| 819 | 339 | City + Footbridge + Facade | association_rule |
| 146 | 326 | Doublemixte + Japantouch2011 | association_rule |
| 645 | 319 | Lafresquedeslyonnais + Trompeloeil + Opticalillusion | association_rule |
| 528 | 315 | Croix + Bellecour + Jean | association_rule |
| 9 | 295 | Ленин + Lenin | frequent_itemset |
| 649 | 291 | Ville + Eglise + Pentax | association_rule |
| 762 | 289 | Terrier + Whippet + Instadogs | association_rule |
| 384 | 287 | Greenhouse + Europe + Grandeserre | association_rule |
| 674 | 284 | City + Croixrousse + Urban | association_rule |
| 201 | 281 | Commun + Amphiprion + Ocellaris | association_rule |
| 151 | 268 | Cross + Villeurbanne | association_rule |
| 868 | 267 | Basilique + Fourviere | association_rule |
| 639 | 263 | Africain + Studio + Portraits | association_rule |
| 26 | 256 | Lyon2015 + Stadebalmont | association_rule |
| 203 | 251 | Tony + Touroftheuniverse | association_rule |
| 754 | 240 | Quartierstjeancroixrousse + Rhônealps | association_rule |
| 10 | 237 | Tournant + Pont | association_rule |
| 121 | 236 | Blackmetal + Arnalle + Thrash | association_rule |
| 591 | 235 | Street + Unesco + Pavement | association_rule |
| 866 | 234 | Fête + Lumières | association_rule |
| 328 | 233 | Tag + Graph | association_rule |
| 387 | 232 | Bokeh + Arbre + City | association_rule |
| 802 | 229 | Bw + Blackandwhite + Monochrome | association_rule |
| 474 | 226 | Contemporaryart + Igers + Miamiart | association_rule |
| 872 | 217 | Citéinternationale + Parade | association_rule |
| 326 | 214 | Tete + Macro | association_rule |
| 610 | 213 | Fr + Auvergnerhônealpes + July2018Lyon | association_rule |
| 365 | 210 | Railway + Station | association_rule |
| 629 | 210 | Round + Lumiere + Manege | association_rule |
| 682 | 203 | 50D + Tamronspaf1750Mmf28Xrdiiildasphericalif | association_rule |
| 60 | 202 | Quartier + Frankrijk + Frankreich | association_rule |
| 330 | 199 | Sigma1770Mmf2845 + Heavymetal + Murderdolls | association_rule |
| 353 | 195 | Part + Landscape + Vacances | association_rule |
| 624 | 195 | Grandhôteldieu + Architecture + Grand | association_rule |
| 687 | 195 | 2E + Fêtedeslumières2009 + December8 | association_rule |
| 608 | 193 | Footbridge + Eolight + Fusgängerbrücke | association_rule |

## Sample Association Rules

### Cluster 237: International + Vaccination (1699 photos)

1. `international` → `vaccination` (conf: 1.00, lift: 49.53)
2. `vaccination` → `international` (conf: 1.00, lift: 49.53)
3. `unicefrhône + international` → `vaccination` (conf: 1.00, lift: 49.53)
4. `unicefrhône + vaccination` → `international` (conf: 1.00, lift: 49.53)
5. `international` → `unicefrhône + vaccination` (conf: 1.00, lift: 49.53)

### Cluster 619: Bellecour + Air + Saintjean (1142 photos)

1. `bellecour + air` → `saintjean` (conf: 1.00, lift: 25.22)
2. `saintjean` → `bellecour + air` (conf: 1.00, lift: 25.22)
3. `bellecour + large` → `saintjean` (conf: 1.00, lift: 25.22)
4. `saintjean` → `bellecour + large` (conf: 1.00, lift: 25.22)
5. `bellecour + 8décembre` → `saintjean` (conf: 1.00, lift: 25.22)

### Cluster 614: Burgundy + Placeantoninponcet + Bourgogne (955 photos)

1. `burgundy` → `placeantoninponcet + bourgogne` (conf: 0.85, lift: 31.40)
2. `bourgogne` → `placeantoninponcet + burgundy` (conf: 0.85, lift: 31.40)
3. `burgundy` → `bourgogne + place` (conf: 0.85, lift: 31.40)
4. `bourgogne` → `burgundy + place` (conf: 0.85, lift: 31.40)
5. `burgundy` → `poncet + auvergnerhônealpes` (conf: 0.75, lift: 31.40)

### Cluster 753: Shooting + Photographie (791 photos)

1. `shooting` → `photographie` (conf: 1.00, lift: 21.75)
2. `photographie` → `shooting` (conf: 1.00, lift: 21.75)
3. `shooting + nofilter` → `photographie` (conf: 1.00, lift: 21.75)
4. `photographie + nofilter` → `shooting` (conf: 1.00, lift: 21.75)
5. `shooting` → `photographie + nofilter` (conf: 1.00, lift: 21.75)

### Cluster 431: 프랑스 + 法国 (788 photos)

1. `프랑스` → `法国` (conf: 1.00, lift: 46.06)
2. `法国` → `프랑스` (conf: 1.00, lift: 46.06)
3. `欧洲 + 法国` → `프랑스` (conf: 1.00, lift: 46.06)
4. `프랑스 + 欧洲` → `法国` (conf: 1.00, lift: 46.06)
5. `法国` → `프랑스 + 欧洲` (conf: 1.00, lift: 46.06)

### Cluster 150: Cross + Villeurbanne (548 photos)

1. `cross` → `villeurbanne` (conf: 1.00, lift: 1.02)
2. `villeurbanne` → `cross` (conf: 1.00, lift: 1.02)
3. `cross + insa` → `villeurbanne` (conf: 1.00, lift: 1.02)
4. `insa + villeurbanne` → `cross` (conf: 1.00, lift: 1.02)
5. `cross` → `insa + villeurbanne` (conf: 1.00, lift: 1.02)

### Cluster 78: Blanc + Partipirate + Drapeau (494 photos)

1. `blanc + partipirate` → `drapeau` (conf: 1.00, lift: 37.46)
2. `drapeau` → `blanc + partipirate` (conf: 1.00, lift: 37.46)
3. `noir + partipirate` → `drapeau` (conf: 1.00, lift: 37.46)
4. `drapeau` → `violet` (conf: 1.00, lift: 37.46)
5. `violet` → `drapeau` (conf: 1.00, lift: 37.46)

### Cluster 162: Ldoll2016 + Dd + Rin (453 photos)

1. `ldoll2016 + dd` → `rin` (conf: 1.00, lift: 13.97)
2. `rin` → `ldoll2016 + dd` (conf: 1.00, lift: 13.97)
3. `shooting + ldoll2016` → `rin` (conf: 1.00, lift: 13.97)
4. `rin` → `shooting + ldoll2016` (conf: 1.00, lift: 13.97)
5. `ldoll2016 + dollfie` → `rin` (conf: 1.00, lift: 13.97)

### Cluster 173: Pentaxk10D + Cimetièreloyasse (441 photos)

1. `pentaxk10d` → `cimetièreloyasse` (conf: 1.00, lift: 48.00)
2. `cimetièreloyasse` → `pentaxk10d` (conf: 1.00, lift: 48.00)
3. `pentaxk10d + pentaxart` → `cimetièreloyasse` (conf: 1.00, lift: 48.00)
4. `cimetièreloyasse + pentaxart` → `pentaxk10d` (conf: 1.00, lift: 48.00)
5. `pentaxk10d` → `cimetièreloyasse + pentaxart` (conf: 1.00, lift: 48.00)

### Cluster 603: Street + City + Streetphotography (441 photos)

1. `street` → `city + streetphotography` (conf: 0.70, lift: 33.70)
2. `street` → `ville + streetphotography` (conf: 0.70, lift: 33.70)
3. `city + streetphotography` → `street` (conf: 1.00, lift: 33.70)
4. `ville + streetphotography` → `street` (conf: 1.00, lift: 33.70)
5. `celebration` → `groscaillou + ovo` (conf: 0.73, lift: 30.64)

## Naming Method Distribution

| Method | Count | Percentage |
|--------|-------|------------|
| association_rule | 854 | 92.4% |
| tfidf | 32 | 3.5% |
| frequent_itemset | 28 | 3.0% |
| fallback | 10 | 1.1% |

## Validation: Known Lyon Landmarks

| Expected Landmark | Matched Cluster | Auto-Name | Status |
|-------------------|-----------------|-----------|--------|
| Fourvière | 250 | Dame + Fourvière | ✅ |
| Vieux Lyon | 190 | Boottocht + Lyon2014 | ✅ |
| Place Bellecour | 257 | Bellecour + Saône | ✅ |
| Fête des Lumières | 136 | Eqaflyon + Urlaub | ✅ |
| Demeure du Chaos | 480 | Portrait + Container + Vanitas | ✅ |
| Parc Tête d'Or | 233 | Fresque + Éphémère | ✅ |
| Croix-Rousse | 174 | Villa + Gillet | ✅ |
| Presqu'île | 576 | Flickriosapp + Filter | ✅ |