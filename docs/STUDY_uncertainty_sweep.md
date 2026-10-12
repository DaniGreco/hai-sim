# STUDY — RQ1 su larga scala: ordinare per incertezza (sweep incertezza → errore)

> Documento di **disegno** dello studio. Definisce, per ogni scelta, *cosa* è, *perché* la
> facciamo così, e *a cosa* si applica. Non è codice: è la specifica da cui nascono sia lo
> script di sweep sia la sezione di tesi. Il contesto è in `TESI.md` (§3, RQ1).
> Lingua: italiano, come `TESI.md`; acronimi espansi alla prima occorrenza.

**Stato deciso con Daniele (2026-08-07).** Calibrazione = **Platt per-patologia**. Costo
degli errori = **falso negativo : falso positivo = 10 : 1**, uniforme sulle 18 patologie.
Fase: **solo progettazione** (nessuno script scritto ancora).

---

## 0. La domanda (RQ1)

> **Ordinare i casi per incertezza del modello fa emergere per primi quelli su cui l'AI
> sbaglia?** (`TESI.md` §3). Se sì, un triage che mostra al medico prima i casi "dubbi" è
> giustificato.

Lo studio estende RQ1 lungo tre assi che i dati raccolti ora permettono e che prima non
c'erano: (a) **modello** — un classificatore discriminativo (ensemble di reti) vs modelli
generativi (MedGemma 4B e 27B); (b) **dominio** — NIH (in-dominio) vs VinDr (fuori dominio,
*out-of-domain*, OOD); (c) **punto operativo** — la soglia decisionale e, dove conta, la
calibrazione. Il cuore resta: *l'incertezza trova gli errori?*, decomposto in **falsi
negativi (FN)** e **falsi positivi (FP)**.

---

## 1. Il cubo di dati (le celle di cache)

Ogni cella conserva le **probabilità grezze per-membro** `(N immagini, M membri, 18
patologie)` + la verità + il tempo di inferenza. Tutto ciò che segue si deriva dalla cella
**senza rifare inferenza**: misura di incertezza, soglia, calibrazione, aggregazione,
definizione di errore.

### 1.1 Modelli (asse "modello")

| alias | cos'è | membri (M) | incertezza epistemica? |
|---|---|---|---|
| **ensemble** | Deep Ensemble cross-dominio di DenseNet-121 (TorchXRayVision) | **4**: `chex`, `mimic_ch`, `pc`, `rsna` | **sì** (disaccordo tra membri) |
| **medgemma-1.5-4b** | MedGemma 1.5 4B, generativo, precisione BF16 | **1** | **no** |
| **medgemma-27b** | MedGemma 27B (gen 1), generativo, precisione FP8 | **1** | **no** |

- **Perché 4 membri e non 5:** il quinto membro (`densenet121-res224-nih`) è addestrato
  sullo stesso NIH su cui valutiamo → *leakage* in-dominio (e in parte memorizzazione sul
  nostro campione). Scartato di proposito (`TESI.md` §6.4-bis).
- **Perché MedGemma è a un membro solo:** costruito con `prompt_set=single` (una sola
  parafrasi di prompt). Conseguenza da dichiarare: per MedGemma **non esiste incertezza
  epistemica** (vedi §2). È un risultato, non un difetto: un ensemble sa dire "questo caso
  è fuori dal mio dominio", un singolo modello generativo no.

### 1.2 Dataset (asse "dominio")

| id | cos'è | ruolo | N immagini |
|---|---|---|---|
| **nih** | NIH ChestX-ray14, campione | **in-dominio** (i membri non-scartati non l'hanno visto, ma è il dominio radiografico "classico") | ~30.8k |
| **vindr** | VinDr-CXR (1024px), etichette k-di-3 radiologi, mappate nello schema xrv | **fuori dominio (OOD)** — nessun membro addestrato su VinDr | 15k |

Il contrasto NIH↔VinDr **è** un asse dello studio, non un disturbo: l'incertezza epistemica
dovrebbe *salire* passando OOD.

### 1.3 Patologie (schema `xrv`, 18)

`Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis,
Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia, Lung Lesion,
Fracture, Lung Opacity, Enlarged Cardiomediastinum`. Ogni dataset annota solo un
sottoinsieme; le patologie non annotate da un dataset restano **fuori da ogni metrica**
(vedi §1.4).

### 1.4 Verità (ground truth) — convenzione `{1, 0, -1}`

Per ogni coppia (immagine, patologia): `1` = presente, `0` = assente, `-1` = **non
annotata** da quel dataset. **Le entries a `-1` non sono mai contate come errore né come
successo**: sono non misurabili, si escludono e basta (non si trasformano mai in uno `0`).

---

## 2. Misure di incertezza (la chiave di ordinamento)

L'incertezza è **il numero per cui si ordina**: cosa significa decide cosa vuol dire "caso
più difficile". Tutte si calcolano dalla pila `(M, 18)` di probabilità per-membro. `p_m` =
probabilità del membro *m*; `mean_p` = media (che ignora i `NaN`) sui membri.

| misura | formula | cosa cattura | disponibile su |
|---|---|---|---|
| **predictive_entropy** (entropia predittiva) | `H(mean_p)` | incertezza **totale**: quanto la probabilità media è vicina a 0.5 | tutti |
| **mean_entropy** (entropia media) | `media_m H(p_m)` | incertezza **aleatoria**: ambiguità *nell'immagine* — i membri sono tutti incerti e **concordano** nell'esserlo. Più dati non aiuterebbero | tutti (degenere su MedGemma) |
| **mutual_information** (informazione mutua) | `predictive_entropy − mean_entropy` (≥0) | incertezza **epistemica**: i membri **disaccordano** → immagine diversa da ciò su cui erano addestrati. È quella che dovrebbe segnalare l'"out-of-depth" | **solo ensemble** |
| **std** (deviazione standard) | `std_m p_m` | disaccordo grezzo tra membri | **solo ensemble** |
| **least_confidence** (minima confidenza) | `min(mean_p, 1 − mean_p)` | distanza dalla decisione netta: alta vicino a 0.5, ~0 vicino a 0/1. Per singola patologia è **rank-equivalente** a `predictive_entropy` | tutti |

`H(p)` = entropia binaria di Bernoulli(p): massima a p=0.5, nulla a p∈{0,1}.

**Degenerazione su MedGemma (M=1), da dichiarare:** con un solo membro
`mean_entropy ≡ predictive_entropy`, `mutual_information ≡ 0`, `std ≡ 0`. Quindi per MedGemma
solo `predictive_entropy` / `least_confidence` (incertezza **da confidenza**) sono sensate.
L'asse "confronta le misure di incertezza" è pieno **solo per l'ensemble**.

---

## 3. Definizione di errore e i due tipi (FN / FP)

Data la decisione binaria `pred = (probabilità dopo calibrazione ≥ soglia)` e la verità `gt`,
sulle **sole entries annotate** (`gt ≠ -1`):

- **FN — falso negativo (*miss*, caso mancato):** `pred=0` e `gt=1`. La patologia c'era, non
  è stata segnalata. **In radiologia è l'errore grave.**
- **FP — falso positivo (falso allarme):** `pred=1` e `gt=0`. Segnalata una patologia che non
  c'era.
- **any (errore qualsiasi):** `pred ≠ gt`.

Lo studio produce le curve e gli scalari **separatamente per FN, FP e any**: è la richiesta
centrale di Daniele ("curve di FN o FP ordinati per incertezza").

---

## 4. Viste / livello di aggregazione

La stessa cella si può guardare a tre granularità; ognuna risponde a una domanda diversa.

| vista | unità | come nasce | a cosa serve |
|---|---|---|---|
| **per-immagine** | una lastra | l'incertezza delle 18 patologie è **aggregata** in un numero per immagine, con criterio `mean` (media) o `max` (massimo) | è la vista del **triage**: si ordina una coda di *immagini* |
| **per-entry** | una coppia (immagine, patologia) annotata | nessuna aggregazione | l'oggetto statistico più pulito, decomponibile |
| **per-patologia** | coppie (immagine, patologia) di **una** label | ogni patologia sul suo insieme | separa *dove* l'incertezza funziona da *dove* no; evita che la patologia più frequente o più incerta domini l'ordinamento |

Il criterio label→immagine (`mean` vs `max`) tocca **solo** la vista per-immagine.

---

## 5. Il modello di costo (10 : 1)

Serve a scegliere la soglia in modo ragionato (§6) e a ordinare i risultati (§9).

- **Costo di un falso positivo (FP):** `c_FP = 10`.
- **Costo di un falso negativo (FN):** `c_FN = 100`.
- **Rapporto:** `c_FN : c_FP = 10 : 1` — un caso mancato pesa come 10 falsi allarmi.
  **Uniforme sulle 18 patologie** (scelta di Daniele; un peso per-patologia resta un possibile
  secondo giro, non ora).
- **Costo totale su un insieme:** `C = c_FP·(#FP) + c_FN·(#FN) = 10·#FP + 100·#FN`.
- **Soglia teorica (formula, su probabilità *oneste*):** la teoria della decisione dice di
  dire "presente" quando la probabilità supera
  `τ* = c_FP / (c_FP + c_FN) = 10/110 ≈ 0.091`. Traduzione: con un miss 10× più caro, conviene
  segnalare anche a solo ~9% di probabilità → **rete larga, soglia bassa**. ⚠️ Vale **solo se
  la probabilità è calibrata** (vedi §6 e §7).

---

## 6. Politiche di soglia (il punto operativo — la leva di FN/FP)

La soglia trasforma un numero in "presente/assente" e **governa il bilancio FN↔FP**:
abbassarla → meno FN, più FP; alzarla → il contrario. Non riduce entrambi. È una **scelta di
progetto**, e lo studio ne confronta diverse. Distinzione che decide tutto (vedi §8):

- **soglia di rango (relativa):** dipende solo dall'*ordine* dei punteggi (o dei loro
  quantili). → **immune alla calibrazione** (§7).
- **soglia assoluta:** un valore fisso sulla scala della probabilità. → **la calibrazione la
  sposta**.

| # | politica | definizione | natura | calibrazione conta? |
|---|---|---|---|---|
| S0 | **0.5 fisso** | `pred = prob ≥ 0.5`, uguale per tutte le patologie | assoluta | **sì** |
| S1 | **costo-ottimo empirico 10:1** *(PRIMARIA)* | per **ogni patologia**, la soglia che **minimizza** `10·#FP + 100·#FN`, cercata direttamente sui dati etichettati | di rango | no |
| S2 | **frequenza-matched** *(ancora neutra)* | per ogni patologia **e dataset**, soglia al quantile `(1 − prevalenza)` dei punteggi → predetti-positivi = positivi reali → **FP = FN** | di rango | no |
| S3 | **formula τ\*=0.091** | `pred = prob_calibrata ≥ 0.091` | assoluta | **sì** (ha senso solo su prob calibrate) |
| S4 | **sensibilità-fissa** *(opzionale)* | per patologia, soglia tale che il *recall* = target (es. 0.90) | di rango | no |

**Definizioni estese e razionale:**

- **S0 (0.5 fisso):** il default ingenuo. Serve come **baseline-da-battere**: su un modello
  troppo sicuro di sé (MedGemma sovra-chiama) produce un profilo FN/FP specifico e cattivo, e
  mostrare quanto lo si migliora scegliendo il punto operativo è metà del risultato.
- **S1 (costo-ottimo 10:1) — la primaria.** È il punto operativo *clinicamente onesto*: dati
  i costi, taglia dove il costo totale è minimo. Ragionamento marginale: si abbassa la soglia
  finché ogni malato recuperato (−100) vale i falsi allarmi che costa (+10 ciascuno). Con
  10:1 la rete è larga, FN tende a 0. È **di rango** (sceglie una *posizione* di taglio) →
  la calibrazione non la tocca.
- **S2 (frequenza-matched) — l'ancora neutra.** Rende il numero di predetti-positivi uguale
  ai positivi reali; algebricamente questo impone **FP = FN** (numero di falsi allarmi =
  numero di miss). È il taglio che tratta i due errori come *ugualmente gravi* — utile come
  riferimento pulito e riproducibile, **ma non** clinicamente ottimo quando FN pesa più di FP
  (è il nostro caso). È anche la **correzione standard del base-rate** tra domini: rifarla
  per-dataset assorbe lo spostamento di frequenza NIH↔VinDr.
- **S3 (formula τ\*):** la via teorica. Inclusa soprattutto come **cella cautelativa**:
  applicata ai punteggi *grezzi* (gonfiati) è sbagliata per costruzione; ha senso **solo**
  sulla probabilità calibrata. Mostra concretamente perché la formula richiede la
  calibrazione.
- **S4 (sensibilità-fissa):** l'ottica dello screening ("voglio prendere il 90% dei
  malati"). Di rango. Opzionale.

⚠️ **Disciplina held-out anche per la soglia.** S1, S2, S4 *guardano le etichette* per essere
scelte. Per non barare vanno **fittate su una fetta e riportate sull'altra**, esattamente
come la calibrazione (§7). Fissare la soglia e riportare l'errore sugli stessi casi
sovrastima.

---

## 7. Calibrazione (l'onestà del numero — asse ortogonale)

**Calibrare = riscalare le probabilità perché dicano la verità su se stesse** (dei casi
chiamati "0.8", ~80% sono davvero positivi). È una domanda **diversa** dalla decisione: la
soglia decide *l'azione*, la calibrazione decide *il significato del numero*.

### 7.1 Su cosa la fittiamo — le quattro scelte

1. **Contro cosa:** la **verità osservata** su una **fetta held-out disgiunta** (es. split
   50/50 per seed), minimizzando la *negative log-likelihood* (NLL, "verosimiglianza negativa"
   — premia chi assegna numeri vicini alla realtà). **Mai** fittare e riportare sugli stessi
   casi (è circolare).
2. **Che forma (famiglia):**
   - **Platt** *(PRIMARIA)* — `sigmoid(a·logit(p) + b)`, due parametri per patologia. Il termine
     `b` **sposta** l'intera distribuzione, `a` la **ammorbidisce**. Serve lo spostamento
     perché MedGemma **sovra-chiama sistematicamente** (non è solo gonfiato, è *sbilanciato*
     verso il "presente") → la temperatura da sola è la forma sbagliata.
   - **Temperature (scaling di temperatura)** *(CONFRONTO)* — `sigmoid(logit(p)/T)`, un
     parametro. Solo ammorbidisce. Riferimento più semplice.
   - **Isotonic (isotonica)** *(SOLO CONTROLLO)* — monotona non-parametrica; si adatta a
     qualunque forma ma va in overfit sui dati scarsi → solo diagnostica.
3. **Grana:** **per-patologia** (una `(a,b)` per label). Patologie frequenti e rare sono
   gonfiate in modo diverso. Con le celle piene (15k–30k) il per-patologia è ora affidabile
   (a 200 immagini tornava quasi tutto "identità", cioè niente da fittare).
4. **Per quale popolazione:** **rifittata per ogni (modello × dataset × prompt)**. Un
   calibratore tarato su NIH **non** vale su VinDr, e per un modello generativo non vale
   nemmeno per un prompt diverso. Rifittare per dominio è ciò che assorbe lo shift di
   frequenza (il "prior shift").

### 7.2 Cosa la calibrazione NON tocca (invarianze — perché è monotona)

Platt/temperature/isotonic sono **monotone** (riscalano, non riordinano). Quindi lasciano
**identici**:

- l'**AUROC di discriminazione** (§9) — è ranking puro;
- il **ranking per incertezza** e quindi le curve incertezza→errore, `skill`, AUROC-incertezza;
- **ogni soglia di rango** (S1, S2, S4): "il 30% più alto" è lo stesso gruppo prima e dopo.

Si muovono **solo**: FN/FP a **soglia assoluta** (S0, S3), l'**ECE** e la curva di
affidabilità (§9).

### 7.3 Il punto che evita l'errore di impostazione

> **Calibrare e tenere la soglia a 0.5 NON dà la decisione ottima.** Anche con probabilità
> oneste, con costo 10:1 la soglia giusta è ~9%, non 50%. Per la *decisione* la leva è la
> **soglia dai costi** (S1); la calibrazione **non è né necessaria né sufficiente**. Il vero
> mestiere della calibrazione è un **altro studio**: la fusione medico↔AI (dove un numero
> gonfiato schiaccia il partner) e la domanda "la confidenza è affidabile?".

---

## 8. Interazione soglia × calibrazione — quali combinazioni sono davvero diverse

Non è una griglia piena: per §7.2, su **soglia di rango la calibrazione è un no-op**. Quindi
la griglia informativa è piccola (per ogni modello × dataset):

| soglia \ calibrazione | grezzo | Platt | note |
|---|---|---|---|
| **S0 — 0.5 fisso** | ● | ● | **diversi** — qui si *vede* l'effetto calibrazione |
| **S3 — τ\*=0.091** | ● (sbagliato per costruzione) | ● (via-formula fatta bene) | **diversi** |
| **S1 — costo-ottimo 10:1** | ● | — *(identico)* | **una cella** (di rango) |
| **S2 — frequenza** | ● | — *(identico)* | **una cella** (di rango) |
| **S4 — sensibilità 90%** | ● | — *(identico)* | **una cella** (di rango) |

→ **~6 celle informative** per (modello × dataset), non 5×3=15. Le celle "identico" si
calcolano una volta sola.

---

## 9. Metriche riportate (gli scalari della tabella-madre)

**Sull'incertezza (il cuore di RQ1):**

- **`skill`** — concentrazione cumulata degli errori ordinando per incertezza **decrescente**:
  `(area_actual − area_random) / (area_oracle − area_random)`, ≤ 1. `oracle` = tutti gli
  errori davanti; `random` = ordinamento non informativo (diagonale). **1** = oracolo, **0** =
  come il caso, **< 0** = anti-correlato (l'incertezza indica gli errori al contrario).
  Analogo cumulativo dell'AUSE / concentrazione di Gini.
- **`auroc_unc`** — AUROC dell'incertezza nel predire l'errore: probabilità che un'unità
  sbagliata sia più incerta di una corretta. *Threshold-free* (non dipende dalla soglia).

**Sulla decisione (FN/FP):**

- **`fn_rate`, `fp_rate`** — `#FN/N`, `#FP/N` sulle entries annotate.
- **`cost`** — costo pesato totale `10·#FP + 100·#FN` (e costo medio per caso). **È la colonna
  su cui si ordina** la tabella per rispondere "quale combinazione è migliore".

**Sul modello (indipendenti da soglia e calibrazione):**

- **`auroc_disc`** — AUROC di discriminazione per-patologia: probabilità che un positivo sia
  scorato sopra un negativo. Capacità di *ranking* grezza del modello.
- **`ece`** — *expected calibration error*: media di |confidenza − frequenza osservata| a bin,
  pesata per popolazione del bin. **0** = calibrato. Si muove con la calibrazione (le altre no).

**Per il confronto tra modelli (pannello opzionale, §5 complementarità di `TESI.md`):**

- **`kappa`** — κ di Cohen sull'**overlap degli errori** di due modelli: `0` = sbagliano su
  casi indipendenti (la coppia può aiutare), `1` = sbagliano sugli stessi casi, `<0` =
  complementari. È il substrato dell'argomento di complementarità uomo-AI.

---

## 10. La tabella-madre e le figure

**Tabella-madre** (un CSV/JSON unico), una riga per combinazione:

```
righe: modello × dataset × misura × vista × tipo-errore × politica-soglia × calibrazione
colonne: skill, auroc_unc, fn_rate, fp_rate, cost, auroc_disc, ece, n_errors, n_units
ordinamento primario: cost (crescente), dentro ogni (modello, dataset)
```

**Figure** (riuso di `error_curve.py` e `model_comparison.py`):

- l'albero di curve cumulate FN/FP/any già previsto, sotto
  `out/analysis/<modello>/<dataset>/<misura>/thr<politica>/...`;
- **overlay di confronto** (nuovi): a parità di (dataset, tipo-errore), sovrapporre le curve
  *actual* di {ensemble-MI, ensemble-entropy, MedGemma-entropy} → si *vede* chi trova prima
  gli errori;
- pannello **calibrazione** separato (curva di affidabilità + ECE, grezzo vs Platt,
  per-patologia, su held-out).

---

## 11. Ipotesi / risultati attesi (da confermare o smentire)

1. L'incertezza **epistemica** (mutual_information) dell'ensemble concentra gli errori meglio
   dell'entropia di un modello singolo.
2. Passando **OOD** (VinDr) l'incertezza *sale* e (si spera) resta predittiva degli errori;
   se crolla, è comunque un risultato forte.
3. **Asimmetria FN/FP:** l'incertezza trova meglio i falsi allarmi o i miss? Clinicamente i FN
   pesano di più.
4. **27B ≳ 4B** in discriminazione e in qualità dell'incertezza (il pilot suggeriva Fibrosis
   AUROC 0.86 vs 0.63).
5. Sulla decisione: `(0.5 · grezzo)` pessimo su MedGemma; `(costo-ottimo 10:1)` domina;
   `(0.5 · calibrato)` migliora ma **resta sopra** il costo-ottimo (perché 0.5 ≠ 9%) →
   dimostrazione pulita che *calibrare non sostituisce lo scegliere la soglia*.

---

## 12. Onestà metodologica / limiti (da dichiarare nella tesi)

- **MedGemma è a un membro** → niente incertezza epistemica; il confronto delle *misure* di
  incertezza è pieno solo per l'ensemble.
- **27B (FP8, gen 1) vs 4B (BF16, gen 1.5):** cambiano **dimensione, generazione e
  precisione** insieme → **non** è un confronto di scala pulito. Va descritto così.
- **Prevalenze diverse** NIH vs VinDr → soglie di frequenza/costo **per-dataset**, altrimenti
  lo shift di base-rate OOD si traveste da cedimento del modello.
- **Sia la calibrazione sia la soglia empirica "usano le etichette"** → disciplina held-out
  obbligatoria (fit su una fetta, report sull'altra).
- **`π_train` non è pulito** (ensemble = blend di 4 dataset; MedGemma generativo non ha un
  prior da classificatore) → niente correzione prior *analitica*; si usa il **rifit per
  dominio**, che l'assorbe dai dati.
- **Etichette NIH deboli** su alcune patologie (Infiltration, Pneumonia) → AUROC bassi lì sono
  in parte rumore d'etichetta, non solo modello.

---

## 13. Cosa serve costruire (il gap tecnico — quando si passa dall'implementazione)

Tutto poggia su moduli **già esistenti** (`analysis/error_curve.py`,
`analysis/model_comparison.py`, `uq/calibration.py`, `analysis/inference_cache.py`). Manca:

1. **`compose_cells_to_cache(member_stems) -> InferenceCache`** — carica i 4 cell-membro
   dell'ensemble, allinea per `image_id`, impila sull'asse membri `(N, 4, 18)`. È l'unico
   vero buco (la composizione oggi vive solo dentro l'attore `CompositeCachedAI`, per il
   simulatore, non per l'analisi offline).
2. **soglia empirica per-patologia con held-out** (S1/S2/S4) — non c'è ancora un selettore di
   soglia guidato dal costo.
3. **driver di sweep** — itera la griglia (§8), calibra (§7), calcola gli scalari (§9),
   scrive la tabella-madre e gli overlay (§10).

Riuso diretto: le curve, `skill`/AUROC, il per-patologia, il confronto modelli e i
calibratori Platt/temperature con i loro fitter sono già implementati e testati.
