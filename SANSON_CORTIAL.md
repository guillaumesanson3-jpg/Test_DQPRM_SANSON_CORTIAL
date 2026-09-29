Test exercice Git SANSON Guillaume et CORTIAL Johan 29/09/2026


## Exercice 3 - Dosimétrie
-----------

### Contexte
**Radioembolisation** dans le traitement d'un cancer hépatique à l'aide de **microsphères de verre** marquées à l'$^{90}Y$.

Planification de l'activité à administrer à l'aide d'une acquisition tomographique réalisée au $^{99m}Tc$-MAA

- Déterminer l'activité d'$^{90}Y$ pour délivrer une dose absorbée limite de 120 Gy au lobe hépatique contenant la tumeur
- Déterminer la dose absorbée à la tumeur

Pour illustrer le propos de l'impact de la dosimétrie prévisionnelle dans le traitement des hépatocarcinome, voir [ici](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4731431/#!po=84.5455).

### Modéle de partionnement

[Ho et al.](https://www.ncbi.nlm.nih.gov/pubmed/8753684) ont défini un modèle de calcul basée sur la connaissance de **la répartition de l'activité** dans le foie après la perfusion des microsphères radioactives. Le calcul de la dose absorbée aux  volumes d'intérêt est estimée par la méthologie du *MIRD*.

L'activité perfusée dans le foie se répartie dans lui-même et les poumons s'il existe un *shunt* entre ce premier et ces derniers.

- L'activité dans les poumons est estimée par $A_L = A_{inj.} \times \frac{L}{100}$
    - L pourcentage de shunt pulmonaire
- L'activité dans le foie comprenant la partie saine ($A_N$) et tumorale ($A_T$) est estimée par $A_N+A_T = A_{inj.} (1-\frac{L}{100})$
- Le rapport tumeur/foie sain $r=\frac{\frac{A_T}{m_T}}{\frac{A_N}{m_N}}$ peut être estimé à partir des pseudo-concentrations d'activité mesurées par la segmentation dans la tumeur et le foie sain. A l'aide de l'équation précédente, on peut ensuite exprimer les activités dans le foie sain ($A_N$) et dans la tumeur ($A_T$) en fonction de ce rapport et de $A_{inj.}$.

### Rappels

#### Equation du MIRD

$$ \bar{D}_{k \leftarrow h} = \sum_{h} \tilde{A}_{h} \times S_{k \leftarrow h} $$

où $\tilde{A}_{h}$ est l'activité cumulée dans la source i.e: le **nombre total de désintégration dans la source h** et $S_{k \leftarrow h}$ **le facteur S** liant la source h à la cible k.

#### Equation simplifiée
Dans le cas la cas d'une **radioembolisation**,
- toute l'activité injectée est piègée dans le foie (si pas de *shunt pulmonaire*)
- seule la décroissance physique du radionucléide intervient (pas d'élimination biologique du traceur).

Cela simplifie le calcul

$$\bar{D}_{foie} = A(0)_{foie} \times \frac{T_{phys.}}{ln\,2} \times S_{foie \leftarrow foie}$$

Dans le cas où on utilise un radionucléide qui émet **uniquement des émissions $\beta^-$**, la dernière équation est équivalente à :

$$ \bar{D}_{foie} = A(0)_{foie} \times \frac{T_{phys.} \times \Delta}{ln\,2\times m_{foie}}$$

où $\Delta$ représente **l'énergie totale émise par transition** et $m_{foie}$ la masse du foie.

En réorganisant les équations, on obtient l'activité à injecter pour une dose absorbée déterminée

$$ A(0)_{foie} = \frac{\bar{D}_{foie} \times m_{foie} \times ln\,2}{T_{phys.}\times \Delta}$$

Dans le cadre d'un traitement par radioembolisation avec des µ-sphères de verre, on souhaite délivrer une dose absorbée de 120 Gy dans **l'ensemble du foie perfusé**.

**Question 1.** Lire avec Pandas le fichier `Table.csv` contenu dans le dossier `data` qui contient les valeurs des différents volumes d'intérêt ainsi que les activités dans ces volumes (attention au format du séparateur de colonnes). La première colonne sera utilisée comme index des lignes.

**Question 2.** Ajouter une colonne au tableau avec les masses des différents volumes d'intérêt (on prendra comme valeur de masse volumique $\rho=1.03\ g/cm^3$)

**Question 3.** Déterminer l'activité à injecter dans le lobe droit pour atteindre cette dose absorbée limite en utilisant l'équation simplifiée du MIRD

**Question 4.** Déterminer la dose absorbée à la tumeur pour cette activité injectée

NB. Il n'y a pas eu de shunt pulmonaire identifié durant cette procédure

Données :
* Période de l'yttrium 90 : 64,05 $heures$
* Energie totale émise par transition : 0.9336 $\frac{MeV}{Bq.s}$
* On considère que les tissus hépatiques et la tumeur ont une masse volumique égale à 1.03 $\frac{g}{cm^3}$


Première cellule de code pour répondre à la première question :

import pandas as pd

df = pd.read_csv("data/Table.csv", delimiter="\t", index_col=0)
df

rho_g_per_cm3 = 1.03
df['Masse [g]'] = df["Volume [cm3]"]*rho_g_per_cm3
df

m_foie_lobe_d = df.loc["lobe_droit",'Masse [g]']

import numpy as np

delta_Mev_per_Bq_s = 0.9336
T_y90_s = 64.05*3600
dose_foie_limite_Gy = 120
act_1 = (dose_foie_limite_Gy*m_foie_lobe_d*1e-3*np.log(2))/(T_y90_s*delta_Mev_per_Bq_s*1e6*1.602e-19)
print(f"L'activité à injecter est de {act_1*1e-9:.2f} GBq pour atteindre {dose_foie_limite_Gy} Gy au lobe droit.")

Ainsi le réponse est que l'activité à injecter est de 2.01 GBq pour atteindre 120 Gy au lobe droit.

## La seconde question est : D'après le modèle de partitionnement, on doit estimer le  rapport de concentration entre la tumeur et le foie perfusé,

# a fair rapport des mean sur rapport des masses....
ratio_tum_lobe = (df.loc["tum_dome_SPECT", "Mean"]*df.loc["lobe_droit", "Masse [g]"])/(df.loc["tum_dome_SPECT", "Masse [g]"]*df.loc["lobe_droit", "Mean"])
print(f"Le rapport des concentrations est estimé à {ratio_tum_lobe:.2f}")

Le rapport des concentrations est estimé à 287.78

Dans le cas d'**absence de shunt pulmonaire**, les équations du modèle de partionnement deviennent:

$$
\begin{align}
A_T  &= r \times A_N \frac{m_T}{m_N}\\
A_N  &= \frac{A_{inj.}}{(1 + r \times \frac{m_T}{m_N})}
\end{align}
$$

m_tum = df.loc["tum_dome_SPECT", "Masse [g]"]
m_tot = df.loc["lobe_droit", "Masse [g]"]
A_n = act_1/(1+ratio_tum_lobe*(m_tum/m_tot))
A_t = ratio_tum_lobe*A_n*(m_tum/m_tot)
print(f'Les activités dans le foie perfusé et la tumeur sont {A_n*1e-9:.2f} et {A_t*1e-9:.2f} GBq respectivement.')

Les activités dans le foie perfusé et la tumeur sont 0.45 et 1.56 GBq respectivement.

On peut alors estimer la dose à la tumeur

dose_t = (A_t*T_y90_s*delta_Mev_per_Bq_s*1.602e-19*1e6)/(m_tum*1e-3*np.log(2))
print(f'La dose à la tumeur est {dose_t:.2f} Gy')

La dose à la tumeur est 7675.14 Gy

