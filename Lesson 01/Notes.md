## Způsob zakončení
- nepovinná docházka
- zápočet
	- **článek (referát) o technologii**, která je postavená na AI, Hlubokém učení, atd.
		- minimálně 3 A4, ideálně psáno v LaTeX, bude obsahovat abstrakt
	- **praktické využití** toho co jste se naučily k vytvoření nějakého modelu
- téma musí být schválené mnou
- konzultace / prezentace tématu v období zápočtového týdne a chvíli před ním
- termín – v zápočtovém týdnu (nebo chvíli předním)

## Doporučené materiály a knížky
- materiály doc. Konečného ([skripta](http://phoenix.inf.upol.cz/~konecnja/vyuka/2024W/UMIN.html), [prezentace z předchozího roku](http://phoenix.inf.upol.cz/~konecnja/vyuka/2023W/UMIN.html))
- materiály doc. Outraty ([slidy](http://outrata.inf.upol.cz/courses/mldm/mldm.pdf) předmětu **Machine learning a data mining 1**)
- [kurzy Kaggle](https://www.kaggle.com/learn)
	- budu na ně průběžně odkazovat
- Kniha: The StatQuest Illustrated Guide To Machine Learning, Josh Starmer
- Kniha: Data Science from Scratch: First Principles with Python, 2nd Edition, Joel Grus
- Kniha: Deep Learning with Python, 2nd Edition, François Chollet
- youtube.com/@3blue1brown
- colah.github.io
- V průběhu roku budu doplňovat. Pokud máte nějaké doporučení napište mi nebo prokonzultujte se mnou.

## Náplň hodiny
- jak přistupovat k řešení problému z oblasti umělé inteligence?
	- například: [CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/)

### Preprocessing
Předtím než data budeme moci plně využít potřebujeme je co nejlépe připravit na zpracování. Tomuto kroku se říká předzpracování dat (Data preprocessing). 
- Data mohou být nekonzistentní. Mohou obsahovat chybějící nebo nesmyslné hodnoty. 
- Různé techniky modelování vyžadují různé způsoby zpracování.
- Metody zpracování dat:
	- **Integrace dat z různých zdrojů**: Propojení dat z více tabulek.
	- **Výpočet odvozených hodnot** – výpočet výrazu nad zvolenými sloupci (BMI, jednotková cena), SQL dotaz nad zvolenými daty (kolik hostů přijelo v ten samý den), vypočítání hodnot z geografických dat (souřadnice nejbližšího města), odvozením hodnot z analýzy hlavních komponent, hodnoty odvozené z výsledků úloh
	- **Čištění dat**: Odstranění šumu a anomálií. Zpracování chybějících nebo nekonzistentních dat.
	- **Transformace dat:** Normalizace, agregace nebo obecné škálování dat.
	- **Binarizace a diskretizace dat**: Proces převodu spojitých dat na kategoriální data, či binární.
	- **Tvorba příznaků nebo jejich výběr:** Tvorba nových atributů nebo výběr ze stávajících. Anglicky běžně označováno jako **feature creation** a **feature selection**.

- **Numpy** (**Num**erical **Py**thon) nejdůležitější knihovna pro vědu v jazyce Python. Poskytuje objekt vícerozměrného pole a nad ním spoustu různých operací jak matematických, tak i logických, manipulace s tvary, třídění, atd.
	- [Opravdu stručná příručka](https://numpy.org/doc/stable/user/absolute_beginners.html)
	- [Manuál](https://numpy.org/doc/stable/reference/index.html)
- **Pandas** pro analýzu a manipulaci s daty. Data reprezentujeme pomocí tabulek (DataFrame) nebo "slovníku" (Series).
	- [Základní popis s FAQ](https://pandas.pydata.org/docs/getting_started/index.html#)
- **Matplotlib** nejpoužívanější knihovna na vizualizaci dat v Pythonu.
	- Umožňuje vykreslení dat v obrázcích (Figures), kde každý obrázek obsahovat jednu nebo více os (axis).
	- [První krůčky](https://matplotlib.org/stable/users/getting_started/)
	- [Stručná příručka](https://matplotlib.org/stable/users/explain/quick_start.html)
	- [Tahák – Typy grafů v Matplotlib](https://matplotlib.org/stable/plot_types/index.html)
- **Seaborn** je knihovna pro vizualizaci dat postavená nad matplotlib. Kód psaný v seaborn je snazší a přehlednější.
	- [Tahák – Typy grafů v Seaborn](https://seaborn.pydata.org/examples/index.html)
#### Kurzy pro Preprocessing
- základní práce s knihovnou **Pandas**:  https://www.kaggle.com/learn/pandas
- jak přehledně vizualizovat data knihovnou **Seaborn**: https://www.kaggle.com/learn/data-visualization
- základní zpracování reálných dat https://www.kaggle.com/learn/data-cleaning
### Klasifikace a regrese
**Klasifikace** – predikce **diskrétního** cílového atributu
- například klasifikace nevyžádané pošty ([SPAM / HAM klasifikace](https://www.kaggle.com/code/jacquelinehong/spam-ham-classifier)) nebo klasifikace recenzí filmů jako pozitivních nebo negativních

**Regrese** – predikce **spojitého** cílového atributu, například [odhad ceny domu na základě údajů o nemovitostech](https://www.kaggle.com/code/mahyamahjoob/real-estate-valuation-using-linear-regression), atd.

### Perceptron 
- vysvětlení v Jupyter Notebooku ke cvičení 01.
- někdy také neuron, linear unit
- $y = wx + b$ (rovnice přímky ve 2D / rovnice roviny ve 3D)
	- kde y je výstup sítě
	- x je vstup, který je vynásobený váhovým vektorem w (sklon přímky)
	- b je bias (průsečík přímky)

### Praktický příklad neuronu
- jednoduchá síť vytvořená pomocí knihovny Keras
```python
from tensorflow import keras
from tensorflow.keras import layers

# Model obsahuje jeden perceptron
model = keras.Sequential([
    layers.Dense(
	    units=1,       
		    # kolik perceptronů je ve vrstvě
		input_shape=[3]) 
			# rozměry vstupu
])
```
- váhy jsou reprezentovány tenzory
	- tenzor = zobecnění pojmu vektor
	- tenzory jsou optimalizovány pro výpočet na GPU (TPU)
	- váhy se inicializují náhodně