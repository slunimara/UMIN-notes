- Zdroje: Kniha – Deep Learning with Python, [Rozdělení dat – scikit learn](https://scikit-learn.org/stable/modules/cross_validation.html)
- Kaggle kurz – Intro to Deep Learning kapitola 4. a 5.

- u našich modelů chceme zajistit dvě vlastnosti:
	- vlastnost **adaptace** – model se naučí fungovat na základě předložených (testovacích) dat
	- vlastnost **generalizace** –  model správně funguje i pro data, která nikdy neviděl (tj. jiná než trénovací)
- **parametry** sítě vs **hyperparametry** sítě
	- parametry sítě – váhy a bias sítě
	- hyper-parametry sítě – architekturu modelu, rychlost učení a složitost modelu (počet vrstev a jejich velikost), $\dots$
- [ladění hyperparametrů základní informace](https://aws.amazon.com/what-is/hyperparameter-tuning/)
- [Jak na to v Keras?](https://keras.io/guides/keras_tuner/) alternativně [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)
## Overfitting a underfitting
- **Underfitting** (podučení) znamená, že model je příliš jednoduchý a špatně modeluje trénovací data (tj. nemá naučené relevantní vzory)
- **Overfitting** (přeučení) znamená, že model příliš přesně modeluje trénovací data a v důsledku toho špatně generalizuje
- na začátku trénování modelu je většinou adaptace a generalizace korelována (tj. snižuje se chyba jak nad trénovacími tak i testovacími daty)
- následně v určité fázi trénování modelu se začne adaptace zvyšovat zatímco se generalizace snižuje – model se začne učit vzory, které jsou příliš specifické pro trénovací data a obecně jsou zavádějící 
- nejlepším řešením je získat více dat pro trénovaní – to však není v některých případech možné 
- druhým nejlepším řešením je model nějak omezit (například tak, aby se naučil jen vhodné vzory z dat)
	- proces při kterém se snažíme zamezit přeučení se nazývá **regularizace**
### Techniky regularizace
- **early stopping** – metoda, při které je učení zastaveno před tím než se síť začne učit nevhodným vzorům
- najít **kompromis mezi velikostí** sítě – sítě, které jsou malé se těžce naučí komplexní vzory, naopak velké sítě se naučí i nedůležité vzory
	- pro vytvoření architektury sítě není žádný recept, jde primárně o zkušeností a hlavně zkoušet
	- doporučení však je jít od jednoduššího ke komplexnímu
	- zkuste se řídit principem [Occamovy břitvy](https://cs.wikipedia.org/wiki/Occamova_břitva)
- **regularizace vah** – donutíme síť, aby její váhy nabývaly pouze malých hodnot a tím zlepšíme rozložení vah (při učení může totiž dojít k tomu, že nekteré váhy jsou vysoké a nekteré zase nízké)
	- technicky: k chybové funkci se přičte cena spojena s tím, že síť má velké váhy
	- jsou 2 možnosti: L1 regularizace, L2 regularizace
- **zahození (dropout)**
	- vrstva, která náhodně vynuluje některé výstupy předešlé vrstvy při procesu učení
	- **dropout rate** určuje jak velká část výstupu je vynulována
		- většinou v rozmezí od $0.2$ do $0.5$
	- při testování je pak každý vstup vynásoben parametrem **dropout rate**
	- pozor! přidání dropout vrstvy může zlepšit generalizaci, ale může také zvýšit potřebu přidání více neuronů do vrstvy a dobu tréninku
## Rozdělení dat
- proč rozdělovat data na trénovací, ověřovací a testovací?
	- proč nerozdělovat jen na trénovací a testovací?
- **trénovací množina**: data použita k učení sítě, tj. k úpravě parametrů (tj. vah) klasifikátoru
- **ověřovací množina (validation set)**: data využita pro nastavení a úpravu hyper-parametrů (tj. architektury, nikoli vah) klasifikátoru
- **testovací množina**: data využita pouze pro posouzení výkonnosti (generalizace) vytvořeného klasifikátoru
- nastavení a úprava hyper-parametrů by měla vždy probíhat jen na základě trénovacích a ověřovacích dat
	- z trénovacích dat určíme základní architekturu
	- ověřovací data, využijeme pro úpravu architektury a zlepšení generalizace 
	- při hledání té nejvýhodnější konfigurace hyper-parametrů můžeme dojít k přeučení nad ověřovacími daty, přestože je model nikdy neviděl při trénování
		- **proč?** můžeme si to představit tak, že pokaždé když upravíme hyper-parametry sítě na základě výkonnosti modelu nad ověřovacími daty, tak nějaká informace pronikne i do samotného modelu 
		- tomuto fenoménu se říká **information leak**
	- berte to však s rezervou, pokud jste si jistí, že k úniku informací nedojde (neprovádíte nějakou hyper-optimalizaci na základě testovacích dat), klidně použijte pouze dvě množiny
- pokud jsme spokojeni s hyper-parametry sítě a s výkonnosti modelu nad ověřovacími daty, můžeme model natrénovat nad sjednocením trénovacích a ověřovacích dat a následně změřit jeho výkonnost nad testovacími daty
- po získání konečných hyper-parametrů získáme výsledný model tím, že natrénujeme model na základě všech dat
### Hold-Out validation
- nejjednodušší rozdělení kdy, data jsou rozdělena 2 až 3 množiny 
- 80% trénovací data / 20% testovací (podle [Paretova principu](https://en.wikipedia.org/wiki/Pareto_principle)) případně 60% / 20% / 20% nebo 64% / 16% / 20% 
- obecně záleží na datech
- tento způsob rozdělení trpí tím, že někdy ověřovací a testovací množiny mají příliš málo prvků a tím pádem je výsledný model nepoužitelný (příliš malá generalizace nebo podučení)
### Cross-validation
- jak funguje?
	1. rozdělíte data na K oddílů stejné velikosti
	2. následně pro každou iteraci bude oddíl $i$ reprezentovat testovací množinu a zbylých K - 1 oddílů bude reprezentovat trénovací množinu
	4. evaluace modelu je pak zprůměrována ze všech rozdělení
- není nutno používat ověřovací množinu

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/K-fold_cross_validation_EN.svg/1920px-K-fold_cross_validation_EN.svg.png" style="background-color:white;padding: 1rem" />

### Obecné tipy
- snažte se uchovávat reprezentativní vzorek dat – rovnoměrné poměrové rozdělení tříd ve všech množinách 
	- **class imbalance problem** více: [Google Dev ML](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets) a v materiálech doc. Outraty
	- lze použít např. [Stratified k-fold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold)
- pozor na časová data – zde není vhodné data míchat, zároveň testovací data by měla obsahovat časově novější informace
	- pro řešení problému založených na sekvenčních datech se si ukážeme RNN
- snažte se aby ověřovací a trénovací množiny byly disjunktní – zamezíme tím přeučení
## Škálování a normalizace
- není dobré vkládat do NN čísla, která jsou "velká" ( víceciferná celá čísla) nebo data vícero jednotek (např. vybrané atributy pro učení budou počet ložnic a velikost domu v $m^2$) a různých rozmezí (např. jeden atribut v rozmezí 0-1 a druhý v rozmezí 100-200), protože to může dělat problém při konvergenci sítě
- **škálování** spočívá v tom, že se hodnoty atributů zavedou do předem definovaného rozsahu (například do intervalu $< 0, 1 >$)
- **normalizace** je proces, kdy distribuci hodnot atributu převedeme na [normální distribuci](https://en.wikipedia.org/wiki/Normal_distribution) (tj. střední hodnota rovna 0, směrodatná odchylka rovna 1)
- pojem [normalizace](https://en.wikipedia.org/wiki/Normalization_(statistics)) je dost obecný a často se mu přiřazují jiné významy (někdy se škálování říká i normalizace)