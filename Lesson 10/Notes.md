- Zdroje: [Evoluce](https://cs.wikipedia.org/wiki/Evoluce), [Genetic Algorithm – Knapsack](https://arpitbhayani.me/blogs/genetic-knapsack/), [EvoJAX – Neuroevolution](https://cloud.google.com/blog/topics/developers-practitioners/evojax-bringing-power-neuroevolution-solve-your-problems), [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)

## Evoluční algoritmy
- inspirace evolučním procesem
	- základem jsou dědičné změny v průběhu generací
	- konkrétně se jedná o mechanismy jako:
		- mutace – náhodná změna genů
		- rekombinace (křížení) – kombinace genetické informace během páření, cílem je větší variabilita potomstva
		- přirozený výběr (selekce) – tvrzení založeno na tom, že genetická informace jedinců s výhodnějšími vlastnostmi (silnější, vnímavější, schopnější, atd.) jsou upřednostňovány pro předání potomkům
- idea je tedy využití evolučních mechanismů na populaci potenciálních výsledků za účelem řešení (optimalizačního) problému

### Jak takový proces zhruba probíhá
- na počátku se vygeneruje množina kandidátních jedinců (řešení) nazvaných **populace**
	- to probíhá zpravidla náhodně
- v každé generaci se utkají všichni jedinci a pouze nejlepším je umožněno křížení
- potomci v následující generaci jsou buď kopiemi rodičů, nebo projdou křížením, kdy získají informaci od každého rodiče, a následně mohou projít mutací
	- pozn. proces selekce a aplikace různých evolučních vlastností se může lišit 

<img src="https://user-images.githubusercontent.com/4745789/156874170-608cd9a4-6241-4882-b123-658d14a64c89.png" style="display:block; margin:0 auto; padding: 1rem" />

- **fitness funkce** slouží pro ohodnocování výkonnosti (vhodnosti) kandidátních řešení
	- u některých problému je ona optimalizovanou funkcí
- musíme určit způsob reprezentace jednotlivých instancí tak, abychom mohli snadno aplikovat evoluční mechanismy
	- velmi specifické řešenému problému
	- třeba můžeme instance reprezentovat jako sekvence bitů, což se hodí například u problému batohu (1 / 0 vyberu / nevyberu instanci s indexem i)
- ukončovací podmínka – například skončení po určitém počtu iterací nebo po nalezení uspokojivého řešení

## Neuroevoluce 
- kombinace dvou principů: neuronových sítí a evolučních algoritmů
	- místo vyvíjejících se potenciálních řešení se různě vyvíjí / mění NN – tedy je možno upravovat strukturu, váhy, atd.
	- snaží se tedy najít rovnováhu mezi fitness a různými NN
- většinou unsupervised learning
	- výpočetně náročné
	- lze aplikovat na mnohem obecnější problémy
	- stačí jim jen fitness (jak kvalitně zvládlo vyřešit úlohu)
- Dostupné knihovny: 
	- [EvoJAX](https://github.com/google/evojax?tab=readme-ov-file)
	- [NEAT](https://neat-python.readthedocs.io/en/latest/neat_overview.html) (NeuroEvolution of Augmenting Topologies)
		- knihovna implementující oblíbený algoritmus NEAT (6 stránkový [původní článek](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf))
- [Video pro inspiraci](https://youtu.be/dkvFcYBznPI?si=5IxJn75UTPQlRsLT)