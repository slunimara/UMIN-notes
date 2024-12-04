- Zdroje: [IBM Reinforcement learning](https://www.ibm.com/topics/reinforcement-learning), [OpenAI Spinning UP](https://spinningup.openai.com), [IBM – RLHF](https://www.ibm.com/topics/rlhf), [Amazon – RLHF](https://aws.amazon.com/what-is/reinforcement-learning-from-human-feedback/)

## Reinforcement Learning 
- agent se učí rozhodnutím pomocí opakované interakce s prostředím
	- podobné jako když se učí člověk technikou **pokus a omyl**
	- o problému se naučí tím, že interaguje s prostředím prostřednictvím akcí
	- zároveň je nějak motivován a odměňován za úspěch
	- cílem mnoha aplikací je napodobit reálné biologické učení s pomocí pozitivní zpětné vazby
- prvky ze kterých se takový systém (Markovův rozhodovací proces) učení skládá:
	- **Agent** – model, který chceme naučit požadovanému chování bez explicitního vedení (například. bez labelů atd.)
	- **Environment** – dynamické prostředí se kterým agent interaguje 
		- může se měnit akcemi agenta, ale i samovolně
		- jeho aktuální kompletní stav je reprezentován pomocí **State**
	- **Observation** – všechny informace o prostředí, které agent zná a na základě kterých se rozhoduje
		- je to částečná informace o stavu (**State**) prostředí (**Environment**)
		- obvykle se mění po každém rozhodnutí, které agent udělá
		- agent ho získává z prostředí
		- v kontextu šachu například rozmístění všech figurek
	- **Action** – akce, které může agent provést
		-  například: všechny možné posuny jeho figurkami v aktuálním tahu
	- **Reward function** – měří úspěch agenta a zároveň jej odměňuje případně trestá za chyby 
		- za každou akci je agent je buď odměněn nebo potrestán
		- jedná se o okamžitý užitek
		- například: odměna za zabrání královny může být vyšší než pěšce
	- **Policy** – strategie na základě se agent rozhoduje
		- může to být soubor pravidel nebo funkce
		- tohle je cílem RL
		- například: policy hrání šachu může mapovat nepřátelský šach na akci, která tomu zabrání
	- **Value function** – jedná se o hodnotu měřící dlouhodobý užitek agenta
		- například: může měřit o kolik silnější figurky má
<img src="https://gymnasium.farama.org/_images/AE_loop_dark.png" style="width:30vw; display:block; margin:0 auto; padding: 1rem" />
- použití:
	- Robotika – ideální pro rozhodování v nepředvídatelném prostředí
		- Odkazy: [Nvidia RL](https://www.nvidia.com/en-us/use-cases/reinforcement-learning/?deeplink=content-tab--1), [Google RL](https://deepmind.google/discover/blog/advances-in-robot-dexterity/)
	- Zpracování přirozeného jazyka – využívá se pro zlepšení NLP modelů ať už v oblasti komunikace chatbotů se zákazníkem nebo pro jejich přirozenější lidštější odpovědi  
	- Herní boti
		- [AlphaGo](https://deepmind.google/research/breakthroughs/alphago/) porazil lidského mistra světa ve hře Go
		- [OpenAI Five](https://openai.com/index/openai-five/) bot hrající komplexní MOBA hru Dota2
		- Nebo jiné [videohry](https://deepmind.google/discover/blog/deep-reinforcement-learning/)

### Typy přístupy učení agenta
- neboli přístupy získání **Policy**
- základním dělením je informace o tom, zda má agent přístup (nebo možnost se naučit) model celého prostředí
	- to lze například v šachu, kdy agent má přesně definované možné akce a ví jaký z nich bude mít užitek
	- naopak u problému přistání helikoptéry jednoduše nelze namodelovat prostředí
	- modelem prostředí se myslí funkci, která určuje přechody mezi stavy a odměnami
1. **Model-Free RL:**
    - Agent nezná dynamiku prostředí (pravděpodobnosti přechodů mezi stavy) a učí se přímo na základě zkušeností (tedy na základě odměn)
    - další dělení:
		a) **Value-Based Methods** – Tyto metody se soustředí na odhadování hodnot stavů nebo hodnot stav-akce. Politika se následně odvodí na základě těchto hodnot.
		b) **Policy-Based Methods** – Tyto metody se zaměřují na přímo optimalizaci politiky, místo odhadu hodnot.
2. **Model-Based RL:**
    - Agent si vytvoří model prostředí (např. odhaduje pravděpodobnosti přechodů) a používá ho k plánování.

<img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" style="background-color:white;padding: 1rem" />

### RLHF (Reinforcement learning from human feedback)
- technika používaná ke zlepšení jazykových modelů na základě zpětné vazby uživatele 
-  jak to zhruba funguje:
	1. musíme získat natrénovaný model
	2. supervised fine-tuning – naučíme model generovat odpovědi na základě uživatelských preferencí
		- expert definuje `(prompt, response)` aby ukázal, jak reagovat na různé úkoly
		- někdy modely nemusí správně pochopit otázky jako: "Nauč mě jak si vytvořit životopis?" –> "Použij Microsoft Word." nebo "Kdo je Petr Pavel?" –> "Známá osobnost z České Republiky.", a nebo "O čem je kniha Pán Prstenů?" –> "Je to o hobitovi, který zažije dobrodružství s prstenem."
		- nevýhodou je časová náročnost
	3. reward model training – vytvoříme model odměň, který kvantifikuje lidské preference na číselný signál odměny
		- tím modelu umožníme offline učení, tím je myšleno, že nebude nutné nějakého 
		- tento model odměn se například vytvoří tím, že se uživatelům předloží vždy 2 odpovědi modelu a uživatel odpoví, která se mu líbí víc
	4. policy optimization – jak moc má model odměn ovlivňovat odpovědi modelu
- další z nevýhod může být subjektivita odpovědí (jak definovat co je správně?) nebo poskytnutí škodlivých / nepravdivých odpovědí při tvorbě modelu odměn
- [blog od OpenAI](https://openai.com/index/learning-from-human-preferences/)

<img src="https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/08/31/ML-14874_image001.jpg" style="background-color:white;padding: 1rem" />