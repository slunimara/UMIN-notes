- Zdroje: Kniha – Deep Learning with Python
## Konvoluční neuronové sítě
- angl. convolutional neural networks nebo zkráceně convnets
- rozpoznávání obrázků pomocí základních NN (plně propojených dopředných sítí) neřešitelné, z důvodu: 
		- nevyužití sousednosti pixelů (klasické NN zanedbávají vztah mezi sousedními pixely)
		- nutnosti použít velké množství vah
			- pokud bychom propojily každý pixel s každým
			- u obrázku $500 \times 500$ to dělá $250000$ jednotek jen ve vstupní vrstvě 
- husté vrstvy se učí globální vzory z dat, zatímco konvoluční vrstvy se učí lokální vzory
	- v případě obrázků se hledají vzory v 2D mřížce o specifikované velikosti

- CNN mají dvě velké výhody:
	- vzory, které se CNN naučí následně dokáží aplikovat i na jiných místech – tolerují posun pixelů v obrázku a berou v úvahu vztah okolních pixelů (bílé pixely jsou u bílých pixelů a podobně)
	- využívají hierarchii pro rozpoznávání vzorů (v prvních vrstvách obrázku naleznou triviální vzory – čáry, kolečka atd. – v dalších je sestavují vzory dohromady)
- konvoluční sítě pracují s 3D tensory – **feature maps** – s dvěmi **prostorovými osami** (u obrázku: výška, šířka) a jednou **hloubkovou osou** (u obrázku: RGB / černobílá)
- výstupem je opět feature map ve tvaru (width, height, filters)
	- prostorové osy nyní nazýváme **response maps**
	- filtry obsahují konkrétní vzory ze vstupních dat (například: čárka, kolečko, atd.)
	- response map filtru nad vstupem popisuje odezvu vzoru filtru nad různými oblastmi obrázku
- [Vizualizace CNN](https://poloclub.github.io/cnn-explainer/), [Vizualizace CNN u ručně psaných číslic](https://adamharley.com/nn_vis/cnn/3d.html)
### Konvoluční vrstvy
Konvoluční vrstvy mají dva základní parametry: 
	- velikost posuvného okna
	- výsledný počet filtrů
1. **posuvné okno** je oblast o pevně specifikované velikosti, běžně 3x3 (5x5) pixelů, která postupně projíždí obrázek a získává informace o oblastech (matice tvaru (window_height, window_width, input_depth))
2. získaná informace z lokální oblasti se transformuje z 3D do 1D – matice z posuvného okna se vynásobí s konvolučním jádrem – maticí vah (získána učením) – tomuto se říká konvoluce
3. výsledný 1D vektor se vloží do výstupu (output feature map) a pokračuje se s další oblastí, posuvné okno neprojede celý obrázek
<img src="https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg" style="background-color:white;padding: 1rem" />

- Co ovlivňuje velikost dimenze výsledných dat z konvoluční vrstvy?
- **Padding** – rozšíření původních dimenzí vstupu tak, abychom mohli získat informace o oblasti se středem v každé buňce vstupu
	- cílem je získat prostorové dimenze output feature map stejné jako u vstupu
- **Stride** – udává jak rychle se posuvné okno pohybuje
	- výchozí rychlost je rovna jedné
	- například při nastavení stride na dva je velikost dimenze výstupu snížena dvojnásobně oproti výchozí velikost dimenze výstupu
	- pro snížení velikosti dimenze se, ale používá **pooling**

**Pooling** – funguje podobně jako konvoluční vrstva, s tím rozdílem, že neaplikujeme konvoluční jádro nýbrž jednodušší operaci třeba **max** – **Max-pooling**
	- existuje i **Min-pooling**, **Average-pooling** avšak nejsou tak používané
	- obvykle je použita velikost okna $2 \times 2$ a stride $2$, abychom snížily velikost o faktor $2$
	- cílem je snížit velkost feature map – proč? 
		- **výpočetní náročnost** – abychom mohli na konci aplikovat klasifikaci, tak je nutné mít rozumnou velikost vektorů 
		- k získání **prostorové hierarchie vzorů** – pokud by mezi vrstvami byl příliš malý rozdíl v dimenzích, tak by těžko abstrahoval (hierarchie čárka – ústa – obličej)