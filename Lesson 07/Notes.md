- Zdroje: [What is NLP – IBM](https://www.youtube.com/watch?v=fLvJ8VdHLA0) [NLP vs NLU vs NLG](https://www.youtube.com/watch?v=1I6bQ12VxV0), [Word2Vec – nevýhody](https://medium.com/@pooja93palod/understanding-word2vec-a-beginners-guide-to-word-embeddings-6ecb893dbf61), [Word2Vec – super popis a poznámky na konci](https://medium.com/@hari4om/word-embedding-d816f643140)

## Word embedding
- slova převádíme na vektory čísel kódující bod v prostoru nazývaný **embedding space**
- snažíme se, aby tyto vektory odráželi sémantický vztah mezi slovy
	- tj. synonyma a podobná slova budou v embedding space blízko u sebe (blízkost definujeme geometrickou vzdáleností bodů)
- pro kvalitní embedding je nutné jej vytvořit nad obrovským množstvím dat
- v případě že máme málo trénovacích dat, tak můžeme použít předtrénovaný
	- [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html)
	- [GloVe](https://nlp.stanford.edu/projects/glove/)
	- [Gensim](https://github.com/piskvorky/gensim-data)
- [minimalistická vizualizace embedding prostoru](https://www.101ai.net/text/word-embed), [reálná vizualizace embedding prostoru](http://projector.tensorflow.org)
**Příklad:**
<img src="https://miro.medium.com/v2/resize:fit:1200/1*sAJdxEsDjsPMioHyzlN3_A.png" style="background-color:white;padding: 1rem" />

### Word2Vec
- nejznámější a nejúspěšnější schémata word embedding, kterou vytvořil [Tomáš Mikolov](https://cs.wikipedia.org/wiki/Tomáš_Mikolov)
- používá mělkou 2-vrstvou neuronovou síť k vytvoření embeddingu
	- síť se trénuje na základě obrovského korpusu vět
	- velikost vstupního vektoru je učena množstvím slov, které se má model "naučit" (klidně v řádech statisíců či miliónů)
	- pro natrénování sítě je využita jedna ze dvou metod (CBOW nebo Skip-gramy)
	- po natrénování je vektor každého slova určen jeho váhami propojení s první vrstvou sítě
	- konkrétně první vrstva obsahuje jednotky s lineární aktivací a druhá obsahuje softmax pro predikci pravděpodobnosti jednotlivých slov
	- metoda má i efektivní způsob jak se při výpočtu zaměřit pouze na konkrétní část sítě a tím zrychlit výpočet 
- k učení sítě využíváme jednu ze dvou metod:
	- **Continuous Bag of Words** (CBOW) – sít se snaží predikovat slovo na základě jeho okolních slov
		- vstupem je vektor, obsahující $1$ u okolních slov vynechaného (u ostatních $0$)
		- výstupem je predikce možných slov
		- cílem je mít největší pravděpodobnost u hledaného ve výstupu
	- **Skip-gram** – síť se snaží predikovat okolní slova na základě konkrétního slova
		- vstupem je vektor, obsahující $1$ u aktuálního slova (u ostatních $0$)
		- výstupem je predikce možných slov v okolí
		- cílem je mít největší pravděpodobnost u okolních slov
- nevýhody:
	- každému slovu je přiřazen pouze jeden vektor
	- sítě nemají jak rozeznat mnohoznačná slova vzhledem ke kontextu (např. koruna, jazyk, zámek) a tím pádem jim přiřazuje stejný vektor
	- vytvoření word embeddingu je výpočetně náročné (pokud jej chceme rozumně velký)
- doporučuji video [Word2Vec od StatQuest](https://www.youtube.com/watch?v=viZrOnJclY0)

<img src="https://miro.medium.com/v2/resize:fit:836/0*I5_hlMux7PY0nQzo.png" style="background-color:white;padding: 1rem" />

## Zpracování přirozeného jazyka
- angl. Natural Language Processing (NLP)
- disciplína zabývající se zpracováním a tvorbou dat [přirozeného jazyka](https://cs.wikipedia.org/wiki/Jazyk_(lingvistika)) a také jeho porozuměním
- pod-disciplíny:
	- [rozpoznání mluvené řeči](https://en.wikipedia.org/wiki/Speech_recognition) 
	- [porozumění přirozenému jazyku](https://en.wikipedia.org/wiki/Natural-language_understanding) (NLU) – snaží se o co nejlepší porozumění psaných či mluvených dat 
		- "Siri? Kolik je stupňů?" –> Aktuální teplota je 5 °C.
	- [tvorba přirozeného jazyka](https://en.wikipedia.org/wiki/Natural_language_generation "Natural language generation") (NLG) – snaží se o co nejlepší "odpověď" v lidském jazyce
		- výsledek chceme správně jak sémanticky (aby vyjádřil to co chce vyjádřit) tak syntakticky (správný slovosled, atd.)
		- dalo by se brát jako inverzní k **NLU**
- vstupem NLP jsou většinou nějaká nestrukturovaná data, která si model převede na strukturovaná
	- krásný příklad [What is NLP – IBM](https://www.youtube.com/watch?v=fLvJ8VdHLA0)
	- při převodu na strukturovaná data se používá spousta metod a postupů – hojně využívaná je lingvistika což je jazykověda zabývající se přirozeným jazykem
		- tokenizace
		- získání základního tvaru slova (jinak nazýváno lemma), infinitiv, rozpoznání slovních druhů, atd.
	- **Named Entity Recognition** (NER) – rozpoznání známých pojmenovaných entit v textu (Česká republika, Kofola, Česká Koruna)
- velmi pěkný článek z populárně naučného časopisu Vesmír – [Lingvistika na Matematicko-fyzikální fakultě?](https://vesmir.cz/cz/casopis/archiv-casopisu/2012/cislo-9/lingvistika-matematicko-fyzikalni-fakulte.html)
- kde tedy lze **NLP** použít?
	- sumarizace, klasifikace nebo gramatická oprava textu
	- chatbot
	- analýza sentimentu
	- vylepšení vyhledávačů (například [Google Search Engine](https://blog.google/products/search/search-language-understanding-bert/))
	- generace textu, atd.

- **Prompt Tuning** – technika, při které se vstupem vložíme nějakou pomocnou informaci (prompt) pro zlepšení výsledku používáno u předtrénovaných modelů pro zlepšení multifunkčnosti
	- soft prompts – vytvořené NN sítí
	- hard prompts – vytvořené prompt inženýrem