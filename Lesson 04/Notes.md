- Zdroje: Kniha – Deep Learning with Python, [IMB – RNN](https://www.ibm.com/topics/recurrent-neural-networks), [Sekvenční data](https://aiml.com/what-does-sequential-data-mean-which-models-are-best-suited-for-handling-sequential-data/), [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/),  [Wikipedie – LSTM: architektura, použití](https://en.wikipedia.org/wiki/Long_short-term_memory) [Scaler – architektura LSTM](https://www.scaler.com/topics/deep-learning/lstm/), [Medium – LSTM vs GRU](https://medium.com/@prudhviraju.srivatsavaya/lstm-vs-gru-c1209b8ecb5a)

- sekvenční data jsou data kde je podstatné pořadí prvků
	- vlastně předpokládáme mezi daty nějakou formu závislosti (například: pozice v textu, data z předchozích dnů, atd.)
	- kde se s nimi můžeme setkat: rozpoznávání mluvené řeči, generaci hudby, analýza sendimentu, analýza videa
- **sequence modeling** – síť přímá či generuje sekvence různé délky
	- příklad: analýza sentimentu, popis obrázku, překlad jazyka
	- typy sítí: many-to-one, one-to-many, many-to-many

<img src="https://deeplearningmath.org/images/type_sequence_Modeling.png" style="background-color:white;padding: 1rem" />
## Práce s textem
- pokud chceme pracovat s textem, tak jej do sítě nikdy nedáváme jako řetězce znaků
- místo toho přetvoříme text na vektory, to lze dvěma způsoby
	- rozdělíme text na slova (znaky) a každé slovo (znak) transformujeme na vektor
	- extrahujte n-gramy slov a transformujte každý n-gram do vektoru (n-gramy jsou překrývající se skupiny více po sobě jdoucích slov nebo znaků / sekvence slov nebo znaků textu)
- slova, znaky nebo n-gramy rozloženého textu nazýváme tokeny (procesu rozložení textu, pak tokenizace)
- tokeny následně vektorizujeme pomocí různých metod:
	- **one-hot encoding** 
		- každému slovu je přiřazen index $i$ následně, je slovu přiřazen vektor kde $i.$ složka je nastavena na 1 ("hot") a zbytek je nastaven na 0 
		- více způsobů (viz. jupyter)
		- nevýhody: 
			- **curse of dimensionality** (moc dat = moc velká dimenze)
			- postrádá sémantické informace
			- nezachycuje vztahy mezi slovy
		- už jsme viděli v 2. cvičení s převodem kategoriálního atributu na binární vektor
	- **token embedding (word embedding)**
		- k vytvoření vektorů se používají ML techniky (často založené na NN)
		- nejvíce využíváno při **zpracování přirozeného jazyka** (NLP), protože kódují sémantický význam a vztahy mezi slovy 
		- geometrický vztah mezi vektory slov odráží sémantický vztah mezi slovy
		- populární metoda je **Word2Vec** (používá neuronovou síť k předpovídání okolních slov cílového slova v daném kontextu)
		- v Keras [Embedding layer](https://keras.io/api/layers/core_layers/embedding/)
		- více se o tom budeme bavit za pár týdnů
- Zdroje: [Různé knihovny pro tokenizaci a problémy při zpracování textu](https://neptune.ai/blog/tokenization-in-nlp), [IBM - word embedding](https://www.ibm.com/topics/word-embeddings)
## Rekurentní neuronové sítě
- dopředné sítě zpracovávají každý vstup nezávisle na ostatních
	- pokud chceme síti předložit sekvenční data (například: recenzi filmu)
- naopak rekurentní obsahují nějakou formu vnitřního stavu („pamětí“)
	- informace z předchozích iterací dané sekvence ovlivňují následující iterace stejné sekvence
	- stav se pro každou nezávislou frekvenci počítá zvlášť

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" style="background-color:white; padding: 1rem;" />

- jedno z použití RNN je naučit model sekvenční predikce na základě sekvenčních vstupů
	- například předpovídání denních hladin vody na základě dní 
- další použití může být například: překlad, [NLP](https://www.ibm.com/topics/natural-language-processing), [analýza sentimentu](https://www.ibm.com/topics/sentiment-analysis), atd.
	- pro zajímavost: [Google Překladač](https://en.wikipedia.org/wiki/Google_Neural_Machine_Translation)
- při učení (určení gradientu) používají "forward propagation and backpropagation through time" algoritmy, které jsou upravené pro sekvenční data
	- váhy jsou sdíleny napříč vrstvou s rekurentními jednotkami
- jeden z problémů, které základní RNN mají je ten, že nezvládají dlouhodobé závislosti z důvodu mizejícího gradientu (vanishing gradient problem)
	- problém mizejícího gradientu – situace kdy gradienty, které se používají k aktualizaci sítě při učení, se při backpropagation stanou extrémně malé
	- obvykle v sítích s mnoha vrstvami
	- opak problém explodujícího gradientu (exploding gradient problem) 

<img src="https://media.licdn.com/dms/image/C5612AQH5Im8XrvLmYQ/article-cover_image-shrink_600_2000/0/1564974698831?e=2147483647&v=beta&t=mVx-N8AfjAS5L-ktV6vmi_5LxR1madQ16yT1fRu__Jk" style="background-color:white; padding: 1rem" \>
### Long Short Term Memory sítě

- ukládá informace na později, čímž zabraňuje postupnému mizení starších signálů během zpracování
- výhody: zabraňuje problému mizejícího gradientu, dokáže zpracovávat dlouhodobé závislosti
- nevýhody: výpočetní náročnost, náchylnost k přeučení
- má mnoho aplikací: klasifikace a predikce sekvenčních dat – ručně psané číslice, rozpoznávání řeči, překlad jazyka, hraní videoher, ...
#### Architektura LSTM
- jednoduchý popis architektury:
	- **cell state** $c$  – dlouhodobá paměť
	- **hidden state** $h$ – krátkodobá paměť
	- **forget gate** $F$ – rozhoduje o dlouhodobé paměti
		- konkrétně: jaká informace má být zapomenuta z $c$
		- vstupem do sigmoid je $h_t$ a $x_t$ ta vrátí hodnotu z intervalu $<0, 1>$, kde $0$ znamená zapomenout a $1$ znamená ponechat informaci 
	- **input gate** $I$ – rozhoduje o dlouhodobé paměti
		- konkrétně: jaká informace se uloží do do $c$
		- z tahn získám potřebné informace a sigmoid (funguje jako u **forget gate**) určí kolik z té informace si zapamatovat 
	- **output gate** $O$ – rozhoduje o krátkodobé paměti a o výstupu
<img src="https://miro.medium.com/v2/resize:fit:1080/1*_2yXu6QhUihXUWEQn0bDkw.jpeg" style="background-color:white; padding: 1rem; width:350px" \>
### Gated Recurrent Unit
- má jednodušší architekturu než LSTM
	- má pouze 2 hradla: **update gate** a **reset gate**
	- existuje více typů architektur více [zde](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
- má méně parametrů a tím pádem je i výpočetně efektivnější a méně náchylný k naučení (oproti tomu LSTM se dokáže naučit komplexnější vzory)
- je vhodný pro menší datasety