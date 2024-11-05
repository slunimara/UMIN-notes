   - Zdroje: Kniha – Deep Learning with Python, Sebastian Raschka – [Build a Large Language Model](https://www.manning.com/books/build-a-large-language-model-from-scratch), Nicole Koenigstein - [Transformers in Action](https://www.manning.com/books/transformers-in-action), [Google – Encoder-decoder architecture Video](https://www.youtube.com/watch?v=zbdong_h-x4), [StatQuest –Seq2seq Video](https://www.youtube.com/watch?v=L8HKweZIOmg), [Encoder-Decoder a Bottleneck](https://medium.com/@luvverma2011/demystifying-attention-mechanisms-in-sequence-to-sequence-models-transformers-part-1-98e2962408f0), [Attention vs self-attention](https://medium.com/@nishant.usapkar/self-attention-v-s-attention-understanding-the-differences-3cd1278625de), [Latent space](https://www.baeldung.com/cs/dl-latent-space), [Variational Autoencoders – IBM](https://www.ibm.com/think/topics/variational-autoencoder), [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
#### CAPTCHA breaker
- **Optical character recognition (OCR)** – technologie zaměřena na automatickou extrakci znaků z obrázků do strojově čitelného formátu
	- Více: [IBM – OCR](https://www.ibm.com/think/topics/optical-character-recognition), [Tesseract](https://en.wikipedia.org/wiki/Tesseract_(software))
- cílem projektu bylo vytvořit a natrénovat CNN na rozlomení CAPTCHA používané serverem uloz.to
- projekt vytvořený minulý rok studentem Markem Štecem
- příslušné odkazy na projekt:
	- [dataset](https://drive.google.com/drive/folders/1cOaR0F-C7Qe6C_OkGY5bF6cbE5h6xxIo?usp=sharing "https://drive.google.com/drive/folders/1cOaR0F-C7Qe6C_OkGY5bF6cbE5h6xxIo?usp=sharing")
	- [captcha generator](https://github.com/stecik/ulozto_captcha_generator "https://github.com/stecik/ulozto_captcha_generator")
	- [captcha breaker](https://github.com/stecik/ulozto_captcha_breaker "https://github.com/stecik/ulozto_captcha_breaker")
## Encoder-decoder architektura
- je to many-to-many model / sequence to sequence (seq2seq) model 
	- přímá sekvenci prvků a výstupem je sekvence prvků
- Kodér převede vstupní posloupnost na vektor fixní velikosti.
- Výsledný vektor z kodéru je následně zpracován Dekodérem na výstupní posloupnost.
- **Bottleneck** – při zpracování velké vstupní sekvence na vektor fixní velikosti může dojít ke ztrátě informace a kontextu
	- Řešení pomocí Attention mechanismů – tvorba transformerů
- příklad: překlad textu, sumarizace textu, labeling obrázků, generování textu, atd.
<img src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/b6/f4/variational-autoencoder-neural-network.component.simple-narrative-m-retina.ts=1723488893090.png/content/adobe-cms/us/en/think/topics/variational-autoencoder/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/simple_narrative/image" style="background-color:white;padding: 1rem" />
## Attention mechanismy
- [Neural Machine Translation – Paper](http://arxiv.org/pdf/1409.0473)
- mechanismy umožňující zachytit dlouhodobé závislosti a kontextové vztahy mezi prvky ve vstupní sekvenci
	- jinými slovy jsou to mechanismy umožňující modelu selektivně se zaměřit na různé části vstupní sekvence během dekódování
- cílem je získat **kontextový vektor**, který udává jaká část výstupní informace kodéru pro každý prvek z vstupní sekvence se má použít pro vytvoření aktuálního výstupního prvku dekodéru
- jak vypočítat **kontextový vektor**:
	1. získáme **scores** popisující jak relevantní jsou vstupní prvky pro daný výstupní prvek
		- pro lepší pochopení [blog post](https://muneebsa.medium.com/deep-learning-101-lesson-29-attention-scores-in-nlp-87f68f59e951)
		- [interaktivní kalkulačka](https://www.101ai.net/text/attention) – zkuste si například Movie and Users example
	1. aplikujeme softmax na scores a získáme **váhy $\alpha_{t,i}$** mající pravděpodobnostní rozdělení (v GIFu výraznost barvy při propojení **attention**)
	2. vypočítáme **kontextový vektor** $c_t$ pro krok $t$ jako vážený součet skrytých stavů encoderu $h_t$ a vah $\alpha_{t,i}$
- alternativně se dá kontext vektor uvažovat jako výsledek vyhledávače nebo doporučovacího systému
	- **query** – dotaz nad databází
	- **keys** – s čím se dotaz porovnává (charakteristiky objektu, popis obrázku, title videa, keywords u webových stránek)
	- **value** – skutečná hodnota (například obrázek / video / url na google)
	- výpočet lze jednoduše zapsat následovně`sum(values * scores(query, keys))`
	- v případě překladu máme **query** – cílový jazyk a **value, keys** – vstupní jazyk
<img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Seq2seq_with_RNN_and_attention_mechanism.gif" style="background-color:white;padding: 1rem" />
- **attention** (cross-attention) – použití mezi kodérem a dekodérem 
	- pracuje nad více sekvencemi (používá alignment – zarovnává prvky vstupní a výstupní sekvence)
	- query získány z výstupní sekvence a keys a values z vstupní sekvence
	- využívá se pro získání kontextu mezi dvěmi sekvencemi (překlad)
- **self–attention** – použití uvnitř modelu 
	- pracuje jen nad jednou vstupní sekvencí 
	- query, keys a values jsou získány z jedné sekvence
	- využívá se pro získání kontextových vztahů v rámci jedné sekvence
- **multi–head attention** – jednoduše si to můžeme představit jako několik self-attention jednotek (ty mají různé způsoby přiřazování score), kde jejich výstupy jsou spojeny do jednoho
	- např. self-attention jednotky mohou attention score počítat pomocí [vícevrstvého perceptronu](https://en.wikipedia.org/wiki/Multilayer_perceptron) (plně propojená dopředná síť – potom každá jednotka jiné váhy a tím pádem i jiný výsledek)

 **Positional Encoding**
- attention mechanismy implicitně neuvažují pozici slov v sekvenci, proto se k embedding (ke vstupu do sítě) přidává pozice slov explicitně
- bez pozice slov v sekvenci by mohlo dojít k chybným interpretacím textu

**Residual Neural networks**
- architektonický motiv, při kterém se výsledek $x$ v části sítě spojí s jeho funkční hodnotou $F(x)$ získané z menší sítě
- využitím těchto spojení je síť stabilnější a lépe konverguje

<img src="https://upload.wikimedia.org/wikipedia/commons/b/ba/ResBlock.png" style="background-color:white;padding: 1rem" />
## Transformery
- [Attention Is All You Need – Paper](https://arxiv.org/abs/1706.03762), [How transformers work – Datacamp](https://www.datacamp.com/tutorial/how-transformers-work)
- modely využívající **attention mechanismy**
	- konkrétně využívají **multi-head mechanismy**
- nyní už většinově mají **encoder-decoder architekturu**
- novější specializovanější architektury:
	- [BERT – Bidirectional Encoder Representations from Transformers](https://en.wikipedia.org/wiki/BERT_(language_model)) – transformer se zaměřením na kodér specializovaný na predikci vynechaných míst v textu, využívající se pro klasifikační úlohy textu
	- [GPT – Generative Pre-trained Transformer](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer) – transformer se zaměřením na dekodér s cílem generovat text
<img src="https://deeprevision.github.io/posts/001-transformer/transformer.png" style="background-color:white;padding: 1rem" />
## Latentní prostor
- je to abstraktní vícerozměrný prostor zachycující vlastnosti (charakteristiky) z externě získaných dat
	- vlastnost: body, které jsou blízko ve vnějším světě jsou v latentním prostoru umístění blízko u sebe
	- **latentní proměnná** je proměnná, která může být odvozena pouze na základě matematického modelu jiné **měřitelné proměnné** (ta může být měřitelná či pozorovaná)
- výhodou je to, že získáme (komprimovanou) informaci popisující původní data
- příklady použití v hlubokém učení: Word Embedding, CNN
	- generativní modely nebo autoencodery využívají latentní prostor ([Latent space reprezentation](https://en.wikipedia.org/wiki/Latent_space))

<img src="https://miro.medium.com/v2/resize:fit:1200/1*sAJdxEsDjsPMioHyzlN3_A.png" style="background-color:white;padding: 1rem" />
## Generativní modely
- při generování sekvenčních dat
	- používají se Transformery nebo RNN
	- model následně inkrementálně predikuje následující prvek sekvence na základě předchozích prvků 
		- tomuto se říká sampling
	- modely, které generují text se nazývají **jazykové modely** (language model)
		- snaží se zachytit latentní prostor jazyka (jeho statistickou strukturu)
- při generování obrazových dat
	- používají se [Diffusion model](https://en.wikipedia.org/wiki/Diffusion_model), Autoregressive, GAN nebo VAE
		- **Diffusion modely** (state-of-art) – generování na základě šumu, síť postupně snižuje kvalitu (přidáváním šumu – proces diffusion) obrázku a učí se rekonstruovat původní obrázek ([video](https://youtu.be/kzxz8CO_oG4?si=rm8XSr_iv263fnM1))
		- **Autoregressive modely** – chová se k obrázku jako k sekvenci pixelů
	- modely většinou založeny na vytváření nízkodimenzionálního latentního prostoru
		- v případě VAE je daný prostor dobře strukturovaný (každý směr prostoru kóduje charakteristiku v datech)
		- v případě GAN prostory nejsou zpravidla strukturované
- zajímavosti [film napsaný AI](https://en.wikipedia.org/wiki/Sunspring), [DeepDream](https://en.wikipedia.org/wiki/DeepDream)

<img src="https://www.researchgate.net/publication/288889700/figure/fig1/AS:613924646969357@1523382450772/Overview-of-our-network-We-combine-a-VAE-with-a-GAN-by-collapsing-the-decoder-and-the.png" style="background-color:white;padding: 1rem" />
## VAE
- zkratka za **Variational Autoencoders**
- vychází z **autoencoderů** – sítě s cílem zakódovat vstup do nízkodimenzionálního latentního prostoru a následně dekódovat zpět
	- ty se používají k široké škále činností jako například datová komprese, odšumění obrázku nebo detekce anomálií
- kódují do pravděpodobnostního rozdělení (to je spojité)
- **concept vector** – určité směry v latentním prostoru mohou kódovat zajímavé rysy v původních datech
	- pokud máme obrázky lidí, tak můžeme najít například vektor úsměvu, atd.
- jak funguje?
	- **kodér** získá z obrázku vektor středních hodnot a vektor odchylek latentních atributů (popisují rozmezí pro každý latentní atribut)
	- **dekodér** provádí náhodný výběr z těchto vektorů pro získání unikátních dat 
- při učení se používají dvě chybové funkce
	- **reconstruction loss** – rozdíl mezi originálními vstupními daty a rekonstruovaným (dekódovaným) výstupem
	- **regularization loss** – nutí síť rozumné distribuci v latentním prostoru
## GAN
- zkratka za **Generative Adversarial Networks**
- založeno na soupeření dvou modelů proti sobě (obě se postupně zlepšují ve své úloze)
	- generátor generuje obrázky
	- discriminator určuje zda jeho vstup (obrázek) je reálný (původní) či vygenerovaný
- docela nestandardní model, jelikož se nesnažíme najít minimum, nýbrž se snažíme vytvořit generátor, který maximalizuje pravděpodobnost, že se discriminator splete
	- tato vlastnost implikuje to, že se GAN hůře trénují
- jak funguje?
	- generátor vytvoří body v latentním prostoru a vygeneruje z nich obrázek
	- vytvoří se dávka s reálnými a vygenerovanými obrázky
	- dávka se předloží discriminatoru a ten hádá jejich příslušnost
	- na základě výsledků se oba modely upraví
		- v případě úspěchu discriminatoru je generátoru vnucena úprava vah
		- v případě úspěchu generátoru je naopak discriminator vnucena úprava vah
- [Vizualizace GAN](https://poloclub.github.io/ganlab/)




