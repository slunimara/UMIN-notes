- Zdroje: [Papers WIth Code](https://paperswithcode.com/sota), [IBM future of AI](https://www.ibm.com/think/insights/artificial-intelligence-future), [Quantization](https://huggingface.co/docs/optimum/concept_guides/quantization), [Fungování XLNet a BERT](https://rbcborealis.com/research-blogs/understanding-xlnet/), [T5](https://cameronrwolfe.substack.com/p/t5-text-to-text-transformers-part), [Autoregression](https://aws.amazon.com/what-is/autoregressive-models/), [BART](https://www.digitalocean.com/community/tutorials/bart-model-for-text-summarization-part1), [Omezení AI](https://www.computer.org/publications/tech-news/community-voices/regulations-on-generative-ai), [Etika AI Kaggle](https://www.kaggle.com/learn/intro-to-ai-ethics)

- velmi doporučuji kanál [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers)
## State-of-Art AI
- tvorba multimodal AI modelů – modely, které dokáží pracovat s více typy dat (text, video, audio, 3D, atd.)
	- pro robotiku: [PaLM-E](https://palm-e.github.io) 
- zlepšení generativních modelu pro multimedia
	- nejzajímavější teď video – [Sora](https://openai.com/index/sora/), [Veo](https://deepmind.google), [Adobe Firefly](https://www.adobe.com/products/firefly.html)
- grafické karty
	- pro spotřebitele: [Nvidia RTX](https://www.nvidia.com/en-us/ai-on-rtx/) nebo [AMD Radeon](https://www.amd.com/en/products/graphics/radeon-ai.html)
		- využití AI v řadě různých aplikací Adobe, Blender, Game Engines, videohry ([DLSS](https://www.nvidia.com/cs-cz/geforce/technologies/dlss/) – zvýšení FPS pomocí generování snímků a rozlišení obrazu – upscaling)
		- nebo použití přímo pro tvorbu modelů 
		- existují i speciální aplikace vytvořené přímo pro dané karty ([Amuse](https://www.amd.com/en/products/graphics/radeon-ai.html) – stable diffusion model, [NVIDIA Canvas](https://www.nvidia.com/en-us/studio/canvas/) – tvorba krajinek na základě jednoduchých tvarů )
	- obecně trhu aktuálně naprosto dominuje Nvidia s jejich [A100](https://www.nvidia.com/en-us/data-center/a100/) a nově i [H100](https://www.nvidia.com/en-us/data-center/h100/) grafickými kartami
## Budoucnost v AI?
- consistency models – konkurence diffusion modelů vzhledem k rychlosti generování – [OpenAI blog](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)
- možné využití kvantové výpočetní techniky [Google Quantum](https://quantumai.google)
- **kvantizace** – technika, při které reprezentujeme váhy a aktivace pomocí "menších hodnot" při provádění inference
	- přesněji převádíme na datové typy s nižší přesností: př. `float32` $\to$ `int8`
	- výsledný model vyžaduje méně paměti a je méně výpočetně náročný
	- 1-bit LLM – např. [BitNet b1.58](https://huggingface.co/papers/2402.17764): všechny parametry jsou z množiny $\{-1, 0, 1\}$
	- knihovna [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) – zatím není pro apple silicon
- nebo kompletně jiné způsoby zmenšení modelu – [LoRA](https://hf.co/papers/2106.09685)
- zamezení **halucinace** modelu – situace, kdy si model předloží nesprávné nebo zavádějící informace
- No-code / low-code tvorba aplikací a modelů AI
	- služby jako [Hugging Face AutoTrain](https://huggingface.co/autotrain)
- ekologické otázky z důvodu potřeby obrovského množství energie
	- poslední generace grafických karet zaznamenaly sice obrovský skok ve výkonu, ale za cenu zvýšení spotřeby
	- tvorba nových jaderných elektráren – [Google, Microsoft a Amazon](https://www-theguardian-com.translate.goog/technology/2024/oct/15/google-buy-nuclear-power-ai-datacentres-kairos-power?_x_tr_sl=en&_x_tr_tl=cs&_x_tr_hl=cs&_x_tr_pto=sc) 
	- více: [AI Power Consumption](https://www.forbes.com/sites/bethkindig/2024/06/20/ai-power-consumption-rapidly-becoming-mission-critical/),[How Much Energy Will It Take To Power AI?](https://www.contrary.com/foundations-and-frontiers/ai-inference)
- Personalizované AI modely 
	- mohou běžet na každodenních zařízeních
	- zaměřené na soukromí
	- dokáží lépe pochopit dotazy uživatele z důvodu jejich kontextu 

## State-of-Art Transformer Modely
- **BERT (Bidirectional Encoder Representations from Transformers):**
	- **Architektura:** Encoder-only
	- **Trénování:** některá slova vstupní sekvence zakrývá, pomocí `[mask]` tokenu, a snaží se je predikovat 
		- predikce masek je nezávislá
	- **Cíl:** Masked Language Modeling (MLM)
	- **Silná stránka:** Vynikající pro úlohy jako klasifikace, named entity recognition (NER) a odpovědi na otázky
	- **Omezení:** Není určen pro generování textu, jelikož není trénován pro generování souvislých sekvencí

- **GPT (Generative Pre-trained Transformer):**
	- **Architektura:** Decoder-only
	- **Trénování:** Autoregresivně  – predikce následující hodnoty na základě předchozích hodnot
	- **Cíl:** Kauzální souvislé modelování jazyka – Causal Language Modeling (CLM)
	- **Silná stránka:** Vynikající pro úlohy generování textu.
	- **Omezení:** Problémy s úlohami vyžadujícími obousměrný kontext.

- **T5 (Text-to-Text Transfer Transformer):**
	- **Architektura:** Encoder-decoder architektura
	- **Trénování:** Vkládání promptů, definující specifické úkoly, do vstupní sekvence pro určené úlohy
		- "přelož z Angličtiny do Češtiny: Hello how are you?" –> "Ahoj, jak se máš?"
		- "sumarizuj: Gandalf je fiktivní postava čaroděje, který hraje klíčovou roli v románech J. R. R. Tolkiena Hobit a Pán prstenů. Vedl Společenstvo prstenu a velel armádám Západu ve Válce o Prsten. Ve filmové trilogii Petera Jacksona Pán prstenů jej hrál sir Ian McKellen." –> "Čaroděj Gandalf je fiktivní hrdina z knižních románů a filmů."
	- **Cíl:** Úlohy převodu sekvence na sekvenci
	- **Silná stránka:** Univerzální; zvládá sumarizaci, překlad a klasifikaci.
	- **Omezení:** Výpočetně náročný kvůli své univerzálnosti

- **XLNet:**
	- **Architektura:**  Transformer využívající permutačního přístupu (využívá poznatků z BERT), navíc obsahuje dva typy attention mechanismů
	- **Trénování:** Trénování probíhá podobně jak u BERT až na rozdíl v tom, že predikce masek je na sobě závislá. Masky predikuje v jeho vybraném pořadí. 
	- **Cíl:** Pemutované Masked Language Modeling (MLM)
	- **Silná stránka:** Lepší modelování dlouhých závislostí a využití kontextu ve srovnání s BERT
	- **Omezení:** Složitější trénink; vyžaduje větší výpočetní zdroje.

- **BART:**
	- **Architektura:** Encoder-decoder architektura (Bidirectional Kodér a Autoregresivní Dekodér)
	- **Trénování:** Kodér poškodí text (třeba maskou) a Dekodér se jej snaží zrekonstruovat
	- **Cíl:** Autoenkodér se snaží odstranit šum (rekonstrukce poškozených vstupů).
	- **Silná stránka:** Vynikající pro generativní úlohy (např. sumarizaci).
	- **Omezení:** Může být příliš náročný pro jednodušší klasifikační úlohy. Dá se uvažovat jako spojení (zoběcnění) nad modely jako BERT a GPT.
## Úlohy modelů
- **Masked Language Modeling (MLM):** Používá se v BERT a jeho variantách. Podskupina vstupních tokenů je maskována a model je predikuje.
	- **Pros:** Využívá obousměrný kontext.
	- **Nevýhody:** Neefektivní pro generativní úlohy.

- **Kausální modelování jazyka (CLM):** Používá se v modelech GPT. Předpovídá další token postupně.
	- **Pros:** Ideální pro generování textu.
	- **Nevýhody:** Nepoužívá obousměrný kontext. Pouze kontext z leva.

- **Seq2Seq trénování:** Používá se v T5, BART a mT5. Všechny úlohy považuje za vstupně-výstupní problémy (např. „shrň: text“).
	- **Pros:** Jednotný přístup pro všechny úlohy.
	- **Nevýhody:** Výpočetně náročné.

**Použití modelů pro různé typy úloh:**

| **Úkol**             | **Nejlepší možný (nejlepší) model** |
| -------------------- | ----------------------------------- |
| Analýza sentimentu   | BERT                                |
| Klasifikace textu    | RoBERTa                             |
| Generování textu     | GPT-3 / GPT-4                       |
| Strojový překlad     | mT5 / MarianMT                      |
| Sumarizace           | BART / T5                           |
| Odpovídání na otázky | BERT / RoBERTa                      |
| Generování kódů      | Codex (varianta GPT-3)              |
## Etikcké a právní otázky umělé inteligence
- velmi široké a pokrývají různé oblasti od technických až po společenské
- **Human-centered design (HCD)** – přístup tvorby systémů, kde na první místě je uživatel (obecně člověk)
	- při tvorbě je základem porozumět potřebám lidí a podle toho definovat problém
	- je kladen důraz na empatii a spolupráci s uživateli
	- tvorba iterativním procesem (testování, zpětná vazba a zdokonalování)
	- možné prvky:
		- přizpůsobení systému – možnost vypnout určité funkce
		- komunikace s tvůrci systému či firmou
		- informace o zpracování dat
	- otázky jako:
		- Může použití systému způsobit škodu? 
		- osobní soukromí, zkreslení výsledků, atd.
		- Je benefit vytvoření systému převyšující škodám, které může napáchat? 
	- použití například: při tvorbě uživatelského rozhraní, plánovaní města, výroba spotřebního zboží 
- vznik regulací zaměřených na odpovědnou AI ([AI Act EU](https://artificialintelligenceact.eu), [USA AI](https://ai.gov))

**Kdo je zodpovědný za rozhodnutí AI?**
– výrobce, uživatel, vývojář, nebo provozovatel?

**Je nahrazování lidské práce spravedlivé?**
– mělo by se lidem, které nahradí AI, nějak pomáhat?
– jak zajistit tak rychlou adaptaci lidí na nové technologie?

**Co když je AI používáno pro generování falešných obsahů a polarizaci společnosti?**
– kdo všechno by měl být zodpovědný?
– mělo by být AI omezováno? do jaké míry?

**Využití umělé inteligence ve výuce?**
– do jaké míry je správné nechávat svou práci na AI?
– je správné nechat děti využívat umělou inteligenci?

### Použití umělé inteligence v umění
- velmi komplexní téma – obsahuje témata jako originalita, autorské práva, vliv na uměleckou komunitu

**Kdo vlastní dílo vytvořené umělou inteligencí?**
– opět podobné jako u zodpovědnosti
– mohou to být například umělci, kteří vytvořili data pro trénování?

**Lze nějak opodstatnit použití AI na tvorbu uměleckých děl?**
– myslíte, že je správné využívat AI k tvorbě filmů nebo knih?  

**Mělo by být vždy jasné, že dané dílo vytvořila AI?**
– může mást diváky nebo kupující

**Lze tvorbu AI brát za originální, či za umění jako takové?**