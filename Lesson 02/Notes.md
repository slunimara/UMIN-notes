- Zdroje: Kniha – Deep Learning with Python, Kaggle Kurz – Intro to Deep Learning, Wikipedie – [FFNN](https://en.wikipedia.org/wiki/Feedforward_neural_network), ostatní zmíněné přímo v textu
- Kaggle kurz – Intro to Deep Learning kapitola 2. a 3.

## Neuronová Sít
- síť se dá reprezentovat jako funkce $F$ (síťová funkce) přiřazující vstupům výstupy
- organizace neuronů do vrstev
	- vrstvu můžeme chápat jako jakoukoli transformaci dat
	- existuje mnoho druhů vrstev:
		- například: hustě propojené, aktivační, konvoluční, rekurentní, atd.
		- Seznam některých vrstev v dokumentaci [Keras layers API](https://keras.io/api/layers/)
	- **Aktivační Funkce (AF)** je funkce která se aplikuje na výstup každé vrstvy
		- historicky nejpoužívanějšími byly sigmoid a hyperbolický tangens
		- nyní se primárně používá ReLU (Rectified Linear Unit), kde je záporná část "usměrněna" na nulu
		- bez aktivačních funkcí se NN mohou naučit jen lineární vztahy 
		- přehledná [tabulka](https://en.wikipedia.org/wiki/Activation_function) aktivačních funkcí 

### Praktický příklad plně propojené sítě
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
	# Síť přijímá 2 vstupy.
    layers.Dense(units=4, input_shape=[2]),
    layers.Activation("relu"),
    # Aktivační funkce vrstvy pomocí funkce ReLu.
    layers.Dense(units=3, activation='relu'), 
    # Aktivační funkce lze zapsat i v argumentu.
    layers.Dense(units=1),
    # Výstupem sítě je jedna hodnota.
])
```

- **Pozn.** hluboké sítě NN jsou lepší než široké NN – [proč?](https://www.reddit.com/r/MachineLearning/comments/h0g83p/d_why_are_deeper_networks_better_than_wider/)
### Feed-Forward Neural Network
- česky dopředná neuronová síť
- orientovaný acyklický graf
- data proudí pouze jedním směrem (od vstupní k výstupní vrstvě)
- vhodný pro klasifikaci i regresi

- **Pozn.** u NN, která řeší regresní problém je výstupem reálné číslo
- **Pozn.** u NN, která řeší klasifikační problém je výstupem vektor pravděpodobností příslušnosti vstupu k předem určeným kategoriím
## Kvalita sítě
- kvalitu neuronové sítě určujeme pomocí **chybové funkce** (loss function, objective function)
	- zjednodušeně vezme cílovou hodnotu a porovná ji s predikovaným výstupem, z porovnání následně určí **chybu sítě** (jak moc se síť zmýlila v predikované hodnotě oproti vstupu)
- běžné chybové funkce pro regresní problémy jsou například:
	- pro regresní problémy: **MAE** – mean absolute error, **MSE** – mean squared error
	- pro klasifikační problémy: **Binary Cross-Entropy**, **Categorical Cross-Entropy**
- Zdroje: [Chybové funkce](https://medium.com/artificialis/neural-network-basics-loss-and-cost-functions-9d089e9de5f8), [Chybové funkce 2](https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9)

## Učení sítí
- **učení sítě** = úprava vah tak, aby se co nejvíce minimalizovala chyba sítě
- Jak probíhá?
	- Učení probíhá na za pomocí **chybové funkce** a **optimalizačního algoritmu**. Optimalizační algoritmus upravuje váhy neuronové sítě tak, aby minimalizoval chybu sítě. Úprava vah se nazývá učení sítě.
	- Sada vah, která minimalizuje chybu chybové funkce, je považována za řešení problému učení (globální optimum chybové funkce). Zde je derivace chybové funkce f’(w) = 0 neboli přírůstek (rychlost změny) funkce v tomto bodě je 0. Pozor, při hledání můžeme narazit na lokální minimum.

### Optimalizační algoritmus 
- angl. optimizers, česky by se to dalo přeložit jako optimalizátor
- určuje průběh učení
- většina optimalizátorů využívá tzv. [gradientní sestup](https://cs.wikipedia.org/wiki/Gradientn%C3%AD_sestup) (nebo stochastický grandientní sestup, když data bereme dávkově)
- Zdroj: [Optimalizátory, GD, Learning rate](https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0)

#### Rychlost učení
- při procesu učení sítě chceme, aby se síť blížila k optimálnímu řešení
- to jakou rychlostí se k onomu optimu blíží nazýváme **learning rate**
	- malé learning rate = pomalejší konvergence k optimu

### Praktický příklad
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(
	    units=4, 
	    input_shape=[2],
	    activation='relu'
	),
    layers.Dense(units=3, activation='relu'),
    layers.Dense(units=1),
])

model.compile(
	optimizer="adam",
	loss="mae",
)

result = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=256,
    epochs=10,
    verbose=False,
)
```

- [Simulátor neuronové sítě](https://playground.tensorflow.org) 
	- Doporučuji vyzkoušet různé hodnoty parametrů a dat.

## Backpropagation
- [Simulátor dopředného a zpětného průchodu](https://www.mladdict.com/neural-network-simulator) kde lze pěkně vidět výpočet NN u problému XOR.
- [Krok po kroku]( https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) jak backpropagation funguje. 


