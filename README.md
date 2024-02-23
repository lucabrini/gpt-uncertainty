# GPT-Uncertainty

## Introduzione

I Large Language model offrono una vasta gamma di capacità incredibili. Tuttavia quando sbagliano è molto difficile che lo ammettano. 
È molto più probabile invece che generino una risposta che sembra molto confidente e ben strutturata ma che in realtà è completamente inventata. Questo fenomeno è chiamato **Allucinazione** ed è un problema molto frequente nei modelli generativi. 

Un modello di linguaggio viene addestrato su una enorme quantità di testo preso da Internet, libri o da letteratura. Il modello apprende le correlazioni e le co-occorrenze delle parole nei testi, cercando di riprodurre la struttura delle informazioni che ha visto. Tuttavia, non "capisce" quello che sta leggendo nel senso stretto del termine.
I modelli hanno dunque allucinazioni poiché essi non ragionano come un essere umano, non fanno reasoning, ma generano semplicemente testo sulla base dei token più probabili.

Quello che sarebbe bello facesse - e il progetto ambisce ad approfondire questa tematica - è fare domande al suo interlocutore in modo da apprendere più informazioni riguardo il dominio della richiesta postagli, diminuire il proprio livello di incertezza ed aumentare quindi il grado di confidenza della risposta.

### Esempio di applicazione

Immaginiamo di avere un modello LLM per le diagnosi mediche. Lo stato del modello dipende quindi dalle informazioni dei sintomi. Immaginiamo che la conoscenza attuale sia un insieme di sintomi che portano a più malattie. 

Quello che farebbe un medico è porre determinate domande al paziente in modo da aumentare la probabilità di identificare correttamente la condizione medica. Formalmente sta aumentando la probabilità di una malattia candidata ed abbassando quella delle altre.

Un modello LLM attuale invece, semplicemente genera e restituisce la malattia, che potrebbe anche inventare, a cui sono associati i token più probabili.

## Calcolo dell'incertezza

All'interno di un LLM possono essere considerati due tipi di incertezza:
- Incertezza legata alla generazione del testo, strettamente legata alle probabilità dei token
- Incertezza legata alla conoscenza del modello

Noi ci siamo concentrati sul calcolo empirico della seconda, utilizzando pipeline esterne per un approccio generale e non task specific.

### BSDetector

*Jiuhai Chen, Jonas Mueller* (2023). [Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness](https://doi.org/10.48550/arXiv.2308.16175)

L'approccio del BSDetector è stato scelto per il fatto che accedendo al modello gpt-3.5-turbo di OpenAI tramite API si ha accesso solo a pochi hyperparametri e alla risposta del modello. Altre informazioni come le distribuzioni di probabilità non sono accessibili.

#### Funzionamento

![bsdetector_pipeline](https://github.com/lucabrini/gpt-uncertainty/assets/86363063/dac4235c-791a-499c-b5a6-416ba4ae1000)

##### Observe Consistency

Il concetto alla base di questo step è molto semplice: quando viene posta ad un LLM una domanda di cui è insicuro, questo tende a rispondere ogni volta in maniera diversa.

Sulla base di ciò dunque vengono richiesti k samples al LLM e viene calcolata la similarità semantica tra ognuna delle k risposte e la risposta originale. Concettualmente significa che più l'LLM tende a dare risposte semanticamente diverse più l'incertezza è alta.

Quantitativamente parlando, la consistenza osservata tra i k samples è pari alla media delle k similarità sematiche.

###### deBERTa

La similarità semantica viene calcolata tra la risposta originale ed ognuno dei k samples utilizzando un modello di Natural Language Processing chiamato [deBERTa](https://huggingface.co/docs/transformers/model_doc/deberta)
Le due stringhe, separate da un SEP token, vengono codificate insieme al contesto ( in questo caso la domanda posta al LLM ) e date in input al modello.
Per ovviare ai problemi di simmetria, si considerano entrambi gli ordini e la similitarità finale è pari alla media tra le due.

```python
y = context + y
yi = context + yi
p = compute_probabilities(y, yi)
p1 = compute_probabilities(yi, y)

def compute_probabilities(x: str, y: str):
  sequence = x + " [SEP] " + y
  encoded_input = tokenizer.encode(sequence, padding=True)

  outputs = model((torch.tensor([encoded_input]).to(device).clone().detach()))
  predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()

  torch.mps.empty_cache()

  return 1-predicted_probability[2] # Prendo la similarità
```

##### Self Reflection

Secondo il paper, la "self-reflection certainty" rappresenta la confidenza stimata del LLM quando gli viene chiesto se una certa risposta, in un determinato contesto, è corretta.

Diversamente dal fare sampling delle risposte o dal calcolare entropie/similarità basate sulle probabilità dei token (di cui non abbiamo accesso), la self-reflection certainty è una valutazione della confidenza intrinseca al LLM. Questo avviene sulla base di [Kadavath et al., 2022; Lin et al., 2022]([https://doi.org/10.48550/arXiv.2207.05221](https://doi.org/10.48550/arXiv.2207.05221)), il quale afferma che i LLM attuali sono in grado di riconoscere forte evidenze e valutare affermazioni.

Per calcolare la self-reflection certainty viene chiesto al modello di classificare la risposta in una delle categorie ad ognuna delle quali è associato un peso:

| Categoria | Spiegazione           | Peso |
| --------- | --------------------- | ---- |
| A         | Risposta corretta     | 1.0  |
| B         | Risposta non corretta | 0.0  |
| C         | Non è sicuro          | 0.5  |

Inoltre, viene chiesta anche la motivazione. Questo approccio è chiamato Chain of Thoughts prompting ed è provato (*[Wei et al. (2022)]([https://doi.org/10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903))*) che migliori la correttezza di una risposta

Non viene chiesto un valore numerico (tra 0 e 100) perché si è notato (*[Peng et al. (2023)](https://doi.org/10.48550/arXiv.2304.03277](https://doi.org/10.48550/arXiv.2304.03277))*) che c'è Accondiscendence bias: il modello tende a dare valori di correttezza superiori al 90% anche se la risposta è del tutto sbagliata.

##### Incertezza finale

BSDetector aggrega l'observed consistency e la self-reflection certainty in uno score di confidenza
$$C = βO + (1 − β)S$$
Dove:
- $β$ è un parametro di tradeoff tra self-reflection e observed consistency
- $S$ è la self-reflection certainty
- $O$ è l'observed consistency

### Prompt

#### Observe Consistency

```python
"""
Please strictly use the following template to provide answer: explanation: [insert step-by-step analysis], answer: [provide your answer] + Question: [User Provided]
"""
```

#### Self-reflection Certainty
```python
"""
1. Question: [User Provided], Proposed Answer: [User/LLMs Provided]. Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure. The output should strictly use the following template: explanation: [insert analysis], answer: [choose one letter from among choices A through C] 

2. Question: [User Provided], Proposed Answer: [User/LLMs Provided]. Are you really sure the proposed answer is correct? Choose again: (A) Correct (B) Incorrect (C) I am not sure. The output should strictly use the following template: explanation: [insert analysis], answer: [choose one letter from among choices A through C]
"""
```

## 20 Questions Game
[ChatGPT’s Information Seeking Strategy: Insights from the 20-Questions Game](https://aclanthology.org/2023.inlg-main.11.pdf)

Il gioco delle 20 domande è composto da due giocatori. Il primo pensa ad un oggetto (target d'ora in poi) e il secondo deve indovinare il target tra una lista di oggetti candidati ponendo delle domande a cui si può rispondere solo con si o no.

Questo gioco viene usato per lo studio dello sviluppo cognitivo e capacità di information seeking negli umani: da bambini a adulti. Ed è anche uno dei modi ideali per studiare la capacità di apprendimento dell'informazione così come la capacità di porre le giuste domande per LLM.

### Impianto generale
Si considerano due "agenti" LLM:
- Questioner: data una lista di candidati, pone le domande.
- Oracle: dato un target e una lista di candidati risponde alle domande poste dal Questioner

Il paper considera vari game set differenziati dal numero di candidati iniziali e dalla similarità degli stessi all'interno di un game set. Si consideri per esempio i due insiemi di candidati seguenti:
- [Labrador, Pastore Tedesco ]
- [Cane, Gatto]
Il secondo insieme è più facile da analizzare rispetto al primo.

Per ogni gioco all'interno del game set vengono generati i dialoghi tra Questioner e Oracle

### Prompt

#### Questioner

```json
[
	{
		"role": "system", 
		"content": "You are playing an interactive game with the user, who is assigned an item from a list of candidates. Ask as few questions as possible to identify the item, making only one question at each turn.\n\nThe user can only respond with 'yes' or 'no'."
	},
	{
		"role": "user", "content": "This is the list of candidates: {candidates}."
	}
]
```

#### Oracle
```json
[{
	"role": "system", 
	"content": "You are playing an interactive game with the user, in which you are assigned one item from a list of candidates. \nThe user will have to guess which one it is by asking yes/no questions, and you have to stricly respond to each question only with 'yes' or 'no'.\nYou must respond with 'Yes! That's correct.' only if the user guesses exactly (letter by letter) your assigned item: .\nThe item assigned to you is {target}."
}]
```
## Il nostro lavoro

Il nostro lavoro è stato quello di andare a calcolare l'incertezza del modello, utilizzando il BSDetector, ad ogni passo del dialogo tra Questioner ed Oracle.

Preso lo script originale del 20 questions game del paper, esso è stato adattato introducendo dunque il calcolo dell'incertezza mediante BSDetector.

È stata definita la classe `LLModelWrapper` che wrappa le API di OpenAI e implementa la pipeline del BSDetector. Espone i metodi:
- ask
- run_bsdetector
- observe_consistency
- sample_multiple_outputs

Un dialogo de `20 questions game` è un'insieme di coppie <domanda, risposta>. 
Ogni generazione di domanda e di risposta è stata generata utilizzando la classe precedente, la quale calcola anche la confidenza della generazione.

La confidenza è stata calcolata sia sulla generazione della domanda che sulla generazione della risposta.

### Esempio 
Ecco un esempio di dialogo tra Questioner e Oracle

- **Target**: fox
- **Item candidati**: [elk,chicken,robin,starling,fox,partridge,hamster,buffalo]

| step | target | question                                             | answer               | question_confidence | question_observed_consistency | question_self_reflection | answer_confidence | answer_observed_consistency | answer_self_reflection |
| ---- | ------ | ---------------------------------------------------- | -------------------- | ------------------- | ----------------------------- | ------------------------ | ----------------- | --------------------------- | ---------------------- |
| 0    | fox    | Is your item a bird?                                 | No.                  | 0.71329             | 0.59041                       | 1.0                      | 0.93952           | 0.91361                     | 1.0                    |
| 1    | fox    | Is your item a mammal?                               | Yes.                 | 0.62481             | 0.89259                       | 0.0                      | 0.44414           | 0.20592                     | 1.0                    |
| 2    | fox    | Is your item typically kept as a pet?                | No.                  | 0.34034             | 0.48619                       | 0.0                      | 0.99567           | 0.99381                     | 1.0                    |
| 3    | fox    | Is your item larger than a dog?                      | No.                  | 0.26675             | 0.38107                       | 0.0                      | 0.64375           | 0.91964                     | 0.0                    |
| 4    | fox    | Is your item commonly found in forests or woodlands? | Yes.                 | 0.3902              | 0.55743                       | 0.0                      | 0.54554           | 0.67221                     | 0.25                   |
| 5    | fox    | Is your item a fox?                                  | Yes! That's correct. | 0.7925              | 0.70357                       | 1.0                      | 0.99548           | 0.99355                     | 1.0                    |

### Step by Step analysis

Un ulteriore approccio che abbiamo utilizzato, su spunto di ulteriori analisi del paper 20 questions game, è stato quello di verificare per ogni coppia <domanda, risposta> e per ogni item candidato, se questo potesse essere un candidato dell'insieme di coppie <domanda, risposta> ad un istante t ( con $t \in [0, \space dialogo.lunghezza]$ ).

Per farlo abbiamo utilizzato un ulteriore agente LLM che verifica, dato un dialogo $D_t$ (fino un certo istante t) e un item se quest'ultimo soddisfa tutte le coppie di <domanda, risposta> di $D_t$. Questa richiesta viene campionata $k=5$ volte e viene conteggiato il numero di occorrenze positive. 

Un item i-esimo quindi, in un certo istante t, avrà associato uno score di probabilità: $$Sit = \frac{conteggi\_positivi}{k}$$
Dopo aver calcolato tutti gli score e dopo averli normalizzati, viene inoltre calcolata la distribuzione di probabilità  tra tutti gli item candidati.

#### Prompt

```python
"""
Given this dialogue:
	{dialogue} 
Is the dialogue true for this item?
Item: {candidate}

The output must strictly use the following template:"
	EXPLANATION: [insert your analysis];
	ANSWER: [is the dialogue true, yes or no]
"""
```

## Analisi dei risultati

### Superclasse dei target
Se il Questioner chiede se l'item è una superclasse del target, l'Oracle risponde positivamente chiudendo il dialogo.

#### Esempio:
- **Target:** fawn
- **Item candidati:** [ budgie, zebra, deer, penguin, pelican, sheep, fawn, ostrich ]

```
questioner: Is the item you have assigned an animal?
answerer: Yes.
questioner: Can the animal you have assigned fly?
answerer: No.
questioner: Does the animal you have assigned have hooves?
answerer: Yes.
questioner: Is the animal you have assigned a deer?
answerer: Yes! That's correct.
```

### Item disgiunti

Consideriamo il seguente dialogo:
- **Target:** zebra
- **Item candidati:** [ pelican, bear, zebra, walrus, vulture, chickadee, finch, otter ]
```
questioner: Is your item a bird?
answerer: No.
questioner: Is your item a mammal?
answerer: Yes.
questioner: Is your item a marine mammal?
answerer: No.
questioner: Is your item a land mammal?
answerer: Yes.
questioner: Is your item a carnivore?
answerer: Yes.
questioner: Is your item a large carnivore?
answerer: Yes.
questioner: Is your item a bear?
answerer: No.
questioner: Is your item a walrus?
answerer: No.
questioner: Is your item an otter?
answerer: No.
questioner: Is your item a vulture?
answerer: No.
questioner: Is your item a zebra?
answerer: Yes! That's correct.
```

E concentriamoci sul seguente pezzo:

```
questioner: Is your item a mammal?
answerer: Yes.
questioner: Is your item a marine mammal?
answerer: No.
questioner: Is your item a land mammal?
```

marine mammal e land mammal sono due classi disgiunte. Se un item è un mammifero e non è di un mammifero marino, allora sicuramente sarà un mammifero di terra. GPT non riconosce le classi disgiunte e questo fa inevitabilmente aumentare gli step del dialogo.

### Item rari o poco usati

#### Esempio
- **Target**: rhubarb
- **Item candidati**: [ falcon, owl, rhubarb, goose, cherry, apple, rooster, grapefruit ]

```
questioner: Is the item an animal?
answerer: No.
questioner: Is the item a fruit or vegetable?
answerer: Yes.
questioner: Is the item a fruit?
answerer: No.
questioner: Is the item a vegetable?
answerer: Yes.
questioner: Is the item a leafy vegetable?
answerer: No.
questioner: Is the item a root vegetable?
answerer: No.
questioner: Is the item a vegetable that grows above the ground?
answerer: Yes.
questioner: Is the item red or orange in color?
answerer: No.
questioner: Is the item green in color?
answerer: No.
questioner: Is the item yellow in color?
answerer: No.
questioner: Is the item a purple or violet color?
answerer: No.
questioner: Is the item brown in color?
answerer: No.
questioner: Is the item white in color?
answerer: No.
questioner: Is the item black in color?
answerer: No.
questioner: Is the item a vegetable that is commonly yellowish-green in color?
answerer: No.
questioner: Is the item a vegetable that is commonly purple in color?
answerer: No.
questioner: I apologize, but I am unable to determine the item based on the information provided. Could you please let me know the name of the vegetable?
answerer: Yes! That's correct. The item assigned to me is rhubarb.
```

### Observed-consistency della domanda

In molti casi, l'observed consistency della domanda è molto bassa. Questo è dovuto al fatto che la generazione dei samples delle domande del Questioner è ogni volta semanticamente diversa dalla domanda generata inizialmente.

Da ciò si deduce che non sempre GPT genera la domanda che porta all'information gain più alto. Se così non fosse (quindi generi sempre la domanda che porta all' IG più alto), l'observed consistency sarebbe sempre molto alta.

### Observed-consistency della risposta

Una bassa observed consistency della risposta significa che l'Oracle GPT non conosce la risposta alla domanda postagli dal Questioner.

### Tempi di generazione molto alti
Poiché vengono eseguiti k sample e un modello NLP per ogni risposta generata dall'LLM i tempi crescono molto.
$$O(token * k)$$
### Problema legato alle distribuzioni di probabilità in istanti diversi

Consideriamo il dialogo:
- **Target:** seal
- **Item candidati**: [ shotgun, seal, hamster, chipmunk, deer, spear, rocket, tomahawk ]

e le distribuzioni di probabilità per ogni coppia <domanda, risposta>

| intra_dialogue_id | target | question                            | answer               | p_distribution                                                                                                                                 |
| ----------------- | ------ | ----------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 0                 | seal   | Is your item a living organism?     | Yes.                 | {'shotgun': 0.0, 'seal': 0.25, 'hamster': 0.25, 'chipmunk': 0.25, 'deer': 0.25, 'spear': 0.0, 'rocket': 0.0, 'tomahawk': 0.0}                  |
| 1                 | seal   | Is your item a mammal?              | Yes.                 | {'shotgun': 0.0, 'seal': 0.25, 'hamster': 0.25, 'chipmunk': 0.25, 'deer': 0.25, 'spear': 0.0, 'rocket': 0.0, 'tomahawk': 0.0}                  |
| 2                 | seal   | Is your item a domesticated animal? | No.                  | {'shotgun': 0.1429, 'seal': 0.1429, 'hamster': 0.0, 'chipmunk': 0.1429, 'deer': 0.1429, 'spear': 0.1429, 'rocket': 0.1429, 'tomahawk': 0.1429} |
| 3                 | seal   | Is your item a small animal?        | No.                  | {'shotgun': 0.1667, 'seal': 0.1667, 'hamster': 0.0, 'chipmunk': 0.0, 'deer': 0.1667, 'spear': 0.1667, 'rocket': 0.1667, 'tomahawk': 0.1667}    |
| 4                 | seal   | Does your item have antlers?        | No.                  | {'shotgun': 0.1429, 'seal': 0.1429, 'hamster': 0.1429, 'chipmunk': 0.1429, 'deer': 0.0, 'spear': 0.1429, 'rocket': 0.1429, 'tomahawk': 0.1429} |
| 5                 | seal   | Is your item a seal?                | Yes! That's correct. | {'shotgun': 0.0, 'seal': 1.0, 'hamster': 0.0, 'chipmunk': 0.0, 'deer': 0.0, 'spear': 0.0, 'rocket': 0.0, 'tomahawk': 0.0}                      |

Come possiamo notare all'istante 0 vengono esclusi i candidati:
- shotgun
- spear
- rocket
- tomahawk

GPT ha quindi escluso che questi item siano dei candidati. Tuttavia, come si può notare al passo 2, nonostante il candidato shotgun sia stato escluso al passo 0, esso è ritornato a far parte della lista dei possibili candidati.

Questo fenomeno si presenta in molti dei dialoghi analizzati. Ciò significa che GPT tendenzialmente non considera le informazioni di cui è venuto a conoscenza negli step precedenti


## References

### 20 Questions Game

- https://aclanthology.org/attachments/2023.inlg-main.11.Supplementary_Attachment.pdf
- https://aclanthology.org/2023.inlg-main.11.pdf
- https://github.com/leobertolazzi/20q-chatgpt

### Measure GPT Uncertainity

- https://arxiv.org/pdf/2302.09664.pdf
- https://github.com/sylinrl/calibratedmath
- https://arxiv.org/pdf/2308.16175.pdf

