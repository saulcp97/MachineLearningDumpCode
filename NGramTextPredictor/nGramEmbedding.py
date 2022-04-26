import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#
######################################################################
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that in an n-gram language model, given a sequence of words
# :math:`w`, we want to compute
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# Where :math:`w_i` is the ith word of the sequence.
#
# In this example, we will compute the loss function on some training
# examples and update the parameters with backpropagation.
#
#The probabilities of log_softmax can be normalized as torch.exp(model(features))

CONTEXT_SIZE = 64
EMBEDDING_DIM = 16
# We will use Shakespeare Sonnet 2

protoPhrase = ["Hello World"]

dictionaryElem = "0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z , . ! ? = + - * /".split() + ["\n" , " ", '']
print(dictionaryElem[:3])
vocab = dictionaryElem
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.augmented1 = nn.Linear(context_size * embedding_dim, 512)
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        #print(inputs.shape, embeds.shape)
        input = self.augmented1(embeds)
		
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    def dragonHead(self, head):
        actual = self.state_dict()
        for name, param in head.items():
            if name not in actual:
                continue
            param = param.data
            actual[name].copy_(param)
        self.eval()
    def showNames(self):
        for name,_ in self.state_dict().items():
            print("Nombre: ", name)
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(dictionaryElem), EMBEDDING_DIM, CONTEXT_SIZE)
model.load_state_dict(torch.load("memoryShard.pt"))
model.eval()
#model.showNames()
# Learning Rate
# Finetuning: 		0.000001
# Fast training 	0.001
optimizer = optim.SGD(model.parameters(), lr=0.000001)

def trainPhrase(text, showLoss = False):
	LocalLoss = 0
	for j in range(len(text) - (CONTEXT_SIZE)):
		next = text[j+CONTEXT_SIZE]
		prev = tuple(text[j:j+CONTEXT_SIZE])
		context_idxs = torch.tensor([word_to_ix[w] for w in prev], dtype=torch.long)
		model.zero_grad()
		#print("Shape Input: ", context_idxs.shape)
		log_probs = model(context_idxs)
		pV = list(log_probs[0])
		mult = 99 * (text[j+CONTEXT_SIZE - 1] == vocab[pV.index(max(pV))] and text[j+CONTEXT_SIZE - 1] != ".") + 1
		loss = loss_function(log_probs, torch.tensor([word_to_ix[next]], dtype=torch.long)) * mult
		loss.backward()
		optimizer.step()
		LocalLoss += loss.item()
	
	next = ""
	prev = tuple(text[-CONTEXT_SIZE:])
	context_idxs = torch.tensor([word_to_ix[w] for w in prev], dtype=torch.long)
	model.zero_grad()
	log_probs = model(context_idxs)
	loss = loss_function(log_probs, torch.tensor([word_to_ix[next]], dtype=torch.long)) * 0.1
	loss.backward()
	optimizer.step()
	LocalLoss += loss.item()
	
	if showLoss:
		print("Local Loss: ", LocalLoss)
	return LocalLoss

def presetRunTime(Times):
	INDEX = math.floor(random.random() * len(protoPhrase))
	losses = []
	test_sentence = "!" * CONTEXT_SIZE + protoPhrase[INDEX].lower()
	trigrams = [(tuple(test_sentence[i:i+CONTEXT_SIZE]), test_sentence[i+CONTEXT_SIZE]) for i in range(len(test_sentence) - (CONTEXT_SIZE))]
	trigrams.append((tuple(test_sentence[-CONTEXT_SIZE:]), ""))
	#print(trigrams[:3])
	context_idxs = torch.tensor([word_to_ix[w] for w in trigrams[0][0]], dtype=torch.long)
	pV = list(model(context_idxs)[0])
	ind = pV.index(max(pV))
	print("PreIteration", ind, vocab[ind], " probability: ", torch.exp(max(pV)))
	Iterations = Times
	for epoch in range(Iterations):
		total_loss = 0
		for context, target in trigrams:
			# Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
			# into integer indices and wrap them in tensors)
			context_idxs = torch.tensor([word_to_ix.get(w, word_to_ix[" "]) for w in context], dtype=torch.long)

			# Step 2. Recall that torch *accumulates* gradients. Before passing in a
			# new instance, you need to zero out the gradients from the old
			# instance
			model.zero_grad()
			# Step 3. Run the forward pass, getting log probabilities over next
			# words
			log_probs = model(context_idxs)
			# Step 4. Compute your loss function. (Again, Torch wants the target
			# word wrapped in a tensor)
			loss = loss_function(log_probs, torch.tensor([word_to_ix.get(target, word_to_ix[" "])], dtype=torch.long))
			#print(torch.tensor([word_to_ix[target]]))
			# Step 5. Do the backward pass and update the gradient
			loss.backward()
			optimizer.step()

			# Get the Python number from a 1-element Tensor by calling tensor.item()
			total_loss += loss.item()
		losses.append(total_loss)
	pV = list(model(context_idxs)[0])
	ind = pV.index(max(pV))
	print("PosIteration", ind, vocab[ind], " probability: ", torch.exp(max(pV)))
	print(losses)  # The loss decreased every iteration over the training data!

def imitationTraining(it):
	Iterations = it
	for i in range(Iterations):
		total_loss = 0
		prefix = "!" * CONTEXT_SIZE
		sentence = input("- ")
		frase = prefix + sentence.lower()
		for j in range(len(frase) - (CONTEXT_SIZE)):
			next = frase[j+CONTEXT_SIZE]
			prev = tuple(frase[j:j+CONTEXT_SIZE])
			context_idxs = torch.tensor([word_to_ix[w] for w in prev], dtype=torch.long)
			model.zero_grad()
			log_probs = model(context_idxs)
			loss = loss_function(log_probs, torch.tensor([word_to_ix[next]], dtype=torch.long))
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print("Local Loss: ", total_loss)
	torch.save(model.state_dict(), "memoryShard.pt")


def teachingFeelings(tries, divinity = False):
	Iterations = tries
	for i in range(Iterations):
		total_loss = 0
		prefix = "!" * CONTEXT_SIZE
		sentence = input("*you* ").lower()
		trainPhrase(prefix + sentence)
		
		frase = prefix + "*you* " + sentence
		SolAns = []
		maxmsgO = "*me* "
		calculust = frase + "\n*me* "
		
		if not divinity:
			while len(maxmsgO) < 200 + CONTEXT_SIZE:
				context_max = torch.tensor([word_to_ix[w] for w in calculust[-CONTEXT_SIZE:]], dtype=torch.long)
				pV = list(model(context_max)[0])
				maxi = pV.index(max(pV))
				maxmsgO += vocab[maxi]
				calculust += vocab[maxi]
				if vocab[maxi] == '':
					break
			print(maxmsgO)
		else:
			print("*me* ")
			divineEye(8, calculust)
		correction = frase + "\n*me* " + input("*what i should answer?* ").lower()
		trainPhrase(correction, True)

		#print(maxmsgO)
	torch.save(model.state_dict(), "memoryShard.pt")


def infiniteFeelings():
	complete = False
	prefix = "!" * CONTEXT_SIZE
	Text = "!" * CONTEXT_SIZE
	while not complete:
		print("################################")
		total_loss = 0
		sentence = input("*you* ").lower()
		if sentence == "":
			break
		#trainPhrase(prefix+sentence)
		randomTraining()
		
		Text = Text + "*you* " + sentence
		SolAns = []
		maxmsgO = "*me* "
		calculust = Text + "\n*me* "
		
		
		#Funciona raruno
		#print("*me* ")
		#divineEye(3, calculust)
		
		while len(maxmsgO) < 200 + CONTEXT_SIZE:
			context_max = torch.tensor([word_to_ix[w] for w in calculust[-CONTEXT_SIZE:]], dtype=torch.long)
			pV = list(model(context_max)[0])
			maxi = pV.index(max(pV))
			maxmsgO += vocab[maxi]
			calculust += vocab[maxi]
			if vocab[maxi] == '':
				break
				#minimum = pV.index(min(pV))
				#maxmsgO += vocab[minimum]
				#calculust += vocab[minimum]
				#print("Caracter calculado: ", calculust[-1])
			#maxmsgO += "."
		print(maxmsgO)
		
		
		correction = input("*what i should answer?* ").lower()
		if correction == "":
			break
		Text = Text + "\n*me* " + correction + "\n"
		err = 0
		for epoch in range(10):
			err = trainPhrase(Text, True)
		print("Error per Input letter: ", (err/len(Text)))
		#print(maxmsgO)
		print("Complete text: ", Text)
	torch.save(model.state_dict(), "memoryShard.pt")


def createRandomText():
	maxmsgO = "!" * CONTEXT_SIZE
	while len(maxmsgO) < 200 + CONTEXT_SIZE:
		context_max = torch.tensor([word_to_ix[w] for w in maxmsgO[-CONTEXT_SIZE:]], dtype=torch.long)
		pV = list(model(context_max)[0])
		maxi = pV.index(max(pV))
		maxmsgO += vocab[maxi]
		if vocab[maxi] == '':
			break
	print("Maximum: ", maxmsgO[CONTEXT_SIZE:])
# To get the embedding of a particular word, e.g. "beauty"
#print(model.embeddings.weight[word_to_ix["b"]])

def divineEye(wSize, provided = ""):
	maxmsgO = "!" * CONTEXT_SIZE + provided
	minAvProb = 1 / pow(len(dictionaryElem), wSize)	
	
	#print("Minimum prbability: ", minAvProb)
	while len(maxmsgO) < 200 + CONTEXT_SIZE + len(provided):
		bestP = -1
		bestVec = None
		vectorW = torch.tensor([word_to_ix[w] for w in maxmsgO[-CONTEXT_SIZE:]], dtype=torch.long)
		probG = torch.exp(model(vectorW)[0])
		PList = []
		#for i in range(len(dictionaryElem) - 1):
		bestP = -1
		bestVec = None
		#print("Caracter: ", dictionaryElem[i], probG[i])
		auxiL = maxmsgO + ""
		ProbPrel = 1
		while len(auxiL) < len(maxmsgO) + wSize:
			context_max = torch.tensor([word_to_ix[w] for w in auxiL[-CONTEXT_SIZE:]], dtype=torch.long)
			pV = list(model(context_max)[0])
			maxi = pV.index(max(pV))
			ProbPrel = ProbPrel * torch.exp(max(pV))
			#print(torch.exp(max(pV)), "/", ProbPrel)
			auxiL += vocab[maxi]
			if vocab[maxi] == '':
				break
		#print("GroundMax: ", auxiL[len(maxmsgO):], " - Prob: ", ProbPrel)
		minAvLocale = max(minAvProb, ProbPrel)
		#print("GroundMax: ", auxiL[len(maxmsgO):], " - Prob: ", ProbPrel, "/", minAvLocale)
		for i in range(len(dictionaryElem) - 1):
			PList.append((dictionaryElem[i], probG[i]))
		while len(PList) > 0:
			cha, prob = PList.pop(0)
			if prob >= minAvLocale and prob > bestP:
				#print("Letra ", cha, " prob: ", prob)
				aMsg = maxmsgO[-(CONTEXT_SIZE - len(cha)):] + cha
				veSup = torch.tensor([word_to_ix[w] for w in aMsg], dtype=torch.long)
				probG = torch.exp(model(veSup)[0])
				if len(cha) < wSize and cha[-1] != '':
					for i in range(len(dictionaryElem)):
						localProb = probG[i] * prob
						localString = cha + dictionaryElem[i]
						PList.insert(0, (localString, localProb))
				else:
					bestP = prob
					bestVec = cha
					#print("New Best: ", bestVec, " - ", prob)
			else:
				if bestVec == None and prob > minAvProb:
					bestP = prob
					bestVec = cha
		#print("Iteration ", i, ": ", bestVec, " / ", bestP)
		#print(bestP)
		#if bestVec == None:
		#	bestVec = auxiL[len(maxmsgO)-1:]
		maxmsgO += bestVec
		if maxmsgO[-1] == '':
			break;
	print(maxmsgO[(CONTEXT_SIZE + len(provided)):])
	#for i in range(len(dictionaryElem)):
	#			PList.append()
	#3 Letters Dictionary 2 pos= 3*2
	#11 12 13 22 21 23 33 31 32

#for k in range(1):
def randomTraining():
	presetRunTime(10)
	for epoch in range(4):
		for i in range(10):
			for j in range(10):
				if random.random() < 0.5:
					iText = "!" * CONTEXT_SIZE + "*you* What is " + str(i) + " + " + str(j) + "?\n*me* It is " + str(i + j) + "."
					trainPhrase(iText.lower())
					iText = "!" * CONTEXT_SIZE + "*you* What is " + str(j) + " * " + str(i) + "?\n*me* It is " + str(i * j) + "."
					trainPhrase(iText.lower())

#imitationTraining(0)
#createRandomText()
#teachingFeelings(2)
#teachingFeelings(2, divinity=True)
infiniteFeelings()
for i in range(256):
	randomTraining()
	presetRunTime(1)
	createRandomText()
#print("\nDivine Eye")
#divineEye(3)
torch.save(model.state_dict(), "memoryShard.pt")
######################################################################
# Exercise: Computing Word Embeddings: Continuous Bag-of-Words
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep
# learning. It is a model that tries to predict words given the context of
# a few words before and a few words after the target word. This is
# distinct from language modeling, since CBOW is not sequential and does
# not have to be probabilistic. Typically, CBOW is used to quickly train
# word embeddings, and these embeddings are used to initialize the
# embeddings of some more complicated model. Usually, this is referred to
# as *pretraining embeddings*. It almost always helps performance a couple
# of percent.
#
# The CBOW model is as follows. Given a target word :math:`w_i` and an
# :math:`N` context window on each side, :math:`w_{i-1}, \dots, w_{i-N}`
# and :math:`w_{i+1}, \dots, w_{i+N}`, referring to all context words
# collectively as :math:`C`, CBOW tries to minimize
#
# .. math::  -\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
#
# where :math:`q_w` is the embedding for word :math:`w`.
#
# Implement this model in Pytorch by filling in the class below. Some
# tips:
#
# * Think about which parameters you need to define.
# * Make sure you know what shape each operation expects. Use .view() if you need to
#   reshape.
#


#Modelo de predicción del medio (interesante para futuras versiones)
"""
raw_text = "We are about to study the idea"
# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print("N-Grams:\n\t"+str(data))


class CBOW(nn.Module):
    def __init__(self):
        pass

    def forward(self, inputs):
        pass
# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

make_context_vector(data[0][0], word_to_ix)  # example
"""