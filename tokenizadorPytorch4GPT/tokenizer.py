#Create Tokenizer with Byte-Pair-Encoding for Tomi
import re
import pickle

class tokenizerTomi:
    def __init__(self, inputText, do_lower_case=True, cut_token="</w>", max_len=512):
        self.inputText = inputText
        self.tokenizedText = self.inputText
        self.do_lower_case = do_lower_case
        self.cut_token = cut_token
        self.max_len = max_len
        self.initTokenId = 0
        self.lastTokenId = 0
        self.forced_tokens = ["Title", "Tags", "Descripcion", "Texto", "<End>", "<-->"]
        self.except_token = ["🛑"]

        self.charDict = self.createCharDict(inputText)
        self.tokenDict = self.createTokenDict()
        self.tokenList = self.createTokenList()

        self.tokenEncoder = {}
        self.tokenDecoder = {}

    def createCharDict(self, inputText):
        charDict = {}
        for char in inputText:
            if char not in charDict:
                #Count the number of times a character appears in the text
                charDict[char] = inputText.count(char)
                if char in self.except_token:
                    print("Character: ", char, "Count: ", charDict[char]) # DEBUG
        return charDict

    def createTokenDict(self):
        tokenDict = {}
        for token in self.charDict:
            if self.charDict[token] >= 1:
                tokenDict[token] = self.charDict[token]
            #elif token in self.except_token:
            #    tokenDict[token] = 1
        return tokenDict

    def createTokenList(self):
        tokenList = []
        for token in self.tokenDict:
            tokenList.append(token)
        
        self.initTokenId = max([ord(c) for c in tokenList]) + 1
        print("InitTokenId: ", self.initTokenId) # DEBUG
        return tokenList

    def cleanText(self):
        #set tokenizedText to inputText
        self.tokenizedText = self.inputText
        #Check if the cut_token is in the tokenizedText, if is show an error

        print("Start Process...") # DEBUG
        if self.cut_token in self.tokenizedText:
            print("The cut_token is in the text, please remove it")
            return
        print("Splitting...") # DEBUG
        #Change all the spaces for the cut_token + space
        self.tokenizedText = self.tokenizedText.replace(' ', self.cut_token + ' ')
        #add a space before each punctuation mark using replace
        self.tokenizedText = self.tokenizedText.replace('.', self.cut_token + '.')
        self.tokenizedText = self.tokenizedText.replace(',', self.cut_token + ',')
        self.tokenizedText = self.tokenizedText.replace(';', self.cut_token + ';')
        self.tokenizedText = self.tokenizedText.replace(':', self.cut_token + ':')
        self.tokenizedText = self.tokenizedText.replace('!', self.cut_token + '!')
        self.tokenizedText = self.tokenizedText.replace('¡', self.cut_token + '¡')
        self.tokenizedText = self.tokenizedText.replace('?', self.cut_token + '?')
        self.tokenizedText = self.tokenizedText.replace('¿', self.cut_token + '¿')
        self.tokenizedText = self.tokenizedText.replace('(', self.cut_token + '(')
        self.tokenizedText = self.tokenizedText.replace(')', self.cut_token + ')')
        self.tokenizedText = self.tokenizedText.replace('[', self.cut_token + '[')
        self.tokenizedText = self.tokenizedText.replace(']', self.cut_token + ']')
        self.tokenizedText = self.tokenizedText.replace('{', self.cut_token + '{')
        self.tokenizedText = self.tokenizedText.replace('}', self.cut_token + '}')
        self.tokenizedText = self.tokenizedText.replace('<', self.cut_token + '<')
        # The char '>' is not allowed in the tokenizer.
        self.tokenizedText = self.tokenizedText.replace('\t', '\t' + self.cut_token)
        print("Finished Splitting...") # DEBUG

        return self.tokenizedText

    def buildTokenizer(self):
        print("Building Tokenizer...") # DEBUG
        #Declare the bool if is finished
        isFinished = False
        #Declare the ocurrences of the last token
        lastTokenOcurrences = 0
        #Declare the list of discarded tokens
        discardedTokens = []
        #Start the while loop
        while not isFinished:
            #Build a dictionary with the same keys and values as the tokenDict, what are not in the discardedTokens list
            actTokenDict = {k: v for k, v in self.tokenDict.items() if k not in discardedTokens}
            #Get the token with the highest ocurrences
            actToken = max(actTokenDict, key=actTokenDict.get)

            #print("Actual token: ", actToken, "with", self.tokenDict[actToken]) # DEBUG
            #Search in the substring tokenizedText for the positions of the actToken
            
            #print("Searching for: ", actToken) # DEBUG
            #print("Is in?", actToken in self.tokenizedText) # DEBUG
            #print("How many?", self.tokenizedText.count(actToken)) # DEBUG
            #print("First: ", self.tokenizedText.find(actToken)) # DEBUG
            
            if actToken == '?':
                actToken = '\?'
            elif actToken == ')':
                actToken = '\)'
            elif actToken == '(':
                actToken = '\('
            elif actToken == '\\':
                actToken = '\\\\'
            elif actToken == '+':
                actToken = '\+'
            elif actToken == '*':
                actToken = '\*'
            elif actToken == '[':
                actToken = '\['
            elif actToken == ']':
                actToken = '\]'

            #print("Searching for: ", actToken) # DEBUG
            positions = [m for m in re.finditer(actToken, self.tokenizedText)]
            actTokenPositions = [m.start() for m in re.finditer(actToken, self.tokenizedText)]
            #Check if the actToken + the cut_token is in the tokenizedText

            if actToken == '\?':
                actToken = '?'
            elif actToken == '\)':
                actToken = ')'
            elif actToken == '\(':
                actToken = '('
            elif actToken == '\\\\':
                actToken = '\\'
            elif actToken == '\+':
                actToken = '+'
            elif actToken == '\*':
                actToken = '*'
            elif actToken == '\[':
                actToken = '['
            elif actToken == '\]':
                actToken = ']'

            #if actToken in self.except_token:
            #    print("The token: ", actToken, "is in the except_token list, so it will be ignored")
            #    discardedTokens.append(actToken)
            if actToken + self.cut_token in self.tokenizedText:
                #Add the actToken to the discardedTokens list
                discardedTokens.append(actToken)
            else:
                #Do a list of the substring of the actToken with all their next characters
                #actTokenList = []
                discardedTokens.append(actToken)
                for i in range(len(actTokenPositions)):
                    subToken = self.tokenizedText[actTokenPositions[i]:actTokenPositions[i] + len(actToken) + 1]
                    if subToken not in self.tokenDict:
                        if self.tokenizedText.count(subToken) >= 2:
                            self.tokenDict[subToken] = self.tokenizedText.count(subToken)
                            #Substract to actToken Ocurrences the ocurrences of the subToken
                            prevOcurrences = self.tokenDict[actToken]
                            self.tokenDict[actToken] = self.tokenDict[actToken] - self.tokenizedText.count(subToken)
                            afterOcurrences = self.tokenDict[actToken]
                            #print(prevOcurrences == afterOcurrences, "-", actToken,"-", subToken,"-","Before: ", prevOcurrences, "After: ", afterOcurrences) # DEBUG
            #Check if self.tokenDict have the same size as discardedTokens
            if len(self.tokenDict) == len(discardedTokens):
                isFinished = True

        #Remove from the dictionary the tokens of tokenDict that have 0 ocurrences, except the ones in the tokenList.
        for token in list(self.tokenDict.keys()):
            #if self.tokenDict[token] == 0:
                #print("Should remove", token, "?") # DEBUG
            if self.tokenDict[token] == 0 and token not in self.tokenList:
                #print("Removing...", token) # DEBUG
                del self.tokenDict[token]
        print("Size of tokenDict: ", len(self.tokenDict)) # DEBUG

        #Create the tokenEncoder and tokenDecoder
        self.lastTokenId = self.initTokenId
        for token in self.tokenDict:
            if token == "🛑":
                print("Found the 🛑 token")
            #if token not in self.except_token:
            self.tokenEncoder[token] = self.lastTokenId
            self.tokenDecoder[self.lastTokenId] = token
            self.lastTokenId += 1
            #else:
                #self.tokenEncoder[token] = token
                #self.tokenDecoder[token] = token

        #        print("Is 🛑 in the tokenized","🛑" in self.tokenizedText) # DEBUG
        print("The exceptions tokens:",self.except_token, "🛑" in self.except_token) # DEBUG
        #print("🛑 encoder", self.tokenEncoder["🛑"]) # DEBUG
        print("Finished Building Tokenizer...") # DEBUG
        
    def Tokenize(self):
        #Using the tokenEncoder dictionary, encode the inputText
        #Order the tokenEncoder by the ocurrences of the tokens in the tokenDict

        tokenEncoderOrdered = list({k: v for k, v in sorted(self.tokenDict.items(), key=lambda item: item[1], reverse=True)})
        
        #Get a list from tokenEncoderOrdered with the tokens that are not in the tokenList
        tokenEncoderOrdered = [token for token in tokenEncoderOrdered if token not in self.tokenList]

        #print("OrderedDict: ", list(OrderedDict)[:10]) # DEBUG
        #1. Convert to list
        #tokenEncoderOrdered = [self.tokenEncoder]
        #print("TokenEncoderOrdered: ", tokenEncoderOrdered) # DEBUG

        #tokenEncoderOrdered = [token for token, Occ in list(sorted(self.tokenEncoder.items(), key=lambda t: t[1]))]
        
        #tokenEncoderOrdered.reverse()
        print("tokenEncoderOrdered First: ", tokenEncoderOrdered[:10]) # DEBUG
        print("Encoding...") # DEBUG
        self.tokenizedText = self.tokenizedText.replace(self.cut_token, '')
        #print("To Encode:" , self.tokenizedText) # DEBUG
        for token in tokenEncoderOrdered:
            #if token in self.except_token:
            #    print("Encoding: ", token, chr(self.tokenEncoder[token])) # DEBUG
            self.tokenizedText = self.tokenizedText.replace(token, chr(self.tokenEncoder[token]))
        for token in self.tokenList:
            self.tokenizedText = self.tokenizedText.replace(token, chr(self.tokenEncoder[token]))
        return self.tokenizedText

    def Decode(self, text):
        #Using the tokenDecoder dictionary, decode the inputText
        print("Decoding...") # DEBUG
        #print("To Decode:" , text) # DEBUG
        for token in self.tokenDecoder:
            text = text.replace(chr(token), self.tokenDecoder[token])
        return text


    def saveTokenizer(self, fileName):
        print("Reduce the size of the file...") # DEBUG
        self.tokenizedText = ""
        self.inputText = ""

        print("Saving Tokenizer...")
        #Use pickle to save the tokenizer
        with open(fileName, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadTokenizer(self, fileName):
        print("Loading Tokenizer...")
        #Use pickle to load the tokenizer
        with open(fileName, 'rb') as handle:
            self = pickle.load(handle)
        return self

    def introduceText(self, text):
        self.tokenizedText = text
        self.cleanText()

InputText = ""
#Open the file HuecoRandomFormated.txt and save the text in the variable textT
with open("HuecoRandomFormated.txt", "r", encoding="utf-8") as file:
    InputText = file.read()

codedText = ""
decodedText = ""
tokenizador = tokenizerTomi(InputText)
#Load the tokenizer
tokenizador = tokenizador.loadTokenizer("tokenizerHuecoRandom.pkl")
tokenizador.introduceText(InputText)
#tokenizador.cleanText()
#tokenizador.buildTokenizer()
codedText = tokenizador.Tokenize()
#print("Encoded", tokenizador.inputText, ":",codedText)
decodedText = tokenizador.Decode(codedText)
#print("Decoded", codedText, ":", decodedText)
#Check if InputText is the same as decodedText

#Calculate the compression ratio
compressionRatio = len(codedText) / len(InputText) * 100
print("Compression Ratio:", compressionRatio, "% - Coded:", len(codedText), " - Input:", len(InputText))
if InputText == decodedText:
    print("The text is the same")
else:
    print("The text is different")
    #Locate the position of the difference between the two texts
    for i in range(min(len(InputText), len(decodedText))):
        if InputText[i] != decodedText[i]:
            print("The difference is in position", i, "The character:", InputText[i], "is different than", decodedText[i])
            break
#Write two files, with the InputText and the decodedText using unicode
with open("InputText.txt", "w", encoding="utf-8") as file:
    file.write(InputText)
with open("DecodedText.txt", "w", encoding="utf-8") as file:
    file.write(decodedText)
#Write the codedText in a file
with open("CodedText.txt", "w", encoding="utf-8") as file:
    file.write(codedText)
tokenizador.saveTokenizer("tokenizerHuecoRandom.pkl")
#Do a list with all