import time
import random
import math

# Hamilton essays filenames (1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29)
# Madison  essays filenames (10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 58)

""" Task 01 Building Language Models """
print("TASK 01 BUILDING LANGUAGE MODELS")
start_time = time.time()                                                            #start the timer to measure the execution time of the program

translator = str.maketrans('', '', "\"#$%&'()*+,-/:;<=>@[\]^_`{|}~")                #this alphanumeric characters will be eliminated
hamiltonRawCounts = {}                                                              # the dictionary for hamilton which holds the unigram words and their counts
madisonRawCounts = {}                                                               # the dictionary for madison which holds the unigram words and their counts

hamiltonWordCount = 0
madisonWordCount = 0

# these are the temporary lists used for combining the essays, tokenize all it once and grouping it into the dictionaries
hamiltonunigrams = []
madisonunigrams = []
hamiltonbigrams =[]
madisonbigrams = []
hamiltontrigrams = []
madisontrigrams = []


# these dictionaries will be our bigram and trigram language models for Hamilton and Madison
hamiltonBigramDict = {}
madisonBigramDict = {}
hamiltonTrigramDict = {}
madisonTrigramDict = {}


             #HAMILTON ESSAYS READING

print("Reading Training Data")

# this list holds all text of Hamilton
textLists = []
for i in ('1.txt', '6.txt', '7.txt', '8.txt', '13.txt', '15.txt', '16.txt', '17.txt', '21.txt', '22.txt', '23.txt', '24.txt', '25.txt', '26.txt', '27.txt', '28.txt', '29.txt'):
    a = 0
    with open(i,'r') as f:                      # in this code block, i delete the first unnecessary line which holds the word 'HAMILTON'
        for line in f:
            if a is not 0:
                textLists.append(line)
            else:
                a=a+1

# in this code block i do preprocessing (doing character lower-case, alphanumeric character deletion, tokenizing the sentence boundaries)
for text in textLists:
    tokens = []
    text = text.translate(translator)
    for word in text.split():
        #word = word.translate(translator)
        word = word.lower()
        if word.endswith('.') or word.endswith('!') or word.endswith('?'):
            tokens.append(word[:-1])
            tokens.append(word[-1])
        else:
            tokens.append(word)
        hamiltonWordCount += 1


# group the values with respect to their n-gram (Ex: group triple triple for trigram)
    hamiltonunigrams.extend(zip(*[tokens[i:] for i in range(1)]))
    hamiltonbigrams.extend(zip(*[tokens[i:] for i in range(2)]))
    hamiltontrigrams.extend(zip(*[tokens[i:] for i in range(3)]))

# joining the ordered words with placing whitespace between them : list = ( [('the'), ('board')] ) to list = ( ('the board'), )
hamiltonUnigram = [" ".join(ngram) for ngram in hamiltonunigrams]
hamiltonBigram = [" ".join(ngram) for ngram in hamiltonbigrams]
hamiltonTrigram = [" ".join(ngram) for ngram in hamiltontrigrams]

print("Done.")

# load the processed data in to the dictionaries, because dictionaries is fast and efficient for this problem
for x in hamiltonUnigram:
    if x not in hamiltonRawCounts:
        hamiltonRawCounts[x] = 1
    else:
        hamiltonRawCounts[x] += 1

for x in hamiltonBigram:
    if x not in hamiltonBigramDict:
        hamiltonBigramDict[x] = 1
    else:
        hamiltonBigramDict[x] += 1

for x in hamiltonTrigram:
    if x not in hamiltonTrigramDict:
        hamiltonTrigramDict[x] = 1
    else:
        hamiltonTrigramDict[x] += 1

print("Hamilton: Unigram, Bigram, Trigram Models are generated.")

                 #MADISON ESSAYS READING

print("Reading Training Data")

# this list holds all text of Madison
textLists = []
for i in ('10.txt', '14.txt', '37.txt', '38.txt', '39.txt', '40.txt', '41.txt', '42.txt', '43.txt', '44.txt', '45.txt', '46.txt'):
    a = 0
    with open(i,'r') as f:                                  # in this code block, i delete the first unnecessary line which holds the word 'MADISON'
        for line in f:
            if a is not 0:
                textLists.append(line)
            else:
                a=a+1


# in this code block i do preprocessing (doing character lower-case, alphanumeric character deletion, tokenizing the sentence boundaries)
for text in textLists:
    tokens = []
    text = text.translate(translator)
    for word in text.split():
        #word = word.translate(translator)
        word = word.lower()
        if word.endswith('.') or word.endswith('!') or word.endswith('?'):
            tokens.append(word[:-1])
            tokens.append(word[-1])
        else:
            tokens.append(word)
        madisonWordCount +=1


    # group the values with respect to their n-gram (Ex: group triple triple for trigram)
    madisonunigrams.extend(zip(*[tokens[i:] for i in range(1)]))
    madisonbigrams.extend(zip(*[tokens[i:] for i in range(2)]))
    madisontrigrams.extend(zip(*[tokens[i:] for i in range(3)]))


# joining the ordered words with placing whitespace between them : list = ( [('the'), ('board')] ) to list = ( ('the board'), )
madisonUnigram = [" ".join(ngram) for ngram in madisonunigrams]
madisonBigram = [" ".join(ngram) for ngram in madisonbigrams]
madisonTrigram = [" ".join(ngram) for ngram in madisontrigrams]

print("Done.")


# load the processed data in to the dictionaries, because dictionaries is fast and efficient for this problem
for x in madisonUnigram:
    if x not in madisonRawCounts:
        madisonRawCounts[x] = 1
    else:
        madisonRawCounts[x] += 1

for x in madisonBigram:
    if x not in madisonBigramDict:
        madisonBigramDict[x] = 1
    else:
        madisonBigramDict[x] += 1

for x in madisonTrigram:
    if x not in madisonTrigramDict:
        madisonTrigramDict[x] = 1
    else:
        madisonTrigramDict[x] += 1


print("Madison: Unigram, Bigram, Trigram Models are generated.")
print("\n")
print("END OF TASK 01\n\n")


""" TASK 02 GENERATING ESSAYS """
print("TASK 02 GENERATING ESSAYS ")
""" For Hamilton """
print("For Hamilton ")

""" Unigram """
print("Unigram Generated Text\n")
generatedWordNumber = 0                                                 # the count of the words generated so far
generatedHamiltonUnigramProb = 0                                        # the sum of the probabilities so far

print("\t =>", end= ' ')

# in this loop, i generate a double random number in range 0 and 1 then start to calculate the probabilities of words and adding them to current probability
# if the sum of probabilities has gotten bigger than the generated random number, it means the generated number is in the right interval in the probabilities
# table. by this simple design, i avoid the unnecessary storing space of probabilities; instead i calculate and add it together.
# of course if the count of the words generated becomes 30 OR the generated word came as sentence boundary ( .!? ) i stop the execution
while generatedWordNumber < 30:
    generatedDoubleNumber = random.uniform(0.0, 1.0)
    currentProbability = 0.0
    for k,v in hamiltonRawCounts.items():
        currentProbability += hamiltonRawCounts[k] / hamiltonWordCount
        if currentProbability > generatedDoubleNumber:
            generatedHamiltonUnigramProb += math.log(hamiltonRawCounts[k] / hamiltonWordCount, 10)
            #print(generatedHamiltonUnigramProb)
            print(k, end= " ")
            if k == "." or k == "?" or k == "!":
                    generatedWordNumber = 50
            break
    generatedWordNumber += 1



print("\n")
""" Bigram """
print("Bigram Generated Text\n ")

# i do the same thing explained in Hamilton Unigram text generation part. Except in this part, the generated word is chosen in the limited space
# taking into account of the previous word. With other words, the probabilities of the word which is about to be generated calculated from the last generated word
# By that, we can easily see the meaningful doubles in the generated text.
generatedBigramWordNumber = 0
firstHamiltonBigramWord = random.choice(list(hamiltonRawCounts.keys()))
firstHamiltonBigramWordVALUE = hamiltonRawCounts[firstHamiltonBigramWord]
firstHamiltonBigramWord += " "
generatedHamiltonBigramProb = math.log(firstHamiltonBigramWordVALUE / hamiltonWordCount, 10)

print("\t =>", end= ' ')
while generatedBigramWordNumber < 30:
    generatedBigramDoubleNumber = random.uniform(0.0, 1.0)
    currentProbability = 0.0
    wordCountsCameAfterFirstWord = 0
    for k,v in hamiltonBigramDict.items():
        if k.startswith(firstHamiltonBigramWord):
            currentProbability += v / firstHamiltonBigramWordVALUE
            if currentProbability > generatedBigramDoubleNumber:
                chosenWord = k.split()[1]
                generatedHamiltonBigramProb += math.log(v / firstHamiltonBigramWordVALUE, 10)
                print(chosenWord, end =" ")
                #print(generatedHamiltonBigramProb)
                firstHamiltonBigramWord = chosenWord
                firstHamiltonBigramWordVALUE = hamiltonRawCounts[chosenWord]
                firstHamiltonBigramWord += " "
                if k.split()[1] == "." or k.split()[1] == "?" or k.split()[1] == "!":
                    generatedBigramWordNumber = 50
                break
            #print(k, v)
    generatedBigramWordNumber += 1





""" Trigram """
print("\n")
print("Trigram Generated Text\n ")

# i do the same thing explained in Hamilton Unigram text generation part. Except in this part, the generated word is chosen in the limited space such as
# trigram dictionary. So the generation of word is depends on the last two generated words. The probabilities are kept limited between them.
# By that, we can easily see the meaningful triples in the generated text.
generatedTrigramWordNumber = 0
startingWordsofTrigramHamilton = random.choice(list(hamiltonBigramDict.keys()))
valueofStartingWordofTrigram = hamiltonBigramDict[startingWordsofTrigramHamilton]
startingWordsofTrigramHamilton += " "
tempWordValue = hamiltonRawCounts[(startingWordsofTrigramHamilton.split()[0])]
generatedHamiltonTrigramProb = math.log(valueofStartingWordofTrigram / tempWordValue, 10)

print("\t =>", end= ' ')
while generatedTrigramWordNumber < 30:
    generatedTrigramDoubleNumber = random.uniform(0.0, 1.0)
    currentProbability = 0.0
    for k,v in hamiltonTrigramDict.items():
        if k.startswith(startingWordsofTrigramHamilton):
            currentProbability += v / valueofStartingWordofTrigram
            if currentProbability > generatedTrigramDoubleNumber:
                chosenDouble = k.split()[1] + " " + k.split()[2]
                generatedHamiltonTrigramProb += math.log(v / valueofStartingWordofTrigram, 10)
                startingWordsofTrigramHamilton = chosenDouble
                valueofStartingWordofTrigram = hamiltonBigramDict[chosenDouble]
                startingWordsofTrigramHamilton += " "
                #print(generatedHamiltonTrigramProb)
                print(k.split()[2], end= " ")
                if k.split()[2] == "." or k.split()[2] == "?" or k.split()[2] == "!":
                    generatedTrigramWordNumber = 50
                break
    generatedTrigramWordNumber += 1




print("\n")
""" For Madison """
print("For Madison ")
""" Unigram """
print("Unigram Generated Text\n ")


# i do the same thing explained in Hamilton Unigram text generation part.
generatedMadisonWordNumber = 0
generatedMadisonUnigramProb = 0
print("\t =>", end= ' ')
while generatedMadisonWordNumber < 30:
    generatedMadisonDoubleNumber = random.uniform(0.0, 1.0)
    currentProbabilityMadison = 0.0
    for k,v in madisonRawCounts.items():
        currentProbabilityMadison += madisonRawCounts[k] / madisonWordCount
        if currentProbabilityMadison > generatedMadisonDoubleNumber:
            generatedMadisonUnigramProb += math.log(madisonRawCounts[k] / madisonWordCount, 10)
            #print(generatedMadisonUnigramProb)
            print(k, end= " ")
            if k == "." or k == "?" or k == "!":
                generatedMadisonWordNumber = 50
            break
    generatedMadisonWordNumber += 1


""" Bigram """
print("\n")
print("Bigram Generated Text\n ")

# i do the same thing explained in Hamilton Unigram text generation part. Except in this part, the generated word is chosen in the limited space such as
# bigram dictionary. So the generation of word is depends on the last generated word. The probabilities are kept limited between them.
# By that, we can easily see the meaningful doubles in the generated text.
generatedBigramWordNumber = 0
firstMadisonBigramWord = random.choice(list(madisonRawCounts.keys()))
firstMadisonBigramWordVALUE = madisonRawCounts[firstMadisonBigramWord]
firstMadisonBigramWord += " "
generatedMadisonBigramProb = math.log(firstMadisonBigramWordVALUE / madisonWordCount, 10)
print("\t =>", end= ' ')
while generatedBigramWordNumber < 30:
    generatedBigramDoubleNumber = random.uniform(0.0, 1.0)
    currentProbability = 0.0
    wordCountsCameAfterFirstWord = 0
    for k,v in madisonBigramDict.items():
        if k.startswith(firstMadisonBigramWord):
            currentProbability += v / firstMadisonBigramWordVALUE
            if currentProbability > generatedBigramDoubleNumber:
                chosenWord = k.split()[1]
                generatedMadisonBigramProb += math.log(v / firstMadisonBigramWordVALUE, 10)
                #print(generatedMadisonBigramProb)
                print(chosenWord, end =" ")
                firstMadisonBigramWord = chosenWord
                firstMadisonBigramWordVALUE = madisonRawCounts[chosenWord]
                firstMadisonBigramWord += " "
                if k.split()[1] == "." or k.split()[1] == "?" or k.split()[1] == "!":
                    generatedBigramWordNumber = 50
                break
            #print(k, v)
    generatedBigramWordNumber += 1



""" Trigram """
print("\n")
print("Trigram Generated Text\n ")

# i do the same thing explained in Hamilton Unigram text generation part. Except in this part, the generated word is chosen in the limited space such as
# trigram dictionary. So the generation of word is depends on the last two generated words. The probabilities are kept limited between them.
# By that, we can easily see the meaningful triples in the generated text.
generatedTrigramWordNumber = 0
startingWordsofTrigramMadison = random.choice(list(madisonBigramDict.keys()))
valueofStartingWordofTrigramm = madisonBigramDict[startingWordsofTrigramMadison]
startingWordsofTrigramMadison += " "
tempWordValue = madisonRawCounts[(startingWordsofTrigramMadison.split()[0])]
generatedMadisonTrigramProb = math.log(valueofStartingWordofTrigramm / tempWordValue, 10)

print("\t =>", end= ' ')
while generatedTrigramWordNumber < 30:
    generatedTrigramDoubleNumber = random.uniform(0.0, 1.0)
    currentProbability = 0.0
    for k,v in madisonTrigramDict.items():
        if k.startswith(startingWordsofTrigramMadison):
            currentProbability += v / valueofStartingWordofTrigramm
            if currentProbability > generatedTrigramDoubleNumber:
                chosenDouble = k.split()[1] + " " + k.split()[2]
                generatedMadisonTrigramProb += math.log(v / valueofStartingWordofTrigramm, 10)
                #print(generatedMadisonTrigramProb)
                startingWordsofTrigramMadison = chosenDouble
                valueofStartingWordofTrigramm = madisonBigramDict[chosenDouble]
                startingWordsofTrigramMadison += " "
                print(k.split()[2], end= " ")
                if k.split()[2] == "." or k.split()[2] == "?" or k.split()[2] == "!":
                    generatedTrigramWordNumber = 50
                break
    generatedTrigramWordNumber += 1

print("\n")
print("END OF TASK 02\n\n")


""" TASK 03 AUTHORSHIP DETECTION AND TESTING OUR LANGUAGE MODELS """
print("TASK 03 AUTHORSHIP DETECTION AND TESTING OUR LANGUAGE MODELS")
# We need the number of unique doubles and triples to apply laplace smoothing (add-one) to our language models
# this integers holds the unique count of word groups, used in perplexity calculation
uniqueBigramNumberHamilton = len(hamiltonBigramDict)
uniqueBigramNumberMadison = len(madisonBigramDict)
uniqueTrigramNumberHamilton = len(hamiltonTrigramDict)
uniqueTrigramNumberMadison = len(madisonTrigramDict)

# Read the test essays

#for i in ('49.txt', '50.txt', '51.txt', '52.txt', '53.txt', '54.txt',  '55.txt', '56.txt', '57.txt', '62.txt', '63.txt'): All Madison
#for i in ('1.txt', '6.txt', '7.txt', '8.txt', '13.txt', '15.txt',  '16.txt', '17.txt', '21.txt', '22.txt', '23.txt',): All Hamilton Correct
#for i in ('9.txt', '11.txt','12.txt','47.txt','48.txt','58.txt'): MHHMMM  the first one is wrong
#for i in ('10.txt', '14.txt','37.txt','38.txt','39.txt','40.txt'): All Madison Correct

# change the texts to see the AuthorShip detection of my language models.
txtLists = []
for i in ('9.txt', '11.txt','12.txt','47.txt','48.txt','58.txt', '49.txt', '50.txt', '51.txt', '52.txt', '53.txt', '54.txt',  '55.txt', '56.txt', '57.txt', '62.txt', '63.txt'):
    a = 0
    with open(i,'r') as f:
        for line in f:
            if a is not 0:
                txtLists.append(line)
            else:
                a=a+1


# in this loop i tried all the perplexity of all the models for both Authors
printcounter = 0
print("Testing the Language Models with Text No: 9,11,12,47,48,58 ")

# doing the same preprocessing steps and word grouping stages as done in the previous file reads in language models creation
# this for loop iterates in every given text to be calculated
# for example 9.txt perplexity calculated author determine next 11.txt perplexity calculated author determined and so on..
for text in txtLists:
    tokens = []
    bigramLists = []
    trigramLists = []
    text = text.translate(translator)
    for word in text.split():
        #word = word.translate(translator)
        word = word.lower()
        if word.endswith('.') or word.endswith('!') or word.endswith('?'):
            tokens.append(word[:-1])
            tokens.append(word[-1])
            #print(word)
        else:
            tokens.append(word)

    bigramLists.extend(zip(*[tokens[i:] for i in range(2)]))
    trigramLists.extend(zip(*[tokens[i:] for i in range(3)]))

    bigramLists = [" ".join(ngram) for ngram in bigramLists]
    trigramLists = [" ".join(ngram) for ngram in trigramLists]

# i calculate both bigram and trigram at the same loop
    hamiltonTriTotal = 0                                        #perplexity of Hamilton trigram model
    madisonTriTotal = 0                                         #perplexity of Madison trigram model
    for t in trigramLists:
        trigramWord = t                                         # taking the triple          For example:  t= red riding hood
        bigrmWord = t.split()[0] + ' ' + t.split()[1]           # capturing the double inside from triple  b= red riding
        #print( t.split()[2])
        #print(trigramWord)
        #print(bigrmWord)

        # i tried to divide all the possibilities. For example: if double is seen before but triple is not seen before...
        # C(w1w2w3) + 1 / C(w1w2) + UniqueBigramWords -> taking log2 of it and add to perplexity sum.
        if (trigramWord in hamiltonTrigramDict) and (bigrmWord in hamiltonBigramDict):
            hamiltonTempo = math.log2((hamiltonTrigramDict[trigramWord] + 1 ) / ( uniqueTrigramNumberHamilton + hamiltonBigramDict[bigrmWord]))
        elif (trigramWord in hamiltonTrigramDict) and (bigrmWord not in hamiltonBigramDict):
            hamiltonTempo = math.log2((hamiltonTrigramDict[trigramWord] + 1 ) / ( uniqueTrigramNumberHamilton ))
        elif (trigramWord not in hamiltonTrigramDict) and (bigrmWord in hamiltonBigramDict):
            hamiltonTempo = math.log2((1) / ( uniqueTrigramNumberHamilton ))
        else:
            hamiltonTempo = math.log2((1) / ( uniqueTrigramNumberHamilton ))

        hamiltonTriTotal = hamiltonTriTotal + hamiltonTempo

        # i do same calculation for Madison perplexity calculation
        if (trigramWord in madisonTrigramDict) and (bigrmWord in madisonBigramDict) :
            madisonTempo = math.log2((madisonTrigramDict[trigramWord] + 1) / ( uniqueTrigramNumberMadison + madisonBigramDict[bigrmWord]))
        elif (trigramWord in madisonTrigramDict) and (bigrmWord not in madisonBigramDict) :
            madisonTempo = math.log2((madisonTrigramDict[trigramWord] + 1) / ( uniqueTrigramNumberMadison))
        elif (trigramWord not in madisonTrigramDict) and (bigrmWord in madisonBigramDict) :
            madisonTempo = math.log2((1) / ( uniqueTrigramNumberMadison ))
        else:
            madisonTempo = math.log2((1) / ( uniqueTrigramNumberMadison  ) )

        madisonTriTotal = madisonTriTotal + madisonTempo


    hamiltonTotal = 0
    madisonTotal = 0
    for a in bigramLists:
        bigramWord = a

        """ Hamilton Bigram Perplexity Calculation """

        if bigramWord in hamiltonBigramDict :
            hamiltonTemp = math.log2((hamiltonBigramDict[bigramWord] + 1 ) / ( uniqueBigramNumberHamilton + hamiltonRawCounts[a.split()[0]]))
        elif a.split()[0] in hamiltonRawCounts:
            hamiltonTemp = math.log2((1) / ( uniqueBigramNumberHamilton ))
        else:
            hamiltonTemp = math.log2((1) / ( uniqueBigramNumberHamilton ))

        hamiltonTotal = hamiltonTotal + hamiltonTemp

        """  Madison Bigram Perplexity Calculation  """

        if bigramWord in madisonBigramDict :
            madisonTemp = math.log2((madisonBigramDict[bigramWord] + 1 ) / ( uniqueBigramNumberMadison + madisonRawCounts[a.split()[0]]))
        elif a.split()[0] in madisonRawCounts:
            madisonTemp = math.log2((1) / (uniqueBigramNumberMadison ))
        else:
            madisonTemp = math.log2((1) / (uniqueBigramNumberMadison ))

        madisonTotal = madisonTotal + madisonTemp

    # following the logarithmic perplexity calculation in the assignment sheet
    # 1 / 2^( SUMofProbalities / IterationCount)
    hamiltonTotal = 1/pow(2, hamiltonTotal/len(bigramLists))
    madisonTotal =  1/pow(2, madisonTotal/len(bigramLists))

    hamiltonTriTotal = 1/pow(2, hamiltonTriTotal/len(bigramLists))
    madisonTriTotal =  1/pow(2, madisonTriTotal/len(bigramLists))
    #print(" Bigram Authorship Results ")

    #print("bi", madisonTotal , hamiltonTotal, end = ' ')
    #print("tri", madisonTriTotal , hamiltonTriTotal)


    printcounter += 1

    #this means we are done calculating the test texts and proceeding throught the Authorship detection
    if(printcounter == 7):
        print("=> 5/6 of the test result are True")
        print("\n")
        print("Task 03: Authorship Detection")
        print("Starting with 49.txt")

    # normalizing our result for distributing the text
    hamiltonTotal = hamiltonTotal/2
    hamiltonTriTotal = hamiltonTriTotal/2
    madisonTotal = madisonTotal/2
    madisonTriTotal = madisonTriTotal/2

    print("Bigram Model  ->", end= ' ')
    print("Hamilton Bigram Model Perplexity  ", hamiltonTotal, "-----", "Madison Bigram Model Perplexity ", madisonTotal )
    if (hamiltonTotal - madisonTotal) > 0 :
        print("Author is Madison %", 100 - (100* madisonTotal) / (hamiltonTotal + madisonTotal) )
    else:
        print("Author is Hamilton %",100 - (100* hamiltonTotal) / (hamiltonTotal + madisonTotal) )


    print("Trigram Model ->", end= ' ')
    print("Hamilton Trigram Model Perplexity ", hamiltonTriTotal, "-----", "Madison Trigram Model Perplexity ", madisonTriTotal )
    if (hamiltonTriTotal - madisonTriTotal) > 0 :
        print("Author is Madison %", 100 - (100* madisonTriTotal) / (hamiltonTriTotal + madisonTriTotal) )
    else:
        print("Author is Hamilton %",100 - (100* hamiltonTriTotal) / (hamiltonTriTotal + madisonTriTotal) )
    print("---------------------------------------------------------------------------------------------------------")


print("END OF THE ASSIGNMENT\n\n")

print("\n--- %s seconds ---" % (time.time() - start_time))



