from svmutil import *
import re #Regex Lib
import csv # CSV Lib

# Useless Words Array
uselessWords = []
featureVector = []
feelingTweets = []
timeTweets = []
weatherTweets = []

# Format tweets
def formatTweet(tweet):
    # Lower Case
    tweet = tweet.lower()
    # Remove URLs (Don't need them)
    tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+))','',tweet)
    # Remove usernames (Don't need them)
    tweet = re.sub('@[^\s]+','',tweet)
    # Remove extra white spaces
    tweet = re.sub('[\s]+',' ', tweet)
    #Remove hashtag and keep the word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet
# End formatTweet

# Remove repeated chars and replace with single char
# Ex. "heeeeeey" -> "hey"
def replaceRepeatedChars(word):
    stringRepeat = re.compile(r"(.)\1{1,}", re.DOTALL)
    return stringRepeat.sub(r"\1\1", word)
# End replaceRepeatedChars

# Get useless words into an array
def getUselessWords():
    # Open Useless Words File
    uWFile = open('uselessWords.txt','r')
    line = uWFile.readline()
    while line:
        uWord = line.strip()
        global uselessWords
        uselessWords.append(uWord)
        line = uWFile.readline()
    #End while
    uWFile.close()
    #return uselessWords
# End getUselessWords

# Remove punctuation chars and Numbers in a word
def removePunctuationAndNumbers(word):
    return re.sub(r'\W+|\d+', '',word)
# End remove Punctuation

def getEvidence(tweet):
    evidences = []
    # Split tweet into an array of words
    tweet = formatTweet(tweet)
    tweetWords = tweet.split()
    for word in tweetWords:
        # Fix repeated chars
        word = replaceRepeatedChars(word)
        # Remove punctuation and numbers
        word = removePunctuationAndNumbers(word)
        # Check if it's useless
        if (word in uselessWords or len(word) < 3):
            continue
        # If it's not a useless word, add to evidences
        else:
            evidences.append(word)
        # End if
    # End for
    # When all words are processed, return evidences
    return evidences
# End getEvidence

def getFeeling(s1,s2,s3,s4,s5):
    maximum = max(s1,s2,s3,s4,s5)
    if (s1 == maximum):
        return 'Cannot tell'
    elif (s2 == maximum):
        return 'Negative'
    elif (s3 == maximum):
        return 'Neutral'
    elif (s4 == maximum):
        return 'Positive'
    elif (s5 == maximum):
        return 'Not related to weather'
# End get feeling

def getTime(w1,w2,w3,w4):
    maximum = max(w1,w2,w3,w4)
    if (w1 == maximum):
        return 'Same day'
    elif (w2 == maximum):
        return 'Future'
    elif (w3 == maximum):
        return 'Cannot Tell'
    elif (w4 == maximum):
        return 'Past'
# End get time

def getWeather(k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15):
    maximum = max(k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15)
    if (k1 == maximum):
        return 'Clouds'
    elif (k2 == maximum):
        return 'Cold'
    elif (k3 == maximum):
        return 'Dry'
    elif (k4 == maximum):
        return 'Hot'
    elif (k5 == maximum):
        return 'Humid'
    elif (k6 == maximum):
        return 'Hurricane'
    elif (k7 == maximum):
        return 'Cannot Tell'
    elif (k8 == maximum):
        return 'Ice'
    elif (k9 == maximum):
        return 'Other'
    elif (k10 == maximum):
        return 'Rain'
    elif (k11 == maximum):
        return 'Snow'
    elif (k12 == maximum):
        return 'Storms'
    elif (k13 == maximum):
        return 'Sun'
    elif (k14 == maximum):
        return 'Tornado'
    elif (k15 == maximum):
        return 'Wind'
# End get weather

def getFeatureVector():
    getUselessWords()
    with open('train.csv', 'rb') as trainFile:
        trainingTweets = csv.reader(trainFile, delimiter=',', quotechar='|')
        for row in trainingTweets:
            # Get evidences from tweet and append to featureVector
            evidence = getEvidence(row[1])
            for e in evidence:
                global featureVector
                featureVector.append(e)
            # Get Feeling 4-27
            feeling = getFeeling(row[4], row[5], row[6], row[7], row[8])
            # Get Time
            time = getTime(row[9], row[10], row[11], row[12])
            # Get Weather
            weather = getWeather(row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21],
                                     row[22], row[23], row[24], row[25], row[26], row[27], )
            # Append tweet to feeling tweets
            feelingTweets.append((evidence, feeling))
            # Append tweet to time tweets
            timeTweets.append((evidence, time))
            # Append tweet to weather tweets
            weatherTweets.append((evidence, weather))
            # End each word in evidences
            # End row in tweets file
    # End get tweets from training file
    # Remove duplicates from feature vector
    featureVector = sorted(list(set(featureVector)))

def getFeaturesFromEvidence(evidence):
    map = {}
    features = []
    # Initialize empty map
    for w in featureVector:
        map[w] = 0

    for word in evidence:
        if word in map:
            map[word] = (1)
    values = map.values()
    features.append(values)
        #features['contains(%s)' % word] = (word in words)
    return features
# End getFeaturesFromTweet

def getFeelingIndex(opinion):
    if (opinion == 'Cannot Tell'):
        label = 0
    elif (opinion == 'Negative'):
        label = 1
    elif (opinion == 'Neutral'):
        label = 2
    elif (opinion == 'Positive'):
        label = 3
    elif (opinion == 'Not related to weather'):
        label = 4
    return label
#End getFeelingIndex

def getTimeIndex(opinion):
    if (opinion == 'Same day'):
        label = 0
    elif (opinion == 'Future'):
        label = 1
    elif (opinion == 'Cannot Tell'):
        label = 2
    elif (opinion == 'Past'):
        label = 3
    return label
    # End getTimeIndex
#End getTimeIndex

def getWeatherIndex(opinion):
    if (opinion == 'Clouds'):
        return 0
    elif (opinion == 'Cold'):
        return 1
    elif (opinion == 'Dry'):
        return 2
    elif (opinion == 'Hot'):
        return 3
    elif (opinion == 'Humid'):
        return 4
    elif (opinion == 'Hurricane'):
        return 5
    elif (opinion == 'Cannot Tell'):
        return 6
    elif (opinion == 'Ice'):
        return 7
    elif (opinion == 'Other'):
        return 8
    elif (opinion == 'Rain'):
        return 9
    elif (opinion == 'Snow'):
        return 10
    elif (opinion == 'Storms'):
        return 11
    elif (opinion == 'Sun'):
        return 12
    elif (opinion == 'Tornado'):
        return 13
    elif (opinion == 'Wind'):
        return 14
#End getWeatherIndex

def getSVMFeatureVectorAndLabels(tweets,type):
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        #Initialize empty map
        for w in featureVector:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]
        #Fill the map
        for word in tweet_words:
            #set map[word] to 1 if word exists
            if word in map:
                map[word] = 1
        #end for loop
        values = map.values()
        feature_vector.append(values)
        if(type == 0):
            label = getFeelingIndex(tweet_opinion)
        elif(type == 1):
            label = getTimeIndex(tweet_opinion)
        elif(type == 2):
            label = getWeatherIndex(tweet_opinion)
        labels.append(label)
    #return the list of feature_vector and labels
    return {'feature_vector' : feature_vector, 'labels': labels}
#end

def trainTimeClass():
    # Train the classifier
    result = getSVMFeatureVectorAndLabels(timeTweets, 1)
    problem = svm_problem(result['labels'], result['feature_vector'])
    # '-q' option suppress console output
    param = svm_parameter('-q')
    param.kernel_type = LINEAR
    classifier = svm_train(problem, param)
    svm_save_model('time_svm.model', classifier)
# End trainTimeClass

def trainFeelingClass():
    #Train the classifier
    result = getSVMFeatureVectorAndLabels(feelingTweets, 0)
    problem = svm_problem(result['labels'], result['feature_vector'])
    # '-q' option suppress console output
    param = svm_parameter('-q')
    param.kernel_type = LINEAR
    classifier = svm_train(problem, param)
    svm_save_model('feeling_svm.model', classifier)
#End

def trainWeatherClass():
    #Train the classifier
    result = getSVMFeatureVectorAndLabels(weatherTweets, 2)
    problem = svm_problem(result['labels'], result['feature_vector'])
    # '-q' option suppress console output
    param = svm_parameter('-q')
    param.kernel_type = LINEAR
    classifier = svm_train(problem, param)
    svm_save_model('weather_svm.model', classifier)
#End

def getTimeWord(index):
    if (index == 0):
        return 'Same day'
    elif (index == 1):
        return 'Future'
    elif (index == 2):
        return 'Cannot Tell'
    elif (index == 3):
        return 'Past'
#End getTimeWord

def getFeelingWord(index):
    if (index == 0):
        return 'Cannot tell'
    elif (index == 1):
        return 'Negative'
    elif (index == 2):
        return 'Neutral'
    elif (index == 3):
        return 'Positive'
    elif (index == 4):
        return 'Not related to weather'
# End getFeelingWord

def getWeatherWord(index):
    if (index == 0):
        return 'Clouds'
    elif (index == 1):
        return 'Cold'
    elif (index == 2):
        return 'Dry'
    elif (index == 3):
        return 'Hot'
    elif (index == 4):
        return 'Humid'
    elif (index == 5):
        return 'Hurricane'
    elif (index == 6):
        return 'Cannot Tell'
    elif (index == 7):
        return 'Ice'
    elif (index == 8):
        return 'Other'
    elif (index == 9):
        return 'Rain'
    elif (index == 10):
        return 'Snow'
    elif (index == 11):
        return 'Storms'
    elif (index == 12):
        return 'Sun'
    elif (index == 13):
        return 'Tornado'
    elif (index == 14):
        return 'Wind'
# End getWeatherWord

def getPrediction(svm,featVector):
    p_labels, p_accs, p_vals = svm_predict([0] * len(featVector), featVector, svm)
    return int(p_labels[0])
# End getPrediction
def loadTweets(file):
    tweets = []
    with open(file) as textFile:
        for line in textFile:
            tweets.append(line.rstrip('\n'))
        textFile.close()
    return tweets
# Main

getFeatureVector()
#Load SVM
feelingSVM = svm_load_model('feeling_svm.model')
timeSVM = svm_load_model('time_svm.model')
weatherSVM = svm_load_model('weather_svm.model')

#Test the classifier
# Define tweet
# Define tweet
tweets = loadTweets('testTweets.txt')
for tweet in tweets:
    # Format tweet
    formattedTweet = formatTweet(tweet)
    # Get evidences from tweet
    evidence = getEvidence(formattedTweet)
    # Get features from tweet
    features = getFeaturesFromEvidence(evidence)
    # Classify tweet
    print tweet
    print "Feeling:", getFeelingWord(getPrediction(feelingSVM, features))
    print "Time:", getTimeWord(getPrediction(timeSVM, features))
    print "Weather:", getWeatherWord(getPrediction(weatherSVM, features))
    print '------------------------'