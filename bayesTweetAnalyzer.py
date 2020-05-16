import re #Regex Lib
import csv # CSV Lib
import nltk # NLTK Bayes
import pickle #Save clasiffier
import datetime

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

def getFeaturesFromEvidence(evidence):
    words = set(evidence)
    features = {}
    for word in featureVector:
        features['contains(%s)' % word] = (word in words)
    return features
# End getFeaturesFromTweet

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

def trainClassifier():
    # Open Training tweets from CSV
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
            time = getTime(row[9],row[10],row[11],row[12])
            # Get Weather
            weather = getWeather(row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[21],row[22],row[23],row[24],row[25],row[26],row[27],)
            # Append tweet to feeling tweets
            feelingTweets.append((evidence, feeling))
            # Append tweet to time tweets
            timeTweets.append((evidence,time))
            # Append tweet to weather tweets
            weatherTweets.append((evidence,weather))
            # End each word in evidences
            # End row in tweets file
    # End get tweets from training file
    # Remove duplicates from feature vector
    featureVector = list(set(featureVector))
    #saveFeelingClassifier()
    #saveTimeClassifier()
    #print "Time Class Done",datetime.datetime.time(datetime.datetime.now())
    #saveWeatherClassifier()
    #print "Weather Class Done", datetime.datetime.time(datetime.datetime.now())
# End Train Classifier

def saveFeelingClassifier():
    # Define training set
    training_set = nltk.classify.util.apply_features(extract_features, feelingTweets)
    # Train the classifier and save it
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    # Save classifier
    f = open('feeling_classifier.pickle', 'wb')
    pickle.dump(NBClassifier, f)
    f.close()
# End save feeling classifier

def saveTimeClassifier():
    # Define training set
    training_set = nltk.classify.util.apply_features(extract_features, timeTweets)
    # Train the classifier and save it
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    # Save classifier
    f = open('time_classifier.pickle', 'wb')
    pickle.dump(NBClassifier, f)
    f.close()
# End save feeling classifier

def saveWeatherClassifier():
    # Define training set
    training_set = nltk.classify.util.apply_features(extract_features, weatherTweets)
    # Train the classifier and save it
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    # Save classifier
    f = open('weather_classifier.pickle', 'wb')
    pickle.dump(NBClassifier, f)
    f.close()
# End save weather classifier

def loadFeelingClassifier():
    f = open('feeling_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier
# End load feeling classifier

def loadTimeClassifier():
    f = open('time_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier
# End load time classifier

def loadWeatherClassifier():
    f = open('weather_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier
# End load weather classifier
def loadTweets(file):
    tweets = []
    with open(file) as textFile:
        for line in textFile:
            tweets.append(line.rstrip('\n'))
        textFile.close()
    return tweets

# Main
getUselessWords()
trainClassifier()

# Load the classifiers
timeClassifier = loadTimeClassifier()
feelingClassifier = loadFeelingClassifier()
weatherClassifier = loadWeatherClassifier()
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
    print "Time:",timeClassifier.classify(features)
    print "Feeling:",feelingClassifier.classify(features)
    print "Weather:",weatherClassifier.classify(features)
    print '------------------------'
