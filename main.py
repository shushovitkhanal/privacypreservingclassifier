from Classifier import Classifier

if __name__ == '__main__':
    spamFolder1 = 'C:/Users/asus/Desktop/SpamData/enron1 (2)/enron1/spam'
    spamFolder2 = 'C:/Users/asus/Desktop/SpamData/enron2 (1)/enron2/spam'

    hamFolder1 = 'C:/Users/asus/Desktop/SpamData/enron1 (2)/enron1/ham'
    hamFolder2 = 'C:/Users/asus/Desktop/SpamData/enron2 (1)/enron2/ham'

    classifier = Classifier(spamFolder2, hamFolder2)
    classifier.run()

