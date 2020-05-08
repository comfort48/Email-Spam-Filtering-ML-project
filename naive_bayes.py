import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import time

class NaiveBayes:

	

	def load_csv_data(self, fileName):                              # Function to Load the dataset
		
		dataset = pandas.read_csv(fileName, header = None)
		
		return dataset

	def extractDataAttributes(self, uci_dataset):                   # Function to Load the attributes and label
		
		datasetAttributes = uci_dataset.iloc[:, :-1].values
		datasetOutputLabels = uci_dataset.iloc[:, 57].values

		return datasetAttributes, datasetOutputLabels

	def splitDataForClassification(self, datasetAttributes, datasetOutputLabels, testSplitRatio):        # Function to split the data into training data set and testing data set
		
		trainingData, testingData, trainingDataLabels, testingDataLabels = train_test_split(datasetAttributes, datasetOutputLabels, test_size = testSplitRatio, random_state = 1)

		return trainingData, testingData, trainingDataLabels, testingDataLabels

	def transformDataUsingStandardScaler(self, trainingData, testingData):                              # Function to transform and scale the traning data

		scaler = StandardScaler()
		scaler.fit(trainingData)

		trainingData = scaler.transform(trainingData)
		testingData = scaler.transform(testingData)

		return trainingData, testingData

	def fitDataToGaussianNBClassifier(self, trainingData, trainingDataLabels):                         # Function to fit the traning data into Gaussian Naive Bayes classifier
		
		gaussianNBClassifier = GaussianNB()
		gaussianNBClassifier.fit(trainingData, trainingDataLabels)

		return gaussianNBClassifier

	def predictOnGaussianNBClassifier(self, gaussianNBClassifier, testingData):                       # Function to test the classifier on Testing data set

		predictedDataLabels = gaussianNBClassifier.predict(testingData)

		return predictedDataLabels

	def getConfusionMatrix(self, testingDataLabels, predictedDataLabels):                             # Function to get the confusion matrix (constituting - TP, FP, FN, TN) of the classifier 

		confusionMatrix = confusion_matrix(testingDataLabels, predictedDataLabels)

		return confusionMatrix

	def getClassificationReport(self, testingDataLabels, predictedDataLabels):                       # Function to get the classification report metrics like - precision ,recall, accuracy, F1-score

		classificationReport = classification_report(testingDataLabels, predictedDataLabels)

		return classificationReport

	def getAccuracy(self, testingDataLabels, predictedDataLabels):                                   # Function to print the classifier's accuracy

		accuracy = accuracy_score(testingDataLabels, predictedDataLabels) * 100

		return accuracy


def main():

	program_start_time = time.time()

	naive_bayes = NaiveBayes()                                                             # Creating a Naive Bayes Classifier

	uci_dataset = naive_bayes.load_csv_data('uci_spambase.csv')                            # Load data

	datasetAttributes, datasetOutputLabels = naive_bayes.extractDataAttributes(uci_dataset)  # split data 

	testSplitRatio = 0.30																	 # Train:Test ratio

	# Split the data as traning and testing splits 

	trainingData, testingData, trainingDataLabels, testingDataLabels = naive_bayes.splitDataForClassification(datasetAttributes, datasetOutputLabels, testSplitRatio)  

	trainingData, testingData = naive_bayes.transformDataUsingStandardScaler(trainingData, testingData)        # Transform and scale the traning data set

	gaussianNBClassifier = naive_bayes.fitDataToGaussianNBClassifier(trainingData, trainingDataLabels)         # Fit the training data in the classifier

	predictedDataLabels = naive_bayes.predictOnGaussianNBClassifier(gaussianNBClassifier, testingData)         # Predict on the classifier using testing data split

	confusionMatrix = naive_bayes.getConfusionMatrix(testingDataLabels, predictedDataLabels)                   # Confusion Matrix - TP, TN, FP, FN

	classificationReport = naive_bayes.getClassificationReport(testingDataLabels, predictedDataLabels)         # Classification report - Precision, Recall, Accuracy and F1-score

	accuracy = naive_bayes.getAccuracy(testingDataLabels, predictedDataLabels)                                 # Print the accuracy

	totalPredictedDataNumber = confusionMatrix[0][0] + confusionMatrix[0][1] + confusionMatrix[1][0] + confusionMatrix[1][1]

	# Printing the Metrics of the classifier

	print("\nPercentage of True Positives (TP) for the predicted data = " + str(round(round(confusionMatrix[0][0] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Percentage of False Negatives (FN) for the predicted data = " + str(round(round(confusionMatrix[0][1] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Percentage of False Positives (FP) for the predicted data = " + str(round(round(confusionMatrix[1][0] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Percentage of True Negatives (TN) for the predicted data = " + str(round(round(confusionMatrix[1][1] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Classification Report of the model: \n\n" + classificationReport)
	
	print("Accuracy of the Naive Bayes classifier on UCI Email Spam Base dataset is: " + str(round(accuracy, 2)) + '%.\n\n')

	print("**** %s seconds ****" % (time.time() - program_start_time))


if __name__ == "__main__":
	main()