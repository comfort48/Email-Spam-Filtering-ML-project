import numpy
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time

class KNN:

	def load_csv_data(self, fileName):                                 # Function to Load the dataset
		
		
		dataset = pandas.read_csv(fileName, header = None)
		
		return dataset

	def extractDataAttributes(self, uci_dataset):					   # Function to Load the attributes and label
		
		datasetAttributes = uci_dataset.iloc[:, :-1].values
		datasetOutputLabels = uci_dataset.iloc[:, 57].values

		return datasetAttributes, datasetOutputLabels

	def splitDataForClassification(self, datasetAttributes, datasetOutputLabels, testSplitRatio):              # Function to split the data into training data set and testing data set
		
		trainingData, testingData, trainingDataLabels, testingDataLabels = train_test_split(datasetAttributes, datasetOutputLabels, test_size = testSplitRatio, random_state = 100)

		return trainingData, testingData, trainingDataLabels, testingDataLabels

	def transformDataUsingStandardScaler(self, trainingData, testingData):                                      # Function to transform and scale the traning data

		scaler = StandardScaler()
		scaler.fit(trainingData)

		trainingData = scaler.transform(trainingData)
		testingData = scaler.transform(testingData)

		return trainingData, testingData

	def fitDataToKNNClassifier(self, trainingData, trainingDataLabels, kValue):                                # Function to fit the traning data into KNN classifier
		
		KNNClassifier = KNeighborsClassifier(n_neighbors = kValue)
		KNNClassifier.fit(trainingData, trainingDataLabels)

		return KNNClassifier

	def predictOnKNNClassifier(self, KNNClassifier, testingData):                                              # Function to test the classifier on Testing data set

		predictedDataLabels = KNNClassifier.predict(testingData)

		return predictedDataLabels

	def getConfusionMatrix(self, testingDataLabels, predictedDataLabels):                                     # Function to get the confusion matrix (constituting - TP, FP, FN, TN) of the classifier 

		confusionMatrix = confusion_matrix(testingDataLabels, predictedDataLabels)

		return confusionMatrix

	def getClassificationReport(self, testingDataLabels, predictedDataLabels):                             # Function to get the classification report metrics like - precision ,recall, accuracy, F1-score

		classificationReport = classification_report(testingDataLabels, predictedDataLabels)

		return classificationReport

	def getAccuracy(self, testingDataLabels, predictedDataLabels):                                            # Function to print the classifier's accuracy

		accuracy = accuracy_score(testingDataLabels, predictedDataLabels) * 100

		return accuracy

	def printErrorAndAccuracyPlots(self, trainingData, trainingDataLabels, testingData, testingDataLabels):   # Function to find the Optimal value of K using Hyper Parameter Tuning
																											  # and printing the Accuracy and Mean Error Rate graphs
		error = []
		accuracy = []

		for index in range(1, 40):
		    
		    KNNClassifier = self.fitDataToKNNClassifier(trainingData, trainingDataLabels, index)
		    predictedDataLabels = self.predictOnKNNClassifier(KNNClassifier, testingData)
		    error.append((numpy.mean(predictedDataLabels != testingDataLabels))*100)
		    accuracy.append((numpy.mean(predictedDataLabels == testingDataLabels))*100)


		plt.figure(figsize=(30, 20))
		plt.plot(range(1, 40), error, color='blue', linestyle='solid', marker='D',
		         markerfacecolor='red', markersize=8)
		plt.title('K value vs Error Rate')
		plt.xlabel('K value')
		plt.ylabel('Error Rate')
		plt.show()


		plt.figure(figsize=(30, 20))
		plt.plot(range(1, 40), accuracy, color='blue', linestyle='solid', marker='D',
		         markerfacecolor='red', markersize=8)
		plt.title('K value vs Accuracy')
		plt.xlabel('K value')
		plt.ylabel('Accuracy')
		plt.show()




def main():

	program_start_time = time.time()

	knn = KNN()                                                  # Creating a KNN Classifier

	uci_dataset = knn.load_csv_data('uci_spambase.csv')          # Load data

	datasetAttributes, datasetOutputLabels = knn.extractDataAttributes(uci_dataset)    # split data 

	testSplitRatio = 0.30															   # Train:Test ratio

	# Split the data as traning and testing splits 

	trainingData, testingData, trainingDataLabels, testingDataLabels = knn.splitDataForClassification(datasetAttributes, datasetOutputLabels, testSplitRatio)

	trainingData, testingData = knn.transformDataUsingStandardScaler(trainingData, testingData)     # Transform and scale the traning data set

	kValue = 8
 
	KNNClassifier = knn.fitDataToKNNClassifier(trainingData, trainingDataLabels, kValue)            # Fit the training data in the KNN classifier
 
	predictedDataLabels = knn.predictOnKNNClassifier(KNNClassifier, testingData)                    # Predict on the classifier using testing data split

	confusionMatrix = knn.getConfusionMatrix(testingDataLabels, predictedDataLabels)                # Confusion Matrix - TP, TN, FP, FN

	classificationReport = knn.getClassificationReport(testingDataLabels, predictedDataLabels)      # Classification report - Precision, Recall, Accuracy and F1-score

	accuracy = knn.getAccuracy(testingDataLabels, predictedDataLabels)                              # Print the accuracy

	# Printing the Metrics of the classifier

	totalPredictedDataNumber = confusionMatrix[0][0] + confusionMatrix[0][1] + confusionMatrix[1][0] + confusionMatrix[1][1]

	print("\nPercentage of True Positives (TP) for the predicted data = " + str(round(round(confusionMatrix[0][0] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Percentage of False Negatives (FN) for the predicted data = " + str(round(round(confusionMatrix[0][1] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Percentage of False Positives (FP) for the predicted data = " + str(round(round(confusionMatrix[1][0] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Percentage of True Negatives (TN) for the predicted data = " + str(round(round(confusionMatrix[1][1] / totalPredictedDataNumber, 4) * 100, 2)) + '%.\n\n')

	print("Classification Report of the model: \n\n" + classificationReport)

	print("Accuracy of the K Nearest Neighbors classifier on UCI Email Spam Base dataset is: " + str(round(accuracy, 2)) + '%.\n\n')

	print("**** %s seconds ****" % (time.time() - program_start_time))

	# Printing the Accuracy vs K value and Mean Error Rate vs K value graphs 

	knn.printErrorAndAccuracyPlots(trainingData, trainingDataLabels, testingData, testingDataLabels);



if __name__ == "__main__":
	main()