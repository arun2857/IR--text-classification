Text Classification using Decision Tree, KNN, SVM, and Naive Bayes.



Course details: 21CSE377T
Information Retrieval CLA-2 Task 2

-->Name: ARUN SIVAPRASATH.A
Reg No: RA2411026050216
-->Name: Sanjith.s
Reg No: RA2411026050214



Problem Statement:

Classify newsgroup posts into 5 categories. Evaluate the performance of the 4 classifiers.



Dataset:

The dataset used is from the 20 Newsgroups. 5 classes were taken: comp.graphics, rec.autos, sci.med, soc.religion.christian, and talk.politics.guns. There were 4700 documents post-processing. Out of which 3760 were used for training, and 940 for testing. Class distribution was almost equal with an imbalance ratio of 1.10.

Pre-processing:


Convert to lowercase, Tokenize, Remove stop words, and Porter Stemming.

Feature Extraction:

TF-IDF (with 5000 terms) extracted only from the training data. Using scikit-learn pipeline.

Train Test Split:

Stratified train-test split of 80-20 with a fixed random seed of 42.

Models:

Decision Tree, KNN (k=5), Cosine, Linear SVM, Multinomial Naive Bayes.

Evaluation Metrics:

Accuracy, Macro Precision-Recall-f1 score, Confusion Matrix, and 5-fold cross-validation on training data.



Results:

The evaluation metrics for the test data are as follows:

Model Accuracy F1-score

Decision Tree 0.7766 0.7764

KNN (k=5) 0.8904 0.8908

SVM (Linear) 0.9128 0.9125

Naive Bayes 0.9202 0.9202



The best performing model was Naive Bayes. SVM was very close and only 7 documents less than Naive Bayes. The Decision Tree performed the worst.



Experiments:

Varying K for KNN. Increasing K from 1 to 25 increased the F1-score, but after a certain point it plateaued. It peaked at 0.899 with an increase of 0.8605.

Varying Feature Extraction Method. TF-IDF performed better than CountVectorizer by around 6 points for the KNN and SVM models. Bigrams reduced slightly.



Files:

text_classification_battle.py: Complete Code.

dataset_raw.csv and dataset_preprocessed.zip: Dataset and pre-processed data.

.csv and .png: Tables and graphs.



To Run:

Install scikit-learn pandas matplotlib nltk and run the code using python text_classification_battle.py. Results will be stored in results folder.
