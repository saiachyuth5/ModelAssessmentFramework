# ModelAssessmentFramework

Model Assessment Framework for Deep Learning 
CSE 598 - Data Intensive Systems for Machine Learning 
Group 13 - Project Report 

Aditya Deepak Bhat 
ASU ID: 1222133796 
ASU Email: abhat31@asu.edu Arizona State University 
Sai Achyuth Vellineni ASU ID: 1219510708 
ASU Email: svelline@asu.edu Arizona State University 
Sriram Praneeth Turlapati ASU ID: 1223243892 
ASU Email: sturlapa@asu.edu Arizona State University 

Abstract—Evaluating deep learning models is a critical step in any deep learning production system. This helps assess the performance of a model based on the evaluation metrics defined. While evaluation metrics give insights on the model performance on a test/dev set, this is insufficient to test and assess the behavior of the deep learning model. In this project, we propose a deep learning model assessment framework that uses numerous pre train, during-train, post-train tests to not only evaluate the performance of the model, but also to check whether the learned logic of the model is consistent with the expected behavior. This would provide a more comprehensive method to assess the deep learning model before it is updated in the registry or deployed. 
Index Terms—deep learning systems, model assessment, be havioral testing, model versioning. 
I. INTRODUCTION 
Testing is a critical step in any software system which helps to ensure that the implemented software logic is consistent with the expected behavior. Here, the input to the test frame work is the data and the logic implemented and the output is the behavior which is compared with the expected behavior for that input. Numerous tests are grouped together into a test suite which is executed whenever there is a change to the implemented logic (codebase). The change is merged only if all the tests in the suite pass. This helps in early detection of bugs by ensuring correct implementation and also helps identify if the change breaks the expected behavior. 
While this straightforward approach to testing can work well to identify issues in traditional software systems, they may not suffice to assess a machine learning system. In contrast to traditional software systems, the input to machine learning systems is the training data that contains the desired behavior of the system and the output is the model which contains the learned logic. 
We are faced with the following challenges in testing and evaluating machine learning systems: 
• How do we ensure that the Learned Logic is consistent with the Desired Behavior? - This is not as straightfor ward as comparing the generated and expected output for a given input in the case of traditional software systems. 
• How do we assess if a particular change breaks the expected behavior of the machine learning system? 
II. RELATED WORK 
Traditionally, the primary approach to assess the perfor mance of machine learning models has been based on measur ing metrics like accuracy on a held-out dataset. However, these approaches are task-specific and don’t evaluate the behavior of the trained models. CheckList [1] introduces task-agnostic methods for evaluating behavioral testing of NLP models, which help extensively test machine learning models and identify early bugs. 
TensorFuzz [2] implements techniques based on coverage guided fuzzing (CFG) to help perform software tests on neural networks, which can help identify errors which occur on infrequent inputs to the model. It also introduces a coverage metric based on the radial neighborhoods of the model logits on the test set. 
Work done in [3], [5], [6] discusses the differences between testing in traditional software systems and machine learning systems. It uses multiple pre-train, post-train and behavioral tests to effectively test machine learning models like Decision Trees and Random Forests, which help evaluate the trained model before it can be added to the model registry. 
III. PROPOSED SOLUTION 
This project aims to implement a deep learning model assessment framework that can be used to assess a deep learning model after it is updated. The framework will use various tests to evaluate the deep learning model: 
• Pre-train tests - to ensure correct implementation and data (train, test, validation). 
• During-train tests - to capture and evaluate useful metrics during training. 
• Post-train tests - will be performed to help ensure that the model’s behavior is as expected. 
This would help to: 
• Identify issues early in deep learning model training. • Provide an effective method to evaluate the performance and behavior of the model before it can be deployed. Fig. 1 shows the proposed solution. For a new model, first pre-train tests are run to check for correct implementation. Next during train tests and evaluation metrics are captured. This is followed by post-train tests to test the behavior of the

Fig. 1. Proposed Model Assessment Framework. 

learned logic. If all the tests pass, the model is updated into the registry. 
IV. DATASETS AND EXPERIMENTAL SETUP 
We are utilizing three different datasets to assess system performance on different types of data in this project. • Titanic dataset: A classification dataset for estimating the probability of passenger survival. It is composed of a heterogeneous combination of numerical and category data. 
• MNIST dataset: A well-known handwritten digits image classification dataset that predicts the number displayed in the image. Each image is represented by 28x28 pixel values. This data set contains numbers ranging from 0 to 9. The training dataset has 60000 records, whereas the testing set contains 10,000. 
• Amazon Reviews for Sentiment Analysis: The Amazon review dataset is a text dataset containing millions of Amazon consumer reviews of products they’ve purchased on Amazon. The model’s text input consists of 1 to 5 sentences describing their product review. The dataset’s target object is a starred product review with a rating of 1 to 5. 
Experimental setup - We used TensorFlow 2.8 to train the deep learning models. All experiments and tests were run in Google Colab using GPU mode for execution. The tests are implemented using fixtures in pytest which provides a well known environment for testing in Python. 
A. Models Used for experimentation 
• Titanic Dataset - Simple 2-Layer MLP. 
• MNIST Dataset - CNN based architecture as shown in Fig. 2 
• Amazon Reviews Dataset - fully connected CNN and fully bidirectional LSTM models as shown in Figures 3 and 4. 
V. PRE-TRAIN TESTS 
A. Model Validation 
Model validation is a type of pre-train test using which the system can verify that all of the model’s inputs are perfectly 
aligned with the dataset. This will help ensure that executing the model on the given data will not result in an error. The tests under this validation are: 
• The system checks that the number of features in the train data for both the training model and the model’s prediction is the same. 
• The model will check whether the shape of the features and label are aligning with each other 
B. Data Validation 
In Data Validation, the system looks mainly for two types of leakages that can happen in a machine learning dataset: • Label leakage: Label leakage occurs when the model’s training data contains information about the data’s target labels, either directly or indirectly. It allows a model’s generalization error to be over-represented, greatly im proving its performance but rendering the model worth less for any practical applications. 
• Data leakage: When training data contains test data infor mation, data leakage occurs. This misleadingly increases the prediction accuracy. 
VI. DURING-TRAIN TESTS 
The purpose of performing multiple epochs is to improve accuracy over time. So, at the end of each training epoch, the system looks for an improvement in accuracy over the preceding iteration. If accuracy is not improving after multiple successive iterations, the model reverts to best performing iteration. For this, we perform the following operations: 
• Checkpointing: The system checks for the best perform ing iteration in terms of accuracy at the end of each iteration and checkpoints it. If future epochs do not improve accuracy, the system can backtrack this model. 
• Early Stopping: If the accuracy of the model is not improving over successive iterations, the training will be stopped early and the best performing checkpoint of the model will be restored. 
VII. POST-TRAIN TESTS 
A. Invariance Tests 
Invariance tests are tests where we introduce certain pertur bations to the original input data and check for consistency in
Fig. 2. Architecture of CNN model used in MNIST classification. 

Fig. 3. Architecture of the bidirectional LSTM sentiment analysis model for amazon reviews 
model prediction on the perturbed data. The expectation from this test is that if we perturb the data in such a way that we only modify the parts of the data which do no have a direct impact on the output, there should not be any significant decrease in prediction accuracy. This is closely related to the concept of data augmentation where we make make perturbations to input during training and preserve the labels. For instance, when using the sentiment analysis model which is a fully bidirectional LSTM, we did the following test: 
Original Sentence - “How to Not Die Alone is a must-read for millennials navigating any stage of their relationship. The advice is clear, researched-based, and actually easy to follow. It’s the perfect book for anyone who wants to up their dating game.” 
Perturbed Sentence - “LIFE SUCKS?: How To Make It Better is a must-read for millennials navigating any stage of their relationship. The advice is clear, researched-based, and actually easy to follow. It’s the perfect book for anyone who wants to up their dating game.” 
Here, we expect that a mere change in the title should not affect the prediction score and we observed that the model behaves as expected. The model predictions are shown in Table I. 
TABLE I 
MODEL PREDICTIONS (PROBABILITY) FOR INVARIANCE TEST - AMAZON FOOD REVIEWS DATASET


Original Sentence 
Perturbed Sentence
Sentiment 
0.996 
0.980




Fig. 4. Architecture of fully connected CNN sentiment analysis model for amazon reviews 
B. Directional Expectation Tests 
In directional expectation tests, we explore the other end of perturbations which have a direct impact on the output. The expectation is that with such a modification of the input data, there should be a drastic increase or decrease in the model’s prediction. These kind of tests serve as checks to identify cases of inconsistencies which simply cannot be found by just examining the performance of the model on a validation dataset. Below are few directional tests we performed: 
In the sentiment analysis model tested on the amazon reviews dataset - the insertion of a negation should change the label from positive to negative. For example: 
Original Sentence: “I highly recommend this book if you are even thinking of opening a restaurant of any concept!” Perturbed Sentence: “I wouldn’t recommend this book if you are thinking of opening a restaurant of any concept!” 
Table III shows the change in prediction probability for the above directional expectation test using the bidirectional LSTM model. 
TABLE II 
MODEL PREDICTIONS (PROBABILITY) FOR DIRECTIONAL EXPECTATION TESTS - AMAZON SENTIMENT ANALYSIS 


Original Sentence 
Perturbed Sentence
Sentiment 
0.95 
0.17



Similarly, for Titanic Dataset, below are few examples of directional expectations: 
• Changing the gender of the passengers from Female to Male should decrease the survival probability. 
• Changing the Passenger Class for 1 to 3 should decrease the survival probability. 
TABLE III 
MODEL PREDICTIONS (PROBABILITY) FOR DIRECTIONAL EXPECTATION TESTS - TITANIC DATASET 
Test Case 
Original Sentence 
Perturbed Sentence
Gender Change (Female to Male) 
2.3275745e-14 
2.1236768e-14
Class Change (1 to 3) 
2.3275745e-14 
2.2887474e-14



C. Minimum Functionality / Data Unit Tests 
TABLE IV 
RESULTS ON DATA UNIT TESTS ON THE AMAZON REVIEWS DATASET 
Model 
Overall test set 
Subset
fully-connected CNN 
91.3 
84.95
fully-connected bidirectional LSTM 
92.69 
87.79



Fig. 5. Distribution of number of words in each review of the amazon reviews dataset. 
Testing atomic components in the codebase and data unit tests allow us to quantify model performance for specific cases found in the data. This allows us to identify critical scenarios where wrong predictions lead to bad consequences and automate searching for such errors in future models. For example: In the case of sentiment analysis, we can evaluate the performance of the model on very short or long reviews in the amazon reviews dataset. Ideally we would want the model performance to be invariant of whether it is a short/long sentence.
Table IV shows the result of a data unit test on the Amazon Reviews dataset. We can observe that while the accuracy of the fully connected bidirectional-LSTM model is better than the fully-connected CNN, the performance of both the models decreases when we compare the accuracy on the overall test set with the same on a subset of very long or short sentences. As we can see from the table, there is a ≈6% dip in accuracy when the model was tested on the subset. 
Fig. 5 shows a plot of the distribution of number of words in each review of the dataset. We can observe that the distribution is skewed i.e. there are very few reviews that have very few or very high number of words in it. Data Unit Tests help us identify such bias or anomalies in the dataset. These tests can help identify data bias early in training, which can give an indication to add more data to improve the performance of the model in specific scenarios. 
VIII. ERROR ANALYSIS 
This is the step where we analyse the predictions and find reasoning for the wrong classifications or bad performance. This step has to be done manually using some commonly encountered patterns such as misclassified labels , outliers in the dataset , blurry images etc. We assume that the data has been augmented to avoid over fitting before we start checking for these patterns. Example - Images in the MNIST dataset can contain noise that could have been introduced due to poor handwriting, normalization, etc. making it very ambiguous to give a correct prediction of the class of the image. 
A. Frequently misclassified images in MNIST dataset Fig. 6. Error Analysis on MNIST - sample misclassified image 
Figures 6 and 7 show examples of bad data, both of these images were given as input to a CNN with an accuracy of 98% on the MNIST dataset. 
• The original label of Fig. 6 is ‘6’, but the image was misclassified by the network as 0. 
• Fig. 7 original label is ‘5’ but it was misclassified as ‘3’ with high probability. 
Fig. 7. Error Analysis on MNIST - sample misclassified image 
These are clear examples of bad data where the digits are not clearly legible and are ambiguous. We can handle these kinds of misclassifications by removing the incorrect/bad data from the dataset and re-train/test the network. 
IX. EVALUATION PLAN 
When a new model is presented, the evaluation plan for this system is as follows: 
• Start the pre-train tests: All of the pre-trained tests are performed on the new model. 
– Model validation tests are initially run on the model. – Data validation tests are run. 
– The model must pass all of the pre-train tests before the actual training on the data is performed. 
• Model training will be started and the model should pass all the during-train tests. 
– Model Checkpointing is done to checkpoint the best model after each iteration. 
– Early Stopping is done to stop training if the perfor mance of the model doesn’t improve after multiple successive epochs. 
– Evaluation Metrics are captured. 
– The model checkpoint with the best performance is used. 
• Post-train tests will be conducted after training only if all the previous checks/tests have passed. 
– Invariance Tests are checked to identify anomalies which we cannot find just by comparing the accuracy of the model. 
– Directional Tests are checked to find any inconsis tencies with the model. 
– Data Unit Tests are checked to find any bias in the model towards a particular set of data points. 
• Finally, error analysis is performed on the misclassified data and based on the evaluation, we modify the param eters of the model and begin the cycle again if required.

The model will be accepted only if the model passes all the above tests. 
X. CONCLUSION AND FUTURE WORK 
In this project we proposed a model assessment framework to evaluate and test a model before it is updated in the model registry. We explored three main types of tests: (1) Pre train tests to ensure correct implementation and data (train, test, validation), (2) During-train tests to capture and evaluate useful metrics during training, (3) Post-train tests to help ensure that the model’s behavior is as expected. This would help us to identify issues early in deep learning model training and would also provide an effective method to evaluate the performance and behavior of the model before it can be deployed. 
Some of the future work in this project would be: • More Generic implementations of tests - Extending the tests performed by the framework to make it more generic would help provide a robust testing and evaluation framework for new datasets and models. 
• Explore task agnostic behavioral tests - having be havioral tests that are task agnostic would make the framework more robust and easy to use on new datasets. 
• Test Suite Creation - We can create test suites based on the the tests/checks implemented for each dataset. These can be automatically launched to assess the behavior of a new model. 
• Integration with model versioning systems - The suites implemented can be integrated with model versioning systems, where these tests can be used to validate the model before it is updated in the model registry. 
REFERENCES 
[1] Ribeiro, Marco Tulio, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. ”Beyond accuracy: Behavioral testing of NLP models with CheckList.” arXiv preprint arXiv:2005.04118 (2020). 
[2] Odena, Augustus, Catherine Olsson, David Andersen, and Ian Good fellow. ”Tensorfuzz: Debugging neural networks with coverage-guided fuzzing.” In International Conference on Machine Learning, pp. 4901- 4911. PMLR, 2019. 
[3] Effective testing for machine learning systems. (https://www.jeremyjordan.me/testing-ml/) 
[4] How to Trust Your Deep Learning Code. (https://krokotsch.eu/posts/deep-learning-unit-tests/) 
[5] How to Test Machine Learning Code and Systems. (https://eugeneyan.com/writing/testing-ml/) 
[6] Amazon Reviews for Sentiment Analysis. (https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) [7] Titanic - Machine Learning from Disaster. (https://www.kaggle.com/competitions/titanic/data) 
[8] The MNIST database of handwritten digits. (http://yann.lecun.com/exdb/mnist/)
