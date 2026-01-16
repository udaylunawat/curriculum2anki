**Module 5: Feature Engineering, Productionization and deployment of ML models**

**Chapter 34:	Featurization and Feature Engineering**

**34.1	Introduction**

	Most important process in ML is feature engineering;

Data can be in the form of Text, categorical, numerical, time series, image, database tables, Graph-data, etc.

	Text data: BOW, TFIDF, AVGW2V, TFIDFW2V

	Categorical data: One-hot Encoding, response encoding

	Time series (ex: heart rate): transformed to frequency domain, etc.

	Image data (ex: face detection): transformed to arrays

**34.2	Moving Window for Time Series Data**

	Simplest featurization method for time series data;

	Example: ECG

	![][image177]

	Moving window takes a time split of a width of time;

Featurization techniques are decided based on damage; some standard features extracted from window of the time series are Mean, Median, Max and min, Max \- min, Standard deviation, quantile, number of local minima or maxima, mean crossings or zero crossings all of this for each window;

The features are problem specific;

1. Window width – hyper parameter  
2. Features useful: Generate dataset; data points correspond to different windows

**34.3	Fourier decomposition**

	A Method to represent time series data;

	![][image178]

	T \= Time taken for each oscillation

	Freq \= 1/T (Hz)

		Example: Heart rate: 60 – 100 BPM \= 1 – 1.6 BPS \= 1 – 1.6 Hz

	Phase: Difference in angle at which different waves start;

    We can have daily shopping patterns or weekly shopping patterns or annual patterns

Given a composite wave: It can be decomposed into multiple sine waves, different sine waves have different time period, and different peaks: this is transformed onto frequency domain; This process is called as Fourier transformation;

Useful when there is a repeating pattern;

Vectorization using important frequency and amplitudes;

	Accelerometer: electro-mechanical device used to measure acceleration forces. They use Piezo electric effect: where when the crystal is acted upon by an accelerative force an EMF is generated in the circuit.

**34.4	Deep Learning Features: LSTM**

For each problem: a specific set of features are designed prior to Deep Learning; A set of features designed for a problem did not work for another problem

Deep Learnt Features: Used for time series using LSTM

**34.5	Image histogram**

Color histogram: Each pixel has 3 color values; for each color plot a histogram (0 to 255\) and vectorize it; Different objects have distinct color attributes, sky and water come in blue, human faces have cream, wood brown, metal grey, etc.

Edge histogram: At the interface of color separation; region based edge orientation; 

**34.6	Key points: SIFT**

Scale Invariant Feature Transforms;

![][image179]

Detects key points or corners of object and creates a vector for each key point;

These key features are detected and extracted for predictions;

SIFT has scale invariance and small rotation invariance

**34.7	Deep Learning: CNN**

Time series data: LSTM

Images: CNN

**34.8	Relational data**

Data stored in multiple tables and connected through relations: 

	Ex: Shopping database

	Feature examples: Count, Budget

**34.9	Graph data**

	Example: Facebook: User Social Graph

	Task: Recommend new friends

	Features: Number of paths different nodes of the Graph, Number of mutual friends

**34.10	Indicator variables**

Converting a feature into indicator variables:

	Height: Numerical to indicator variable, example: binary: 1 if \> 150, 0 else;

	Country: If country \== India Or USA return 1 else 0

**34.11	Feature Binning**

Extension to indicator variable: Multi class encoding:

	Height: If H\<120 return 1 elif H\<150 return 2 elif H\<180 return 3 else return 4

Task: Predict gender given, height, weight, hair length, eye color

	return 1 if male and return 0 if female

	Table: Converting height numerical data into categorical using binning:

		Train a simple decision tree using height:

			If H \< 130cm yi \= 0 (thresholds determined using Information Gain)

			Thresholds found using DT 

Decision Tree Based Feature Binning

**34.12	Interaction Variables:**

	Example: 	1\. if h\<150 and w\<=60 kgs return 0: 2 way interaction

			2\. h \* w or w\*h2 : math operations 2 way interaction here

			3\. h \< 150, w\<65 and hL \> 75 return 0: 3 way interaction

	Given a task find interaction features:

		Using Decision Trees: Create new features using all leaf nodes;

**34.13	Mathematical transformations**

X is a feature: log(X), sin(X), X2, etc. can be used to transform the features;

Distribution of X can be inferred: Power law distributed can be logged;

**34.14**	**Model specific featurizations** 

Say f1 is power law distributed: when using Logistic Regression: transforming with log is beneficial;

3 features and a regression target: say we have f1 – f2 \+ 2 f3: Linear models are better; Decision Trees may not work here

If interaction of features are prevalent DT perform well;

With Bag of Words: Linear models tend to perform very well (Lr SVM and Logistic Regression);

**34.15	Feature Orthogonality**

The more different the features (Orthogonal relationship) are, the better the models will be;

If there are correlations (less orthogonality) among the features, the overall prediction performance of the model on the target will be poor;

Create new features with errors of a model; easily over-fits; 

**34.16	Domain specific features**

Reading research areas or literature to understand the best feature engineering methods as one technique will not fit all

**34.17	Feature slicing**

	Split data into feature values and build different models on the splits separately;

	Split criteria: feature values are different and sufficient data points are available;

	Example: Customers buying Desktops and Mobile

**34.18	Kaggle Winner Solutions**

Feature engineering methods can be learnt from kaggle winner solutions,

And discussion of competitions

**Chapter 35: Miscellaneous Topics**

**35.1	Calibration of Models: Need for calibration**

Binary or 2 class classification problem: We might require understanding a probability score of belonging to one class;

![][image180]

If:

**![][image181]** then class 1

These probabilities may not be exact probability; values between 0 and 1 can be given by sigmoid;

Calibration is required for computing the loss functions; these loss functions might depend on exact probabilities;

**35.2	Calibration plots**

	Model f trained on a training dataset; For each data point this model will give an output: 

		Build a data frame of x\_i, y\_true and y\_pred;

		Sort this data frame in increasing order of y\_pred;

		Break table into chunks of size m;

In each chunk compute average of y\_pred and y\_true in the chunk; avg\_y\_pred and avg\_y\_true

Task of calibration: make avg\_y\_pred proportional to avg\_y\_true; an ideal model will result in a 450 line between avg\_y\_pred and avg\_y\_true;

**35.3	Platt’s Calibration / Scaling / Sigmoid Calibration**

	P(yq \= 1 / xq) \= 1/ (1 \+ exp(A f(xq) \+ B)

Platt scaling works only if the Calibration dataset distribution is similar to sigmoid function;

Isotonic calibration is generally applied: use case log loss

**35.4	Isotonic Regression**

Works even if calibration plot does not look like sigmoid function; if we fit a sigmoid function there will be a large error

Break the model into multiple linear regression models; piece wise linear models 

![][image182]

Solve an optimization where minimize the gap between the gaps of the model and the actual value;

Large data 🡪 large Cross Validation dataset 🡪 large Calibration data 🡪 Isotonic Reg

Small data 🡪 Platt Scaling

**35.5	Code Samples**

Links: 	[http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html](http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) 

[http://scikit-learn.org/stable/auto\_examples/calibration/plot\_calibration.html\#sphx-glr-auto-examples-calibration-plot-calibration-py](http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py)

**35.6	Modeling in the presence of outliers: RANSAC**

	Random Sampling Consensus:

	To build a robust model in the presence of outliers:

	Sampling will reduce the presence of outliers in the training set

	Apply the model and compute loss;

Determine outliers based on loss function and remove some outliers; repeat the process until satisfied; Reduction of dataset at each stage with removal of outliers based on random sampling, model training and loss computation;

**35.7	Productionizing models**

	Link: [http://scikit-learn.org/stable/modules/model\_persistence.html](http://scikit-learn.org/stable/modules/model_persistence.html) 

1. Using SKLearn: model persistence: storing in pickle format

   Using joblib.dump() from sklearn.externals.joblib

   Load using joblib.load()

2. Custom implementation:

   Store all of the parameters of the model to a file:

   	For low latency applications; Use C/ C++ code for faster predictions;

   Store weights using data structures, Using CPU cache; RF and DTs with if else statements

	Implementation is problem specific

**35.8	Retraining models periodically**

	Time series data: Stock price prediction:

Model need to be retrained periodically to gather current trends and also to get around with new companies, etc.

Dataset changes and dropping model performance or retrain regularly;

**35.9	A/B Testing**

	\= Bucket testing or split run or controlled experiments;

Example: With respect to curing cold: If a new drug is found and need to be experimented for its effectiveness: A part of group of patients is administered with the new drug and another part of the group is administered with the old drug. Person with old drug/data are called control group and the group with new drug is called treatment group. The control and treatment groups are called A and B groups. 

**35.10	Data Science Life Cycle**

	1\. Understand business requirements: define the problem and the customer

	2\. Data acquisition: ETL (extract transform and load), SQL

	3\. Data preparation: Cleaning and pre-processing

	4\. EDA: plots, visualization, hypothesis testing, data slicing

	5\. Modeling, evaluation and interpretation

	6\. Communicate resulys: clear and simple, 1-pager and 6 pagers

	7\. Deployment

	8\. Real-world testing: A/B testing

	9\. Customer/ business buy-in

	10\. Operations: retrain models, handle failures, process

	11\. Optimization: improve models, more data, more features, optimize code

**35.11 	Productionization and deployment of Machine Learning Models**

35.7 	Same topic 

**35.12	Live Session: Productionization and deployment of Machine Learning Models**

	\--- Live sessions notes	

**35.13	Hands on Live Session: Deploy and ML model using APIs on AWS**

	\--- Live sessions notes

**35.14	VC dimension**

	Measure the potential of a class of models:

		Linear Models, RBF SVM, Boosting algorithms, Neural Networks;

1. Practical approach: deploying and measuring the performance  
2. Statistical approach: using Vapnik – Chevvonenkis (also build SVMs)

   VC dimension is mostly used for research work not generally found in applied work;

   Linear model: Decision Surfaces: Logistic Regression, Linear SVM;

   VC dim (linear model) \= maximum number of points that can be shattered by a linear model for all possible configurations \= 3 (that are not collinear)

   Link for further reading: [https://en.wikipedia.org/wiki/VC\_dimension](https://en.wikipedia.org/wiki/VC_dimension) 

Theoretically, RBF SVM is powerful than all models as its VC dimension is infinity;

Practically, this did not apply due to limitations of assumptions;

