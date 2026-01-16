**Module 3: Foundations of Natural Language Processing and Machine Learning**

Chapter 18:	REAL WORLD PROBLEM: PREDICT RATING GIVEN PRODUCT REVIEWS ON AMAZON

18.1	DATASET OVERVIEW: AMAZON FINE FOOD REVIEWS (EDA)

	Amazon largest retailers of the world;

	The dataset consists of reviews of fine foods on Amazon;

	Half a million reviews over 13 years;

	Contains: Review ID, ProductId, UserId, ProfileName, Text, Summary, Score, etc.

Task: Given a review determine whether the review is positive; this helps whether we need to do any improvements in the product;

18.2	DATA CLEANING: DEDUPLICATION

	Garbage in garbage out;

	In fine food reviews, we have:

A single person giving reviews to multiple at the same time stamp and same review; these can be thought of as duplicate data;

pd.DataFrame.drop\_duplicates(list of features)

Other such errors in data needs to be found out; such as values that need to be less than 1, data types, etc.

18.3	WHY CONVERT TEXT TO A VECTOR?

	Linear algebra can be applied when the input is in numerical data type;

	Converting review text strings into a d dimensional numerical vector; 

If we plot the data points of n dimensional vector (including vectorized) text data, we can utilize linear algebra in terms of distance measurements and the side to which the data points exist with respect to a plane that separates the positive data points from negative data points;

	wTx1 \> 0 then review is positive else negative;

If similarity between r1 and r2 is more than similarity between r1 and r3, then r1 and r2 are closer than r1 and r3 (in terms of distances); similar texts must be closer geometrically;

18.4	BAG OF WORDS (BOW)

	Example:

	R1: This pasta is very tasty and affordable;

	R2: This pasta is not tasty and is affordable;

	R3: This pasta is delicious and cheap

	R4: Pasta is tasty and pasta tastes good;

	BOW: 	Construct a dictionary of all unique words in the reviews;

		d – Unique words: for n reviews or n documents;

 Construct a vector of size d; each element in the vector belongs to words such as ‘a’, ‘an’, ‘the’, ‘pasta’, …, ‘tasty’, ….

![][image74]

	Most of the elements in the vector are zero; thus we will have a sparse matrix;

	BOW: The reviews R1 and R2 are completely opposite but the data points are closer to each other; BOW does not perform well when there is a small change in the data;

	We can have a binary bag of words;

	BOW depend on count of words in each review, this discards the semantic meaning of the documents and documents that are completely opposite in meaning can lie closer to each other when there is very small change in the document (with respect to the words)

18.5	TEXT PREPROCESSING: STEMMING, STOP-WORD REMOVAL, TOKENIZATION, LEMMATIZATION.

	R1: This pasta is very tasty and affordable;

	R2: This pasta is not tasty and is affordable;

	R3: This pasta is delicious and cheap

	R4: Pasta is tasty and pasta tastes good;

The underlined words above are stop words which do not add any value to the semantic meaning of the document;

	Now with this we will have sparsity reduced; (nltk tool kit has stop words)

	We should convert everything in **lowercase**;	

Next we can **remove stop words** 

Stemming: Tastes, tasty and tasteful: indicate the same meaning which relates to taste;

	With stemming we can combine words which have same root words;

	Two of the algorithms are: Porter Stemmer and Snowball Stemmer;

	Snowball stemmer is more powerful

Tolenization: Breaking up a sentence into words; 

Lemmatization takes lemma of a word based on its intended meaning depending on the context of surrounding words;

18.6	UNI-GRAM, BI-GRAM, N-GRAMS.

	R1: This pasta is very tasty and affordable;

	R2: This pasta is not tasty and is affordable;

		After data cleaning:

		R1: pasta tasty affordable

		R2: pasta tasty affordable

Uni grams consider each single word for counting; Bi grams considers two consecutive words at a time; We can have n grams similarly;

Uni grams based BOW discards sequence of information; with n grams we can retain some of the sequence information;

Dimensionality of the vector increases when including n grams;

18.7	TF-IDF (TERM FREQUENCY \- INVERSE DOCUMENT FREQUENCY)

BOW: one way to vectorize text data: using n grams, preprocessing and there were variations of bag of words;

TF: Term frequency;

TF(Wi, rj) \= \# of times occurs in rj / total number of words in rj; can also be thought of as probability of finding a word Wi in a document rj;

IDF(Wi, Dc) \= log(N/ni)

rj: jth review, Dc: Document corpus, Wi: ith word; N: \#of documents, ni \= \# docs which contain Wi;

N/ni \>=1; log(N/ni) \>= 0 ; IDF \>= 0

If ni increases log(N/ni) decreases;

If Wi is more frequent in corpus then its IDF is less; 

Rare words in documents that occur more frequently in a review  will have high TF-IDF value; TFIDF \= tf(Wi, rj) \* idf(Wi, Dc);

More importance to rarer words in corpus and give more importance to words that are frequent in a review;

18.8	WHY USE LOG IN IDF?

	IDF(Wi, Dc)  \= log(N/ni);

	Usage of log can be understood from Zipf’s law;

Words occurrences in English or any language follows a power law distribution, some words such as ‘the’ and ‘in’ are frequently found in usage, while such as geometry and civilization occur less in text documents;

When a log – log plot is applied we can find a straight line;

IDF without a log will have large numbers that will dominate in the ML model;

Taking log will bring down dominance of IDF in TFIDF values;

18.9	WORD2VEC.

	This technique takes semantic meaning of sentences;  

Word is transformed into a d dimension vector; This is not a sparse vector and generally has dimensionality \<\< BOW and TFIDF

If two words are semantically similar, then vectors of these words are closer geometrically;

Word2vec also retains relationships between words such as King:Queen::Man:Woman

	Vman – Vwoman || Vking – Vqueen (|| \- parallel vectors)

	Vitaly – Vrome || Vgermany – Vberlin

	Vwalking – Vwalked || Vswimming – Vswam

Word2vec learns these properties automatically; Why part of the Word2vec can be learnt in Matrix Factorization;

Working: Takes a text corpus (large size): for every word its builds a vector: dimension of the vectors is given as an input; this uses words in the neighborhood: if the neighborhood is similar then the words have similar vectors;

	Google took data corpus from Google News to train its Word2Vec method;

18.10	AVG-WORD2VEC, TF-IDF WEIGHTED WORD2VEC

	Each review has a sequence of words/sentences

	Vector V1 of review r1: avg w2v(r1) \= (1/\# words in r1) (w2v(w1) \+ w2v(w2) \+ ….)

	Tfidf w2v (r1) \= (tfidf(w1)\*w2v(w1) \+ tfidf(w2)\*w2v(w2) \+ …) / (tfidf(w1) \+ tfidf(w2) \+ …)

18.11	BAG OF WORDS (CODE SAMPLE)

	Sklearn’s CountVectorizer()

	Counts of each word in the corpus;

	Sparsity \= \# of zero elements / total elements

With sparse matrices, we can store sparse matrices with a memory efficient method by storing the row and column indices with corresponding values;

18.12	TEXT PREPROCESSING (CODE SAMPLE)

Removing html tags, punctuations, stop words, and alpha numeric words, ensuring number of letters \>2, lowercases, using stemming and lemmatization;

Python module re: regular expressions; Can be used extensively for text preprocessing;

18.13	BI-GRAMS AND N-GRAMS (CODE SAMPLE)

Sklearn’s CountVectorizer(ngram\_range \= (1,2))

18.14	TF-IDF (CODE SAMPLE)

	Sklearn’s TfidfVectorizer(ngram\_range \= (1,2))

18.15	WORD2VEC (CODE SAMPLE)

	Using gensim module: stemming words may not have w2v vectors in pre trained model

18.16	AVG-WORD2VEC AND TFIDF-WORD2VEC (CODE SAMPLE)

	We use formulae to compute these vectors; Achieved using a for loop;

Chapter 19:	 CLASSIFICATION AND REGRESSION MODELS: K-NEAREST NEIGHBORS

19.1	HOW “CLASSIFICATION” WORKS?

	Working of classification: 

We are using text which is most informative; text is converted into a vector and based on vector we have either positive or negative reviews;

Task of classification is to find a function that takes input text vector and gives output as positive or negative for any new review text;

We are classifying a new review into two classes positive or negative;

Most of Machine Learning is finding a mathematical function;

Y \= F(X)

The classification model gets training data and the model learns the function F observing examples; then this learnt function is used to predict the output of new unseen input data; Unseen data implies that the data is not used during training;

19.2	DATA MATRIX NOTATION

Given input matrix X; each row Xi represents a review text vector; for each review we will have a class label which is denoted as Yi;

A vector is by default a column vector;

In Machine Learning we will leverage Linear Algebra for computations; We cannot input text into Linear Algebra; thus even y value are vectorized; here we have two classes thus we can have a binary label 0 or 1; 1 representing positive review, 0 for negative review;

19.3	CLASSIFICATION VS REGRESSION (EXAMPLES)

	Dn \= {(xi, yi) | xi ε Rd, yi \= {0, 1}}: Binary classification

	For MNIST: Dn \= {(xi, yi) | xi ε Rnxn, yi \= {0, 1, 2, ……., 9}}: Multi class classification

	Regression: Heights of a school of students: a continuous random variable: Regression

19.4	K-NEAREST NEIGHBOURS GEOMETRIC INTUITION WITH A TOY EXAMPLE

	Simple and a powerful Machine Learning Algorithm: Classification and Regression

	2D toy Dataset:	![][image75]

Blue points: \+ ve data points; Orange points: \- ve data points; Yellow: query point (xq)

Task: Predict the class of query point: It can be classified into a class based on the classes of the neighbors of the query data point;

1. Find ‘k’ nearest points to xq in D; (nearest say in terms of distances)  
2. Take class labels of the k nearest data points and apply majority vote; If majority of the nearest data points belong to positive class then the data point can be assigned with positive class label and vice versa;

   Get nearest neighbors and apply majority vote; If K is even then majority votes can result in ties, thus avoid even numbers for k

19.5	FAILURE CASES OF KNN

1\.	Outliers (nearest neighbors are far to the outlier query point, assigning a class based on kNN is not good)

2\. 	If the dataset is excessively randomized or a completely mixed up data, then there is no useful information in the data; most ML models fail in this case;

19.6	DISTANCE MEASURES: EUCLIDEAN (L2) , MANHATTAN(L1), MINKOWSKI, HAMMING

	![][image76]

	This distance is known as a Euclidean distance (shortest distance between two points)

	It is called as L2 norm when the input is a vector;

	x1 ε Rd 	and x2 ε Rd:

		Euclidean distance: 	||x1 – x2||2 \= (summ(x1i – x2i)2)1/2

					||x1 – x2||2 🡪L2 Norm of vector

					||x1||2 \= (summ(x1i)2)1/2

		Manhattan distance: 	||x1 – x2||1 \= (summ(abs(x1i – x2i)))

					🡪 L1 norm of vector

					||x1||1 \= (summ(abs(x1i)))

		Minkowski distance:	Lp norm: ||x1 – x2||p \= (summ(abs(x1i – x2i))p)1/p

					P\>2

Distances are between two points and Norm is for vectors;

Hamming distance:	Number of locations where there is a difference in two vectors

Text processing (binary BOW or Boolean vector)

					X1 \= \[0, 1, 1, 0, 1, 0, 0\]

					X2 \= \[1, 0, 1, 0, 1, 0, 1\]

					H(X1, X2) \= 1+1+0+0+0+0+1 \= 3

					X1 \= ‘abcadefghik’

					X2 \= ‘acbadegfhik’

					H(X1, X2) \= 0+1+1+0+0+1+1+0+0+0 \= 4

				Hamming distance is applied to Gene encoding

19.7	COSINE DISTANCE & COSINE SIMILARITY

	Similarity vs Distance: As distance increases the two points are less similar;

		Similarity and Distances have opposite behavior

		cos\_dist \= 1 – cos\_sim

	![][image77]

	![][image78]

	cos\_sim(x1, x3) \= 1

	cos\_sim(x1, x2) \= cos θ

	cos\_dist(x1, x3) \= 1 – 1 \= 0

	cos\_dist(x1, x2) \= 1 – cos θ

	Even though dist(x1, x3) \> dist(x1, x2): cos\_dist(x1, x3) \< cos\_dist(x1, x2);

	Cos\_sim(x1, x2) \= x1.x2/||x1||2 ||x2||2

	![][image79]

If x1 and x2 are unit vectors, then square of Euclidean distance between x1 and x2 \= 2 times cosine distance between x1 and x2

19.8	HOW TO MEASURE THE EFFECTIVENESS OF K-NN?

	Given: a text reviews data with class labels;

	Problem: What is polarity (+ve, \-ve) of a new review?

	kNN 🡪 nearest neighbors \+ majority vote

	To measure the performance of the any ML model:

Break the dataset into two sets train and test and there is no intersection between train and test data points (each data point either goes to train set or to test set)

Splitting can be done randomly (one of the methods) into train set \= 70% of the total dataset and test set \= rest 30% of the total dataset;

For each point in test dataset; make data point as a query point, then use train set and kNN model to determine yq; then if class label is correct increment a counter by 1; for metrics we can use this counter which is the number correct classifications in the test set; accuracy can be computed by (counter/test size);

This accuracy will allow us to understand the performance or effectiveness of kNN model;

19.9	TEST/ EVALUATION TIME AND SPACE COMPLEXITY

	Given a query point:

		Input: Dtrain, k, query

		Output: yq 

Knnpts \= \[\]

	For each xi in dtrain:					\~O(nd)

		Compute d(xi, xq) 🡪 di				\~ O(d)

		Keep smallest k distances and store in a list;		\~ O(1)

	Majority vote:						\~O(1)

	Time complexity \= O(nd)

	Space:	At evaluation:

Dtrain	\~O(nd)

		After deployment, Dtrain is required to be present in RAM;

19.10	KNN LIMITATIONS

	1\. Due to large space complexity, we will require large RAM at deployment;

	2\. Low Latency is of high preference;

	 We can use kd tree and LSH for improving kNN performance;

19.11	DECISION SURFACE FOR K-NN AS K CHANGES

	‘k’ in kNN is a hyper parameter;

	k \= 1;

	![][image80]

	Curves that separate \+ve points from –ve points are called decision surfaces;

	For k \= 1: 

		Green curve is not smooth; and happens when k \= 1;

	For k \= 5:

		Yellow curve may be generated when k \= 5; this is a smoother curve

	As k increases smoothness of the decision surfaces increases;

	For k \= n:

		All query points are classified as the majority class;	

19.12	OVERFITTING AND UNDERFITTING

Overfitting: Generating extremely non smooth surfaces on training data to get extremely high accuracy on training set;

Underfit: Not generating any decision surface and classifying all query points with majority class;

Neither overfit nor underfit: Generating smooth surfaces that are less prone to noise;

19.13	NEED FOR CROSS VALIDATION

	Determining k hyper parameter for kNN model:

	We split the data set into train and test set;

	We train the model using training set and compute accuracy using test set;

For every k we determine test accuracy and select k that gives best accuracy on test set; 

Using test dataset to determine best k or best hyper parameters is not right;

Thus we split the data set into train, cross validate and test datasets; we want the model to have well generalization ability on future unseen data;

We train the model parameters compute nearest neighbors on training set, we determine best hyper parameters on cross validation dataset and we evaluate best ML models on test set; this will help ML models have good generalization ability;

19.14	K-FOLD CROSS VALIDATION

	Data is split into train, cross validation and test set; 

While test set should not be touched, cross validation data becomes untouched while training which will lead to loss of information;

We can use k-fold cross validation to incorporate cross validation data during training;

We are trying to get the information from cross validation set for ML model training;

1. Dataset is split randomly into train and test set;  
2. Train set is further divided randomly into k’ equal sized parts;  
3. For each k hyper parameter in kNN, we will use k’-1 parts of train set for training and the remaining 1 part of train set as cross validation data set, we then compute accuracy or model performance on cross validation data set; we will roll around the parts of train data set to get k’ accuracies for k \= 1 (kNN hyperparameter), we will then average this accuracies for k \= 1; as a result we will have average of k’ accuracies of the train set; this is called as k’-fold cross validation, we will repeat the k’ fold cross validation for all hyper parameter k value choices;  
4. We pick best k from best average k’ cross validation accuracies;  
5. And apply the best hyper parameter for measuring performance on test set;

   Generally 10 fold cross validation is applied;

   With k’ fold cross validation time taken for finding optimal k is multiplied by k’ times;

   But finding optimal k is a onetime effort;

19.15	VISUALIZING TRAIN, VALIDATION AND TEST DATASETS

	![][image81]

![][image82]

19.16	HOW TO DETERMINE OVERFITTING AND UNDERFITTING?

	We are finding best k that is neither underfitting nor overfitting;

	Accuracy: \# correctly classified points/ total \# of points

	Error: 1 – accuracy

	We can compute train error and cross validation error;

	Plot a curve on error vs k hyperparameter;

	![][image83]

	 ![][image84]

	If train error is high and validation error is high then we are underfitting;

	If train error is low and validation error is high then we are overfitting;

	At optimal k we will have closer train error and validation error;

19.17	TIME BASED SPLITTING

	Data was split randomly so far;

For Amazon fine food reviews time based splitting is better as we also have timestamp feature in the dataset;

We will sort the data in the ascending order of timestamp;

If time stamps are available we need to check which of random split or time based split is better;

In random splitting: we will get reviews that were written before will occur in test set and the reviews that were written after will occur in train set;

In time based splitting, we will get reviews in train, cross validation and test set based on time stamps; the reviews that were written first occur in train set, then in cross validation and then in test set; the argument here is we should avoid predicting past data based on models trained on future datasets;

Reviews or products change over time; new products get added or some products get be terminated;

As time passes we require retraining the model with new data;

19.18	K-NN FOR REGRESSION

	Classification: Dn \= {(xi, yi) | xi ε Rd, yi ε {0, 1}}; we do majority vote

	Regression: Dn \= {(xi, yi) | xi ε Rd, yi ε R}; we will do mean or median of all k neighbors

19.19	WEIGHTED K-NN

	Weights depending on the distance of nearest neighbors;

	Closer points will have high weights;

We can multiply the labels of nearest neighbors with reciprocal of the distance rather using simple majority vote;

19.20	VORONOI DIAGRAM

	This divides the whole data space into regions based on nearest neighbor;

	![][image85]

19.21	BINARY SEARCH TREE

	kNN: Time complexity: O(n) if d is small and k is small and Space complexity: O(n) 

	We can reduce time complexity to O(log n) using kd tree

Binary search tree:	Given a sorted array, we can find presence of a number in the array in O(log n) time

Construct a binary search tree; find element in array through comparisons;

Depth of BST is O(log n)

19.22	HOW TO BUILD A KD-TREE

BST: Given a list of sorted numbers we can build a tree that divides the list in two at every stage

Kd – Tree:

![][image86]	![][image87]

1. Pick first axis and project all data points on to the axis; pick the middle point; draw a plane through the middle point this divides the space into two equal halves;

![][image88]

2. Change axis to y;

   ![][image89]	![][image90]

3. Change to x axis again and repeat steps; the data space is broken using axis parallel lines or hyper planes into hyper cuboids	

19.23	FIND NEAREST NEIGHBOURS USING KD-TREE

	![][image91]

	xq \<= x1, yq \>= y1, xq \>= x5 🡪 c is one of the nearest points;

Draw a hyper sphere with centre xq and radius Dist(c,q); If there is  another point is inside the hypersphere; this can also be checked as the hyper sphere intersects with y \= y1 line;

We track back to y \<= y1 condition in the tree and search for another nearest data point; now point e can also be a nearest neighbor;

Dist(q, e) \< Dist(c,e): thus c is discarded as 1 nearest neighbor and take e as nearest neighbor; we repeat the steps done with c now with e; we can have a result that point e is the nearest neighbor to query point q;

Best case Time complexity is O(log n) and worst case t.c. is O(n);

For k Nearest Neighbors Time complexities: Best case O(k log n) and Worst case: O(kn)

19.24	LIMITATIONS OF KD TREE

As dimensionality increases the number of back tracks in kd tree worsens time complexity; kd tree nearest neighbor updates is exponentially proportional to dimensionality (2d)

O(log n) is valid for uniform distribution and as distribution moves towards real world clusters, time complexity moves towards O(n);

19.25	EXTENSIONS

	Go through Wikipedia article

19.26	HASHING VS LSH

	Locality sensitive hashing to find nearest neighbors when d is large;

	Given an unordered array apply a hash function;

When we need to find an element in the array we can use the hash table to look up for key value that is equal to query element and the value in the hash table will give us indices; 

Locality sensitive hashing: it computes a hash function such that nearest data points pairs are stored in the same bucket; for a new query point relevant bucket is searched for and k nearest neighbors are searched through data points in this bucket; this will reduce the need for searching throughout the data space;

19.27	LSH FOR COSINE SIMILARITY

	cos\_sim(x1, x2) \= cos θ

	As θ increases the cosine decreases; similarity decreases and distance increases;

LSH is a randomized algorithm; it will not give same answer every time; gives answers based on probability;

If two points are close in terms of angular distance, then the points are similar; these points should go to same bucket in hash table;

We break feature space using random hyper planes; for every plane we will have a unit normal vector ‘wi’

For plane 1: Say we have x1 above and x3 below the hyper planes; we will have:

	W1T . x1 \>=0

	W1T . x2 \<=0

Random hyper planes are generated by generating a random W vector; each value in W vector is randomly generated from Normal distribution N(0,1) (0 mean and 1 variance);

![][image92]

H(x7) \= \[+1, \-1, \-1\]	(3 random hyper planes are generated)	

 The key will hash function value and all the data points will be stored in value bucket;

![][image93]

Time to construct the hash table: O(mdn) : m hyper planes; d dimensionality, n data points

Space O(nd)

	Time at test time: O(md \+n’d); n’ is number of data points in hash table bucket

		m is set roughly to log n;

LSH for cosine similarity can miss nearest neighbors when the nearest neighbor falls in opposite side to any hyper plane;

When iterating across LSH we can get nearest neighbors in a bucket in some iteration; over iterations at each bucket key of the hash table we can do union of all data points;

19.28	LSH FOR EUCLIDEAN DISTANCE

	Simple extension:

		Each axis is divided into units:

Every point is projected on each axis of the data space; hash function will vectorize the data point into same dimensionality vector where each cell represents the part of the axis the data point projection lies on that axis;

![][image94] ![][image95]

![][image96]![][image97]

	LSH for cosine similarity encodes each feature to \+1 or \-1;

![][image98]

On x axis x1 and x2 fall in same bucket, on y axis these are very far;

19.29	PROBABILISTIC CLASS LABEL

Say, with 3 NN we get a query data point prediction as positive and with 5 data points we can get the prediction as negative;

Also, say we go for 7NN; with a query data point it might happen that it nearest neighbors are 4 \+ve and 3 –ve; say another query data point has 7 \+ve and 0 –ve neighbors; then both the cases give prediction as \+ve based on majority vote; but both data points do not have same number of \+ve \# of data points, query data point 1 is less \+ve than query data point 2; thus giving a probabilistic quantification of the prediction will give us more confidence;

Query 1 has 4/7 \+ve and 3/7 –ve;

Query 2 has 7/7 \+ve and 0/7 –ve;

This gives certainty statistic of predictions;

19.30	CODE SAMPLE: DECISION BOUNDARY

	sklearn.neighbors.KNeighbirsClassifier()

	![][image99] ![][image100]

	As k increases the decision boundary smoothens;

19.31	CODE SAMPLE: CROSS VALIDATION

	Train test split()

19.32	REVISION QUESTIONS

1. Explain about K-Nearest Neighbors?  
2. Failure cases of KNN?  
3. Define Distance measures: Euclidean(L2) , Manhattan(L1), Minkowski,  Hamming  
4. What is Cosine Distance & Cosine Similarity?   
5. How to measure the effectiveness of k-NN?  
6. Limitations of KNN?  
7. How to handle Overfitting and Underfitting in KNN?  
8. Need for Cross validation?  
9. What is K-fold cross validation?  
10. What is Time based splitting?   
11. Explain k-NN for regression?  
12. Weighted k-NN ?  
13. How to build a kd-tree.?  
14. Find nearest neighbors using kd-tree  
15. What is Locality sensitive Hashing (LSH)?(  
16. Hashing vs LSH?  
17. LSH for cosine similarity?  
18. LSH for euclidean distance?

  


Chapter 20:	 INTERVIEW QUESTIONS ON K-NN (K NEAREST NEIGHBOUR)

20.1	QUESTIONS & ANSWERS

1. In k-means or kNN, we use euclidean distance to calculate the distance between nearest neighbours. Why not manhattan distance ?(https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-k-nearest-neighbors-algorithm/)  
2. How to test and know whether or not we have overfitting problem?  
3. How is kNN different from k-means clustering?(https://stats.stackexchange.com/questions/56500/what-are-the-main-differences-between-k-means-and-k-nearest-neighbours)  
4. Can you explain the difference between a Test Set and a Validation Set?(https://stackoverflow.com/questions/2976452/whats-is-the-difference-between-train-validation-and-test-set-in-neural-netwo)  
5. How can you avoid overfitting in KNN?

**External Resources:** 1.[https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-k-nearest-neighbors-algorithm/](https://www.analyticsvidhya.com/blog/2017/09/30-questions-test-k-nearest-neighbors-algorithm/)

Chapter 21:	 CLASSIFICATION ALGORITHMS IN VARIOUS SITUATIONS

21.1	INTRODUCTION

	Applying any classification algorithms; 

Rather than knowing multiple algorithms; it is important to know type of problems that arise in real world;

21.2	IMBALANCED VS BALANCED DATASET

	2 classes: Dn: n data points: n1 \+ve, n2 –ve pts; n1 \+ n2 \= n

		If n1 \~ n2 🡪 balanced dataset (n1:n2 \= 58:42)

If n1\<\<n2 🡪 imbalanced dataset (n1:n2 \= 5:95); the results will be biased towards majority class;

	Given imbalanced dataset:

1. Undersampling: n1 \= 100 and n2 \= 900

   Create new dataset with 100 n1 points and 100 n2 points randomly selected; result 100 n1 and 100 n2 points;

   We are discarding valuable information; 80% of the dataset is discarded

   2. Oversampling: n1 \= 100 and n2 \= 900

      Create new dataset with 900 n1 points by repeating each point 9 times; and 900 n2 points; repeating more points from minority class to make the dataset a balanced dataset;

      We can create artificial or synthetic new points through extrapolation to increase n1 from 100 to 900;

      We are not losing any data; we can also give weights to classes; more weight to minority class;

      The nearest data point if belongs to minority class it is counted as 9 points;

   When directly using the original imbalanced dataset; we can get high accuracy with a dumb model that predicts every query point to belong to majority class;

21.3	MULTI-CLASS CLASSIFICATION

	Binary classifier: y ε {0, 1}

	Multi class classifier:

		MNIST: y ε {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	 kNN uses majority vote, it can easily be extended to multi class classification very easily;

	But can we convert a multi class classification into a binary classifier;

The dataset is broken into parts such that for the first classifier we will have dataset with class labels class1 present and class1 absent;

We will build c binary classifiers for multi class classification problem with c labels;

This is called as One versus rest technique to do multi class classification; 

21.4	K-NN, GIVEN A DISTANCE OR SIMILARITY MATRIX

	If data points cannot be vectorized;

	But say we can get similarity matrix between data points;

	Example: similarity between medicines;

We will have rows and column with indices as data points and cell values will be similarity between row data point and column data point; we can convert similarity matrix into distance matrix using reciprocal of each similarity values;

We can use kNN for this distance or similarity matrices as kNN works on distances;

Techniques like Logistic regression on other side require d dimensional vector; thus we can think of limitations of various algorithms;

21.5	TRAIN AND TEST SET DIFFERENCES

	Given a dataset we had random splitting or time based splitting; 

	We can have new products getting added after some time or old products removed;

	The distribution of data points will change a lot;

	Train and CV error can be low; but we can have test error high;

	We need to check the distribution of train and test data sets for any change over time;

	Dn – 	Dtrain (xi, yi) – (xi’, yi’)

		Dtest (xi, yi) – (xi’, yi’)

	Dn’ \--	Dtrain ((xi, yi), 1\)

		Dtest ((xi, yi), 0\)

Train a binary classifier on Dn’; if the accuracy is high then Dtrain and Dtest are dis-similar; we will have small accuracy if Dtrain and Dtest are similar; to get Dtrain and Dtest follow same distribution (stable data) we will need to get small accuracy;

21.6	IMPACT OF OUTLIERS

kNN is highly impacted by outliers when k is small; decision surfaces change due to outliers;

Larger k is preferred when their performances are similar;

Outliers can be detected and removed;

21.7	LOCAL OUTLIER FACTOR (SIMPLE SOLUTION: MEAN DISTANCE TO KNN)

	To detect outliers in data inspired by ideas in kNN;

	We can have different dense clusters in the data space; 

For every data point compute get k Nearest Neighbors, compute average of k distances; sort data points with by average distances; this will remove global outliers;

Compute locate density; to remove local outliers;

21.8	K DISTANCE

	K\_distance (xi) \= distance of the kth nearest neighbor of xi from xi;

	N(xi) \= Neighborhood of xi; set of nearest neighbors

21.9	REACHABILITY-DISTANCE (A,B)

	Reachability\_distance(xi, xj) \= max(k\_distance(xj), dist(xi, xj))

	If xi is in N(xj): then reachability\_distance(xi, xj) \= k\_distance(xj)

	Else: reachability\_distance(xi, xj) \= dist(xi, xj)

21.10	LOCAL REACHABILITY-DENSITY (A)

	LRD(xi) \= 1/(summ (reach\_dist(xi, xj)/|N(xi)|)

Summation over all the points in the neighborhood of reachability distance by number of elements in the neighborhood;

Inverse of average reachability distance of xi from its neighbors;

21.11	LOCAL OUTLIER FACTOR (A)0

	LOF(xi) \= (summ(lrd(xj)/|N(xi)| \* 1/ lrd(xi)

		\= average lrd of pts in the neighborhood of xi / lrd of xi

If LOF(xi) is large, then xi is an outlier; it is large when lrd(xi) is small compared to its neighbors, that is the density near the point is small compared to its nearest neighbors;

If the density is small then the point is outlier;

Sklearn.neighbors.LocalOutlierFactor()

21.12	IMPACT OF SCALE & COLUMN STANDARDIZATION

Features in a dataset can have different scales; the distance measures will not be proper; the features that have large scales dominate the distance measurements;

	Example: 

	X1 \= \[23, 0.2\]

	X2 \= \[28, 0.2\]

	X3 \= \[23, 1.0\]

	Dist(X1, X2) \= 5;

	Dist(X1, X3) \= 0.8;

% difference in X1 and X2 \= 0.05 and X1 and X3 \= 0.8, though X1 and X3 are far compared to X1 and X2, Euclidean distance(X1, X2) \< E. d. (X1, X3)

Since Euclidean distance can be impacted by scale; column standardization should be applied before computing distances;

21.13	INTERPRETABILITY

A machine learning model should be interpretable; in addition to giving predictions it should explain why a prediction is made; this will make decisions reliable;

Example: Cancer prediction: 

An ML model trained on cancer data can provide predictions whether a new patient has cancer or not; but we completely do not rely on machines as this will cause chaos in real life; we will have a doctor who reads the predictions from the ML model and then communicates the findings with the patient; in addition to predictions the doctor needs evidence in terms of test results and other requirements to be provided by the ML model; based on the test results the doctor can explain the patient the presence or absence of cancer; 

Such a model is called an interpretable model;

A model that does not give reasoning or does not provide easy access to its decisions or predictions computation is called a black box model;

kNN is an interpretable model; as it can show the nearest data points based on which it has made prediction; the doctor can read similar patient records and come to a conclusion whether a patient has cancer or not;

The data point vector can be comprised of results of medical tests; such as weight, blood group, etc.

21.14	FEATURE IMPORTANCE AND FORWARD FEATURE SELECTION

Important features: Features those are useful for the machine learning model in making predictions; this improves model interpretability; feature importance allows us understand a model better;

kNN does not give feature importance by default;

Forward feature selection: Given a model f through forward feature selection we can use the model itself to get feature importance; Given a high dimensionality dataset we want to reduce dimensionality to make things easier for computations (curse of dimensionality); One way is to use PCA/ tSNE; but PCA and tSNE care about distances and do not care about classification task; but for classification task using forward feature selection we can discard less important features;

1. Given a dataset of d features; use each feature at a time to train an ML model, the performance of the model is noted with respect to each feature; the feature that gave highest accuracy is selected say this is fs1 (feature selected at stage1)  
   2. Retrain the model with remaining features in concatenation with fs1 one at a time; we will get fs2; here fs2 \+ fs1 will give highest accuracy;  
   3. Repeat these steps up to fsd; these stage wise concatenation of features to gain high performance from the model  is called forward feature selection;  

   Note: At first stage we have the second best feature; it may happen that this feature in combination with the first best feature may not provide good performance; so at each stage we check that given that a feature or set of features are selected previously as best features we now explore for features that add most value to model performance;

   We can have a backward selection; where we try to remove the feature that results in lowest drop in performance of the mode;

   At any iteration we are training the ML model; time complexity is very high;	

21.15	HANDLING CATEGORICAL AND NUMERICAL FEATURES

	Given a dataset we can have multiple feature data types;

For a height prediction regression problem, we have weight, country, hair color, etc as features; Weight is a numerical feature which can be given an input to the model; Hair color and country are categorical text input; ML models take numerical as input; we require to convert text and categorical features into equivalent numerical feature;

1. Say for hair color we have black, red, brown, Gray as values; we can assign numbers to each color as black \= 1; red \= 2, etc; but numbers are said to be ordinal; 2 is greater than 1 but red greater than black is absurd; with this numerical conversion we are inducing an artificial order in the categories;  
   2. Thus one hot encoding is a better option; each category in the color feature is made as a new binary feature; disadvantage: dimensionality of the dataset increases and the dataset will be a sparse matrix as in each row there will only one cell which is non zero;  
   3. Mean replacement category wise: Replace country column by the average height of the people from that country;  
   4. Using domain knowledge: example for country we can replace the value with distance from some reference country; or coordinate location on map; say we have some fact that person near equator are tall and person away from equator is short which is stated by a domain expert; 

   Ordinal features such as ratings: we can convert that into numbers which can be compared (V. good, good, avg, bad, v. bad); \= (5, 4, 3, 2, 1\) or (10, 6, 4, 2, 1\) decisions on this are random;

21.16	HANDLING MISSING VALUES BY IMPUTATION

	Given a dataset Dn, we could have missing values due to many reasonable reasons; 

	Task: Filling missing values: 

1. Mean imputation, median imputation and mode imputation;

   We can impute missing values with mean, median or mode of the feature;

   2.  Classification task: we can also impute based on class labels;

   We can have Mean of all points or mean of all positive points or negative points;

   ![][image101]

   Yellow mean of whole dataset; Green: mean of blue points only;

   3. In addition to imputation we can add a new missing feature: we can create a new feature which also tells whether a missing value appeared in the original feature; we are adding binary features which tell us that a value is missing  
   4. Model based imputation: We will consider feature that has missing values as target variable and make predictions based on non missing features; Missing values are kept into test set; kNN is used for model based imputation where neighborhood can be used to fill or impute missing values;

   

21.17	CURSE OF DIMENSIONALITY

	When d is high:

* In machine learning; let all features be binary where 2 is minimum possible number of values for the feature0 and 1; for d number of features we will have 2d possible values for the dataset; so as dimensionality increases the number of data points required for the model to perform well increases exponentially;

  Hughes phenomenon: for fixed number of training samples the predictive power reduces as the dimensionality increases;

  As dimensionality increases the number of data points required for the model to perform well increases incrementally;

* Distance functions: intuition of distances becomes invalid in high dimensionality; 

  If a large number of random data points are selected in a high dimensionality space, the minimum of distance between every pair of points is roughly similar to the maximum of distance between every pair of points; this makes comparison of Euclidean distances for data points similarity impossible as dimensionality increases; 

  Cosine similarity can be considered instead of Euclidean distance though cosine similarity is impacted by curse of dimensionality the impact is less when compared with the impact on distances; 

  For high dimensional data, the impact of curse of dimensionality is high for dense data compared to the impact on sparse data;

* Overfitting and Underfitting: As dimensionality increases chances of overfitting increases; we can avoid this with forward feature selection; we can also use PCA or tSNE which are not classification oriented, these preserve proximity and variance;

  kNN on text data: cosine similarity and sparse representations as these are less impacted;

  Additionally:

  You can also consider the length of diagonal for a unit hyper cube:

  	For 2D L(dia) \= sqrt(2)

  	For 3D L(dia) \= sqrt(3)

  	For 10 000 D L(dia) \= sqrt(10000) \= 100

  Thus even though the data points lie in the same unit hyper cube they are very far;

  Similarly the average distance of random data points in a nD hyper cube is given as sqrt(10n/n);

  For 2D: avg dist \= sqrt(100/2) \= 7.07

  For 3D: 18.25

  For 6D: 408.24

  Thus the average distance you can get from a 6D dataset is very large; and real world problems come with dimensionality in range of 100s and 1000s, which worsens the distance measurements and the intuition of similarity among data points;

21.18	BIAS-VARIANCE TRADEOFF

	In kNN we saw for k \= n we have underfitting and for k \= 1 we have overfitting;

	Generalization error 	\= error ML model makes on future unseen data

			**Generalization** **Error	\= Bias2 \+ Variance \+ irreducible error**

Bias: due to simplifying assumptions; example for a quadratic function if we assume that the data points follow a linear function we will have bias;

Example: if we have an imbalanced dataset and if we assume that the whole dataset belongs to dominant class we will have bias;

High bias \= underfitting due to simplifying assumptions;

Variance: how much a model changes as training data changes; if the model does not change much with changes in training data we will have a low variance model ; 

	High bias \= overfitting due to restrictions; 

For k \= 1 in kNN small changes in dataset result in large changes in decision surfaces; this model will have high variance; As k increases variance reduces;	

At k \= 1 we will have a low bias and high variance model; as k increases bias increases slightly while variance increases drastically;

High bias leads to underfitting and high variance leads to overfitting thus we need a balance between bias and variance; as variance increases bias decreases;

![][image102]

21.19	INTUITIVE UNDERSTANDING OF BIAS-VARIANCE

Given dataset D split into Dtrain and Dtest; a model is built on train data; We will have train error and test error;

If train error is high we will have high bias which means the model is underfitting; if train error is low then bias will be low

For bias: we should have low train error;

For variance: we should have similar train and test error;

If train error is low and test error is high we will have model overfitting on train data; this model will have high variance; we can also observe for changes in the model predictions due to changes in training data;

20. REVISION QUESTIONS  
    1. What is Imbalanced and  balanced dataset.  
    2. Define Multi-class classification?  
    3. Explain Impact of Outliers?  
    4. What is Local Outlier Factor?  
    5. What is k-distance (A), N(A)  
    6. Define reachability-distance(A, B)?  
    7. What is Local-reachability-density(A)?  
    8. Define LOF(A)?  
    9. Impact of Scale & Column standardization?  
    10. What is Interpretability?  
    11. Handling categorical and numerical features?  
    12. Handling missing values by imputation?  
    13. Bias-Variance tradeoff?

21.21	BEST AND WORST CASE OF ALGORITHM

	kNN: 

1. When dimensionality is small: kNN is a good algorithm; for high dimensionality we will have curse of dimensionality and interpretability reduces; we will high run time complexity;  
   2. At run time we will have high time complexity; kNN cannot be used for low latency applications;  
      3. If we have the right distance measure, then kNN can be applied; if a right distance measure is not known kNN should be avoided; kNN is very good for similarity matrix or distance matrix;

Chapter 22:	 PERFORMANCE MEASUREMENT OF MODELS

22.1	ACCURACY

	Task: Measure performance of ML models;

	Accuracy: ratio of number of correctly classified points to total number of points;

Case 1: if we have an imbalanced dataset, with a dumb model we can get high accuracies; Accuracy cannot be used for imbalanced datasets;

Case 2: If we have models which return a probability score; an inferior model predicts probability values far from true labels (near 0.5 predictions) while a powerful model predicts probability values nearer to true labels (far from 0.5 predictions); accuracy can say that an inferior model is working similar to a powerful model; thus accuracy cannot give an idea whether a model is inferior or powerful;

22.2	CONFUSION MATRIX, TPR, FPR, FNR, TNR

In a binary classification problem we have two classes; a confusion matrix can be built with column vectors as predicted class labels and row vectors as actual class labels; Confusion matrix cannot process probability values;

| Confusion Matrix | Actual 0 | Actual 1 |
| ----- | ----- | ----- |
| Predicted 0 | a | b |
| Predicted 1 | c | d |

| Confusion Matrix | Predicted 0 | Predicted 1 |
| ----- | ----- | ----- |
| Actual 0 | a | c |
| Actual 1 | b | d |

To construct a confusion matrix we need to have true class labels and predicted class labels;

a \= \# of points that have actual and predicted labels as 0;

For Multi class classification: we will have a matrix of size cxc where c is the number of classes;

If the model is sensible, then the principal diagonal elements will be high and off diagonal elements will be low;

a \= True Negatives, b \= False Negatives, c \= False Positives, d \= True Positives

True Negatives: Predicted Negatives that are true or correct

False Negatives: Predicted Negatives that are false, these points are actually positives

True Positives: Predicted Positives correctly;

False Positives: Predicted Positives are wrong, these are actually negatives;

TPR (True Positive Rate) \= TP / P \= TP/ (TP+FN)

TNR \= TN/N \= TN/ (TN \+ FP)

FPR \= FP / N

FNR \= FN / P

Model is good if TPR, TNR are high and FPR, FNR are low;

A dumb model makes one of TPR or TNR \= 0;

The condition for which of TPR, TNR, FPR and FNR should be considered depends on domain;

For medical purposes: we don’t want to miss a patient who has a disease; rather a patient with no disease can further be sent to tests;  but a patient cannot be left untreated due to False Negatives;

22.3	PRECISION AND RECALL, F1-SCORE

	Precision \= TP/ (TP \+ FP); of all the predicted positives how many are actually positive

	Recall \= TP / (TP \+ FN); of all actual positives how many are predicted positive 

F1 score \= 2 \* Pr \* Re / (Pr \+ Re); high precision will result in high precision and high recall; f1 score is harmonic mean of precision and recall;

F1 score \= 2/(1/PR \+ 1/Re) \= 2TP/(2TP \+ FP \+ FN)

22.4	RECEIVER OPERATING CHARACTERISTIC CURVE (ROC) CURVE AND AUC

	Sort the data frame by descending order of predicted class labels;

	Thresholding: 

	![][image103]

We can have n thresholds with n TPR and FPR;

These TPR and FPR can be plotted which generates a curve called Receiver Operating Characteristic Curve;

![][image104]

AUC ranges from 0 to 1; 1 being ideal; AUC can be high even for a dumb model when the data set is imbalanced;

AUC is not dependent on the predicted values, rather it considers the ordering; if two models give same order of predicted values then AUC will be same for both the models; 

AUC for a random model is 0.5;

Preferred AUC is a value \> 0.5; AUC \= 0.5 for a random model and AUC between 0 and 0.5 imply that the predictions are reversed; if the predictions are again reversed then the new AUC value will be 1 – old AUC;

22.5	LOG-LOSS

	Binary classification problem: Log Loss uses probability scores; 

	Log loss \= \- (1/n) summ( yi \* log(pi) \+ (1 \- yi) \* log(1 \- pi)

	![][image105]

Log loss takes care of mis-classifications; this metric penalizes even for small deviations from actual class label

Log – loss: average negative logarithm of probability of correct class label;

For multi class classification: 

![][image106]

22.6	R-SQUARED/COEFFICIENT OF DETERMINATION

	Sum of squared errors: sum of (yi – yavg)2; 

	A simplest model can output for every query point a mean of the whole dataset;

	Sum of squares of residues \= sum (yi – ypred)2;

	R2 \= 1 – (SSres/ SStot);

	Case 1: SSres \= 0; R2 \= 1 – best value

	Case 2: SSres \< SStot; R2 \= 0 to 1;

	Case 3: SSres \= SStot; R2 \= 0: this model is same as simple mean model

	Case 4: SSres \> SStot; R2 \< 0: this model is worse than a simple mean model

22.7	MEDIAN ABSOLUTE DEVIATION (MAD)

SSres \= summ (ei2); if one of the errors is very large, then R2 is highly impacted, it is not robust to outliers;

Median Absolute Deviation is robust metric: Median (|ei – Median(ei)|)

This is a robust measure of standard deviation;

Median(ei) \= central value of errors;

22.8	DISTRIBUTION OF ERRORS

	If we plot PDF and CDF for errors (ei):

	![][image107]

Very few errors are large; ideally we require 0 errors; from CDF we can get the percentage of data points that have errors;

This can be used to compare two models;

If model M1 CDF is above M2 CDF then model M1 is better than model M2

![][image108]

	 

22.9	REVISION QUESTIONS

1. What is Accuracy?  
2. Explain about Confusion matrix, TPR, FPR, FNR, and TNR?  
3. What do you understand about Precision & recall, F1-score? How would you use it?  
4. What is the ROC Curve and what is AUC (a.k.a. AUROC)?  
5. What is Log-loss and how it helps to improve performance?  
6. Explain about R-Squared/ Coefficient of determination  
7. Explain about Median absolute deviation (MAD)? Importance of MAD?  
8. Define Distribution of errors?

Chapter 23:	 INTERVIEW QUESTIONS ON PERFORMANCE MEASUREMENT MODELS

23.1	QUESTIONS & ANSWERS

1. Which is more important to you– model accuracy, or model performance?  
2. Can you cite some examples where a false positive is important than a false negative?  
3. Can you cite some examples where a false negative important than a false positive?  
4. Can you cite some examples where both false positive and false negatives are equally important?  
5. What is the most frequent metric to assess model accuracy for classification problems?  
6. Why is Area Under ROC Curve (AUROC) better than raw accuracy as an out-of- sample evaluation metric?

Chapter 24:	 NAIVE BAYES

24.1	CONDITIONAL PROBABILITY

	A classification algorithm based on probability;

	kNN: Neighborhood based classification or regression model;

	P(A|B) \= P(A) \* P(B|A) / P(B)

	Example: 2 fair 6 sided dice:

	D1: Rolling dice 1; D2: Rolling dice 2;

	Sample space \= 36 outcomes

	P(D1 \= 2\) \= 1/6

	P(D2 \= 2|D1 \= 2\) \= 1/36;

	P(D1+D2 \<=5) \= 10/36

![][image109]

P(D1 \= 2 | D1+D2 \< \= 5\) \= 3/10

![][image110]

There are 10 outcomes where D1 \+ D2 \<= 5 and out of these 10 outcomes D1 \= 2 for 3 outcomes

P(A|B) \= P(A and B) / P(B);

P(A and B) \= P(A intersection B) \= P(A|B) \* P(B) \= P(B|A)\*P(A)

24.2	INDEPENDENT VS MUTUALLY EXCLUSIVE EVENTS

	P(A and B) \= P(A intersection B)

	Independent events:

P(A|B) \= P(A) and P(B|A) \= P(B);

	Mutually exclusive events:

		P(A|B) \= P(B|A) \= 0; 

P(A intersection B) \= P(B intersection A) \= 0

24.3	BAYES THEOREM WITH EXAMPLES

	P(A|B) \= P(B|A) \* P(A) / P(B) if P(B) \!=0

	Posterior \= likelihood\*prior/evidence

	P(A|B) \= P(A and B) / P(B) \= P(A, B) / P(B)

	P(A, B) \= P(B, A)

	P(B|A) \= P(B, A)/P(A)

	P(B,A) \= P(B|A) \* P(A) \= P(A, B) \= P(A|B) \* P(B)

	Example:

		Output: M1: 0.2, M2 \= 0.3, M3 \= 05

		Defective: M1: 0.05. M2 \= 0.03, M3 \= 0.01;

		P(A3|B) \= ?

		P(A1) \= 0.2, P(A2) \= 0.3, P(A3) \= 0.5

		P(B|A1) \= 0.05, P(B|A2) \= 0.03, P(B|A3) \= 0.01;

		P(B) \= summ(P(B|Ai) \*P(Ai)) \= 0.05\*0.2 \+ 0.03 \*0.3 \+ 0.01\*0.5 \= 0.024

		P(A3|B) \= P(A3) \* P(B|A3) / P(B) \= 0.5\*0.01/0.024 \= 5/24

24.4	EXERCISE PROBLEMS ON BAYES THEOREM

	PDFs available in downloads; 

24.5	NAIVE BAYES ALGORITHM

	Naïve: Simplistic, Unsophisticated;

	![][image111]

	![][image112]

![][image113]

Naïve: Features are conditionally independent of each other

24.6	TOY EXAMPLE: TRAIN AND TEST STAGES

![][image114]

	To determine: P(play \= yes |xq) and P(play \= No|xq);

	![][image115]

	Building counts and calculating probability priors, likelihood and evidence;

In the training phase of Naïve Bayes we calculate all likelihood probabilities and evidence probability;

Time complexity: O(ndc) through optimization O(n)

Space complexity: O(dc) d features and c classes

![][image116]

These probabilities are proportional;

P(class \= No|x’) \> P(class \= Yes|x’); thus the prediction will be class \= No for x1 data point;

Test Time complexity: O(dc)

kNN space complexity: O(nd)

As compared to kNN Naïve Bayes is space efficient at run time; we can have low latency applications;

Refer: shatterline, Naïve Bayes

24.7	NAIVE BAYES ON TEXT DATA

	Naïve Bayes is applied successfully to text data;

	Task: Given text, compute probability of the text belonging to a class;

	Text is vectorized; 

	P(y=1|text) 	Proportional to P(y=1|W1, W2, ….., Wd);

			Proportional to P(y=1) \* P(W1|y=1) \* ….. P(Wd|y=1) 

	P(y \= 1\) is class prior and P(W1|y) is likelihood;

Naïve Bayes is often used as baseline benchmark model for text classification and suitable problems; all other algorithms are compared with Naïve Bayes performance;

24.8	LAPLACE/ADDITIVE SMOOTHING

	At the end of training all the likelihoods and priors are computed;

	At test time: 

		Say we find a new word for which likelihood is not available;

We will use Laplace smoothing (not Laplacian smoothing) as the word is not present in training data ideas such as making its likelihood as 0 or 1 or 0.5 is not right; thus we will add a smoothing value to the numerator and denominator for likelihood probability of the new word;

P(W’|y=1) \= (0 \+ α)/(n1 \+ αk)

	N1 \= \# of data points for y \= 1

	Smoothing coefficient, α \!= 0; generally \= 1

k \= number of unique values W’ can take (for a categorical feature it can k \= \# of unique categories)

When α \= large; the likelihood will be around 0.5;

Laplacian smoothing is applied to all words in training data and also to new words that occur in test data; 

We have an additive smoothing and generally 1 additive smoothing is applied that is Laplace smoothing with α \= 1;

24.9	LOG-PROBABILITIES FOR NUMERICAL STABILITY

	We have probability values which are bound between 0 and 1;

While the probabilities are multiplied the result will be extremely low which will affect our model performance and result interpretation; we will also land in rounding up errors for low decimal values;

We will take logarithm of probabilities to avoid the problems of multiplying values between 0 and 1;

24.10	BIAS AND VARIANCE TRADEOFF

	Naïve Bayes: High Bias 🡪 Underfitting and High Variance 🡪 Ovefitting;

	We have α as hyper parameter;

When α \= 0 🡪 small change in Dtrain results in large change in the model; high variance, overfitting;

When α \= Very large; all likelihoods will be equal to 0.5; this results in underfitting, a high bias model; this will result in predicting majority class as class labels for all test data points;

In kNN k is determined by using cross validation, similarly α is determined by cross validation;

24.11	FEATURE IMPORTANCE AND INTERPRETABILITY

	We have likelihood probabilities for all features;

Sort the probability values in descending order; for each class we will get an order of features which are important;

Features which have high likelihoods are most important features in classifying a data point;

Interpretability: Based on likelihood probability of features we can get why a data point is classified to a certain class;

24.12	IMBALANCED DATA

Class priors favors dominating class while comparing probabilities of a data point features belong to a class;

We can use upsampling or downsampling or drop prior probabilities;

Naïve Bayes is modified to account for class imbalance (less used)

When laplace smoothing is applied: alpha impacts more for minority class;

We can have different α values for different classes;

24.13	OUTLIERS

Outliers at test time are taken care by Laplace smoothing; and if a word occurs less frequently then discard that word; 

24.14	MISSING VALUES

	No case of missing value for text data;

	Categorical data: Nan can be considered as another category

	Numerical: Imputation

24.15	HANDLING NUMERICAL FEATURES (GAUSSIAN NB)

	Naïve Bayes developed for numerical features as Gaussian Naïve Bayes;

Let us assume that the numerical feature follows a Gaussian distribution with some mean and standard deviation; before this we need to consider the data points that belong to the class in consideration; after considering the **class data points** only we can compute the probability from PDF of the distribution of the feature; the distribution PDF can be computed from **mean, standard deviation** and the assumption of **Gaussian distribution**; 

Naïve Bayes on numerical features with assumption of Gaussian is called as Gaussian Naïve Bayes; on binary feature we will have Bernoulli Naïve Bayes; we can also have Multinomial Naïve Bayes when the distribution is assumed to be Multinomial;

Naïve Bayes has a fundamental assumption that all features are conditionally independent;

24.16	MULTICLASS CLASSIFICATION

	Inherent property of Naïve Bayes; can be extended to multi classes;

	We can directly compute class based probabilities;

24.17	SIMILARITY OR DISTANCE MATRIX

	kNN can be implemented for an easily distance matrix;

Naïve Bayes cannot be directly applied on distance matrix; we need probability values;

24.18	LARGE DIMENSIONALITY

Naïve Bayes can be extensively used for text classification; as dimensionality increases we need to consider using log of probabilities;

24.19	BEST AND WORST CASES

	If conditional independence deters NB performance deteriorates;

NB can still work reasonably well; as opposed to theoretical rigor where NB should work only for conditionally independent features;

NB is extensively used for Text classification and categorical features; NB is not much used to real valued features as real world distributions come in varied forms other than Gaussian;

NB is interpretable, we have feature importance, low runtime complexity, low train time complexity, low run time space complexity; NB is basically performing counting; 

NB can easily overfit and is tackled using Laplace smoothing;

24.20	CODE EXAMPLE

	Sklearn Naïve Bayes module;

	Different ML algorithms are implemented by just changing a single line of code;

	In ML the task is to understand the applications of different algorithms;

24.21	REVISION QUESTIONS

1. What is Conditional probability?  
2. Define Independent vs mutually exclusive events?  
3. Explain Bayes Theorem with example?  
4. How to apply Naive Bayes on Text data?  
5. What is Laplace/Additive Smoothing?  
6. Explain Log-probabilities for numerical stability?  
7. In Naive Bayes how to handle Bias and Variance tradeoff?  
8. What Imbalanced data?  
9. What is Outliers and how to handle outliers?  
10. How to handle Missing values?  
11. How to Handling Numerical features (Gaussian NB)  
12. Define Multiclass classification?

Chapter 25:	 LOGISTIC REGRESSION

25.1	GEOMETRIC INTUITION OF LOGISTIC REGRESSION

	Logistic Regression is actually a classification technique: 

Logistic Regression can be interpreted in terms of Geometry, Probability and loss function;

If data is linearly separable (a hyper plane can separate the data points into two classes);

Equation of plane: WTx \+ b \=0

	If hyper planes passes through origin then b \= 0; WTx \= 0

Logistic Regression: assumption: Classes are almost or perfectly linearly separable;

![][image117]

Task: Find plane that separates the data points;

Assumptions:

	kNN: Neighborhood similarity

	NB: Conditional Independence

	Log Reg: Linearly separable

Negative data points are labeled \-1 instead of 0;

	Distance of a point from the hyper plane is: di \= WTxi/||W||

	If W is a unit normal vector to the hyper plane:

		||W|| \= 1

	Then, di \= WTxi;

Now we will have a classifier where if the data point is in the same direction of the normal vector then it belongs to positive class else negative;

For this we have: WTxi \> 0 then yi \= \+1

For positive data points:	yi WTxi \> 0 🡪 the data point is correctly classified 

			yi \= true label \= \+1 and WTxi \> 0

For negative data points:	yi WTxi \> 0 🡪 the data point is correctly classified

			yi \= true label \= \-1 and WTxi \< 0

For all data points: yi WTxi \> 0 🡪 the data point is correctly classified

If yi WTxi \< 0 🡪 the data point is incorrectly classified;

The ML model should have minimum number of misclassifications;

Thus find W that maximizes summ(yi WTxi)

**W\* \= argmax(summ(yi WTxi))**

25.2	SIGMOID FUNCTION: SQUASHING

 (yi WTxi) \= signed distance;

Outliers impact W values largely;

![][image118]

![][image119]

Case 2:

![][image120]

This formulation is largely impacted by outliers or any changes in the training data;

We will use squashing to reduce the effect of this;

Idea of squashing: if the signed distance is small use it as it is and if the signed distance is large make it small;

We have sigmoid function which has this property; we can have other functions

Sigmoid(x): 1/(1+e\-yWx)

![][image121]

So if a point lies on the Hyperplane we will have WTx \= 0; then this point belonging to positive or negative class is 0.5; this can be seen from sigmoid(0) \= 0.5;

Max sum of signed distances was outlier prone;

Maximize sum of sigmoid of signed distances which is outlier resistant;

**W\* \= argmax summ 1/(1+ exp(-yi WT xi)**

The distances are squashed from \[– infinity, \+ infinity\] to \[0, 1\]

Sigmoid is easily differentiable and has probabilistic interpretation which helps in solving the optimization problem;

25.3	MATHEMATICAL FORMULATION OF OBJECTIVE FUNCTION

	Sigmoid function is a monotonic function;

	A function is monotonic which increases with x at all values;

	Log(x) is a monotonic function;

If a function is monotonic then when applied a monotonic function retains same maxima or minima; 

Argmin f(x) \= argmin g(f(x)) if g(x) is monotonic;

W\* 	\= argmax summ 1/(1+ exp(-yi WT xi)

	\= argmax summ log(1/(1+ exp(-yi WT xi))

	\= argmax summ \-log((1+ exp(-yi WT xi))

	\= argmin summ log((1+ exp(-yi WT xi))

	\= argmin summ log((exp(-yi WT xi))

	\= argmin summ (-yi WT xi)

	\= argmax summ (yi WT xi)

		yi \= {-1, \+1}

The formulation is same but the sigmoid version will be less impacted by outliers

Probabilistic interpretation:

**W\* \= argmin summ(-yi logpi – (1 – yi)log(1 – pi)); pi \= sigmoid(WTx)**

**yi \= {0, 1}**

Above Geometric interpretation:

**W\* \= 	argmin summ log((1+ exp(-yi WT xi))**

**yi \= {-1, \+1}**

25.4	WEIGHT VECTOR

	**W\* \= 	argmin summ log((1+ exp(-yi WT xi))**

	W\* \= Weight vector \= d dimensional vector;

Geometric intuition: The W vector is the optimal weight vector normal to a hyper plane that separates data points into classes where positive data points are in the direction of W;

![][image122]

Decision: Given xq predict yq

For Logistic Regression: If WTxq \> 0 then yq \= \+1; If WTxq \< 0 then yq \= \-1, if the point is on the hyper plane we cannot determine the class of the query point;

		Probabilistic interpretation: P(yq \= \+1) \= sigmoid (WTxq)

			(Remember the need for sigmoid function: squashing)

	Interpretation of Weight vectors:

Case 1:	If Wi \= \+ve then if xqi increases Wixqi increases, sigmoid (WTxq) increases, then P(yq\=+1) increases;

Case2:	If Wi \= \-ve then if xqi increases Wixqi decreases, sigmoid (WTxq) decreases, then P(yq\=+1) decreases and P(yq\=-1) increases;

25.5	L2 REGULARIZATION: OVERFITTING AND UNDERFITTING

	W\* \=  argmin summ log((1+ exp(-yi WT xi))

	Let Zi \= yi WT xi/||W||

	Then W\* \= argmin summ(i \= 1 to n) (log(1+exp(-Zi)))

		exp(-Zi)) \>= 0; 1 \+ exp(-Zi)) \>= 1; log(1+exp(-Zi))) \>= 0; 

Minimum value of summ(i \= 1 to n) (log(1+exp(-Zi)) is 0;

	If Zi tends to \+ infinity then the summ(i \= 1 to n) (log(1+exp(-Zi)) is 0;

If the selected W results in correctly classifying all training points and if Zi tends to infinity, then W is the best W on training data; this is a case of overfitting on training data as this does not guarantee good performance on test data; the train data may contain outliers to which the model has been perfectly fitted;

The overfitting tendency is constrained by using regularization; 

	W\* 	\=  argmin summ log((1+ exp(-yi WT xi)) \+ λWTW 

		\=  argmin summ log((1+ exp(-yi WT xi)) \+ λ\*||W||2 

	The regularization term will constrain W from reaching \+infinity or \- infinity;

	![][image123]

λ is a hyper parameter of regularization; this is determined using cross validation;

If λ \= 0 there is not regularization; the loss term optimization results in an overfitting model 🡪 high variance;

If λ is large then the loss term is diminished; the training data does not participate in the optimization and we are just optimizing for the regularization term; this leads to an underfitting model 🡪 high bias;

25.6	L1 REGULARIZATION AND SPARSITY

	Optimization problem: W\* 	\=  argmin summ log((1+ exp(-yi WT xi)) \+ λWTW

					\= 	logistic loss			\+ Regularization

	Objective for Regularization: Avoid W reaching infinity;

	L1 Regularization:

		W\* \= argmin logistic loss for training data \+ λ||W||1

		L1 Regularization induces Sparsity in the W vector;

Less important features become vanished in Logistic Regression with L1 Regularization;

If L2 Regularization is used Weight for less important features becomes small but remains non-zero;

	Elastic net: W \* \= argmin summ(i \= 1 to n) log(1+exp(-Zi) \+ λ1||W||1\+ λ2||W||2

25.7	PROBABILISTIC INTERPRETATION: GAUSSIAN NAIVE BAYES

No book encountered that covers all of geometric, probabilistic and loss minimization interpretations of logistic regression;

Naive Bayes: real valued features are Gaussian distributed and class label is a Bernoulli random variable;

Assumption: Xi is a feature, P(Xi|yi) is a Gaussian distribution and Xi‘s are conditionally independent

P(Y=1|X) \= 1/(1+exp(WTx))

Logistic Regression: Gaussian Naïve Bayes \+ Bernoulli;

![][image124]

![][image125]

![][image126]	

Link: [https://www.cs.cmu.edu/\~tom/mlbook/NBayesLogReg.pdf](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) 

25.8	LOSS MINIMIZATION INTERPRETATION 

	 W\* \= argmin number of misclassifications; 

	For optimization the function should be differentiable;

	![][image127]

	Logistic loss is a good approximation of 0-1 loss (step loss)

	With hinge loss we will have Support Vector Machines;

	![][image128]

25.9	HYPERPARAMETER SEARCH: GRID SEARCH AND RANDOM SEARCH

	![][image129]

	K in kNN is an integer; Logistic Regression λ is a real valued hyper parameter;

	Hyper parameter tuning technique: Grid Seach (Brute force)

	For λ in \[0.001, 0.01, 0.1, 1, 10, 100\]:

Compute cross validation error and select the best hyper parameter value from the plot;

	Elastic Net: λ1 and λ2;

	![][image130]

	Multiple computations or training is required;

	As \# of hyper parameters increase, the \# of training instances increase exponentially;

Random Search: avoids brute force and reduces the time spent for hyper parameter tuning by looking at a smaller set of hyper parameter choices; the technique considers random hyper parameter values for computing cross validation error in turn provides best hyper parameter choices that optimizes the algorithm;

25.10	COLUMN STANDARDIZATION

	Mean centering and scaling; used in kNN;

In logistic regression it is also mandatory to perform column standardization as we are using distances to compute the weight vector and the scale of features impacts the distance values;

	

25.11	FEATURE IMPORTANCE AND MODEL INTERPRETABILITY

We have d features and through optimization we determine W vector of d dimensionality, each element in W vector corresponds to respective features; 

If all features are independent:

Weight element values indicate how important the feature is;

	In kNN: we have forward feature selection;

		Forward feature selection is algorithm independent can be applied for NB and LR

	In NB: Probability values give feature importance (Conditional probabilities)

	In Logistic Regression: Weights can be used to determine feature importance;

If absolute value of weight is large then contribution of the corresponding feature is large, thus this feature is important;

25.12	COLLINEARITY OF FEATURES

	Feature importance interpretation is done from weight vectors assuming independence;

	If there is co-linearity then we cannot interpret feature importance from weight vector;

	Two features are collinear if a feature can be expressed as a function of other feature;

Multi collinear feature is a feature that can be expressed as a function of multiple features;

Weight vectors are impacted by multi collinear features;

To use Weight vector for interpreting feature importance then we need to remove multi collinear features; 

Multi collinear feature can be determined by adding noise to features; if we add small noise to the values that is perturbation to the feature and if the weight vector varies (after training)a lot then the features are multi collinear; we then cannot use W vector for feature importance;

Multi collinear test is mandatory;

25.13	TRAIN & RUN TIME SPACE & TIME COMPLEXITY

Training Logistic Regression: Solving the optimization problem using Stochastic Gradient Descent;

Train time:

	Time complexity: O(nd)

Run time:

	Space: O(d)

	Time: O(d)

Logistic Regression is applicable for low latency applications; popular algorithm for internet companies, memory efficient at run time;

If dimensionality is large, L1 regularization inducing Sparsity can be applied to reduce run time complexity; As the hyper parameter increases Sparsity also increases; this achieves both a suitable bias variance trade off and the low latency requirements;

As λ increases, bias increases, latency decreases and Sparsity increases; but the model is just a working model not an optimized model as regularization and Sparsity are induced;

25.14	REAL WORLD CASES

	Decision surface: Linear Hyperplane

	Assumption: data is linearly separable or almost linearly separable;

	We also have good interpretability;

Imbalanced data: Upsampling or down sampling; with imbalanced data we will have incorrect hyper planes that can optimize the loss function where the hyper plane can make all the data points lie on one side optimizing the loss function for majority class;

Outliers: due to sigmoid function the impact is less;

We can compute W from Dtrain through training and generate distances through WTxi; the points that have large distances are outliers can be removed to make Dtrain free from outliers; the model is retrained on the outlier free dataset to get final Weight vector

	Mssing values: mean, median or model imputation;

	Multi class classification: One Vs Rest and Max entropy/Softmax/Multinomial LR models

Input similarity matrix: Extension of LR, Kernel LR (SVM) can be trained on similarity matrix;

Best case:

Data is linearly separable, low latency, faster to train; high dimensionality (higher chance that data is linearly separable and we can use L1 regularization)

Worst case:

	Non linear data; 

25.15	NON-LINEARLY SEPARABLE DATA & FEATURE ENGINEERING

	We need to transform features to make it linearly separable;

A circularly separable data set can be transformed into linearly separable data set by transforming features into a quadratic space;

F12 \+ F22 \= R2 is not linear in F1 and F2 but linear in F12 and F22;

Most important aspects of applied AI is:

1. Data analysis and Visualization  
   2. Feature Engineering (the difference between candidates, Deep Learning almost automates feature engineering)  
   3. Bias variance trade off

   We can use multiplication of features, polynomials, trigonometric Boolean, exponential or logarithmic transformations for making data linearly separable and even function combinations of several features;

25.16	CODE SAMPLE: LOGISTIC REGRESSION, GRIDSEARCHCV, RANDOMSEARCHCV

	Load libraries and datasets

C increases, overfitting happens;

Sklearn.linear\_model.LogisticRegression; sklearn.model\_selection.GridSearchCV

![][image131]

GridSearchCV will train for 5 hyper parameter values;

Experiment to understand different behavior of the models;

As hyper parameter value changes we can have varying weights;

25.17	EXTENSIONS TO LOGISTIC REGRESSION: GENERALIZED LINEAR MODELS (GLM)

	![][image132]

	Link: [http://cs229.stanford.edu/notes/cs229-notes1.pdf](http://cs229.stanford.edu/notes/cs229-notes1.pdf) 

Chapter 26:	 LINEAR REGRESSION

26.1	GEOMETRIC INTUITION OF LINEAR REGRESSION

Logistic Regression: A two class classification technique

Linear Regression: Find a hyper plane that best fits the training data (continuous variable data)

	yi \= WTxi \+ b

![][image133]

π: best fit plane; the points that are not on the hyper plane have errors due to incorrect prediction;

	error \= true value – predicted value

Task: Minimize the sum of errors across the training data;

26.2	MATHEMATICAL FORMULATION

	![][image134]

As there are positive and negative valued errors we need to make the values free from signs; we can use squared error;

Formulation:

	(W\*, W0) 	\= argmin summ (i \= 1 to n) (yi – ypred)2

			\= argmin summ (i \= 1 to n) (yi –  (WTxi \+ W0))2

Regularization:

	(W\*, W0) 	\= argmin summ (i \= 1 to n) (yi –  (WTxi \+ W0))2 \+ λ ||W||2

We can also use L1 regularization;

Loss minimization:

For classification: Zi \= yi f(xi) for logistic regression we have sigmoid function, we can also have step function or hinge loss which form other ML algorithms;

For regression: Zi \= yi – f(xi); f(xi) \= WTxi \+ W0; and the loss function is squared loss;

Terminology:

1\. Linear regression with L2 regularization is also referred to as "Ridge Regression"   
2\. Linear regression with L1 regularization is also referred to as "Lasso Regression".   
3\. Linear regression with l1+L2 regularization is also referred to as "Elastic Net Regression".

Regularization in Linear Regression:

For logistic regression we introduced regularization to constrain weight elements reaching infinity;

In real world we round up the values to an easily understandable decimal; this results in small errors even though we know the underlying relationships;

Without regularization we will have noise propagated to the weight values; to reduce the effect of noise in features on the weight vector we introduce regularization;

26.3	REAL WORLD CASES

	No problem of imbalanced data;

Feature Importance and Interpretability: same as for Logistic Regression; we can use weight vector elements if the data does not have multi collinear features;

Outliers: In logistic regression we have sigmoid function that is squashing and limiting the impact of outliers;

In Linear Regression we will have squared loss; to remove outliers we can compute distances of a Hyperplane that best fitted on the training set; the data points that are very far from the hyper plane are removed and the hyper plane is regenerated to fit the outlier free data set; iterable up to satisfaction;

Outliers impact the model heavily; this technique of iterated removal of outliers from model training is called RANSAC;

26.4	CODE SAMPLE FOR LINEAR REGRESSION

	Boston housing dataset; load data and split data; do EDA and Feature engineering;

	Fit Linear Regression on the train data;

![][image135]

![][image136]

Chapter 27:	 SOLVING OPTIMIZATION PROBLEMS

27.1	DIFFERENTIATION

	y \= f(x)

	Differentiation of y with respect to x \= dy/dx \= df/dx \= y’ \= f’

	dy/dx \= change in y due to change in x \= (y2 – y1)/ (x2 – x1) \= slope of the tangent to f(x)

	![][image137]

![][image138]

27.2	ONLINE DIFFERENTIATION TOOLS

	Link: [https://www.derivative-calculator.net](https://www.derivative-calculator.net) 

27.3	MAXIMA AND MINIMA

	Maxima: The maximum value a function takes;

	Minima: The minimum value a function takes;

	Slope at maxima / minima / saddle point \= 0 \= df/dx

	F(x) \= log(1 \+ exp(ax))

F’(x) \= a exp(ax)/(1 \+ exp(ax))

Most of the functions cannot be readily solved; we will use SGD to solve optimization problems;

27.4	VECTOR CALCULUS: GRAD

Differentiation of vectors results in a vector; we will be applying element wise partial differentiation;

Logistic loss:

	![][image139]

Solving the logistic loss is hard and thus we can use gradient descent technique

27.5	GRADIENT DESCENT: GEOMETRIC INTUITION

Iterative algorithm; initially we make a guess on the solution and we move towards the solution iteratively through solution correction;

Slope reaches zero when arriving at the optimum;

Gradient Descent:

1. Pick an initial point \= xold randomly  
   2. Xnew\= xold – r\* \[df/dx\]xold (r \= step size) with this gradient descent moves towards optimum;  
      3. Step 2 is update step which is iterated unless xnew \~ xold;   
      4. In addition the gradient decreases at each iteration; first iteration update will be large, then successive updates decreases until optimum is reached; the update reduces as slope reduces

27.6	LEARNING RATE

	Gradient Descent Update equation:

If learning rate does not reduce, gradient descent can jump over the optimum and this can be an iterative jump over where the algorithm does not reach the optimum; we are having oscillations without convergence;

We should reduce step size that is the learning rate is reduced at every iteration such that the convergence is guaranteed;

27.7	GRADIENT DESCENT FOR LINEAR REGRESSION

	![][image140]

	With Gradient descent we need to compute the updates over all data points which is expensive;

27.8	SGD ALGORITHM

	Linear regression:

		Gradient descent update:

			**Wj+1 \= Wj – rs \* summ(i \= 1 to n)(-2xi)(yi – WTj xi)**

			**Wj+1 \= Wj – rs \* summ(i \= 1 to k)(-2xi)(yi – WTj xi)**

		Changed iteration over n to k where we compute for k random points;

As we pick k random points we call this Gradient Descent as Stochastic Gradient Descent;

As k reduces from n to 1 the number of iterations required to reach optimum increases; At each iteration we have k and r changing; k is randomly picked between 1 and n which is called as batch size; r is reduced at each iteration progressively;

SGD is efficient; we are adding randomness to Gradient Descent to reduce time complexity at run time;

27.9	CONSTRAINED OPTIMIZATION & PCA

	![][image141]

	UTU is constraint for the optimization objective function;

	![][image142]

	Finding optimum under constraints:

		We need to modify the objective function using Lagrangian Multipliers;

	![][image143]

![][image144]

	λ and μ are Lagrangian Multipliers which are \>= 0

We can get solution for optimization by solving the partial derivatives of Lagrangian function with respect to x, λ and μ.

PCA can be reformulated as:	

![][image145]

	Su \= λu, then λ \= Eigen value and u is the Eigen vector of S;

27.10	LOGISTIC REGRESSION FORMULATION REVISITED

	![][image146] ![][image147]

	Lagrangian equivalent: L \= logistic loss – λ(1 \- WTW)

	Regularization is imposing an equality constraint on the logistic loss function;

Regularization in optimization formulation can be interpreted using equality constrained Lagrangian Multiplier;

27.11	WHY L1 REGULARIZATION CREATES SPARSITY?

	L1 Regularization induces Sparsity into weight vector; 

Sparsity implies that most of the elements of the weight vector are zero;

In optimization formulation for comparison Loss and λ can be ignored as they are same for both L1 and L2 regularizations;

L2 formulation has: min W for (W12\+W22\+ …. \+ Wd2): this is a parabola;

L1 formulation has: min W for (|W1|+|W2|+ …. \+ |Wd|): this is a v shaped curve;

L2(W1) \= W12; dL2/dW1 \= 2\*W1: derivative is a straight line

L1(W1) \= |W1|; dL1/dW1 \= \+1 or \-1: derivative is a step function

W1j+1 \= W1j – r(dL2/dW1)W1j; for both L1 and L2

Let W1 is positive (as similarly we can do for W1 negative)

For L2: W1j+1 \= W1j – r\*(2\*W1j)

For L1: W1j+1 \= W1j – r\*1

L2 updates occurs less when compared to L1 updates as we reach closer to optimum; that is the rate of convergence decreases because in L2 regularization we have 2 \* W1 \*r which is less than r; 

![][image148]

This happens because L1 derivative is constant and L2 derivative is not constant;

The chance of weights reaching 0 is more for L1 regularization as the derivative is constant and independent of the previous weight value; L2 regularization has derivatives reducing as the derivative is dependent on the previous iteration weight value which is converging to optimal;

27.12	REVISION QUESTIONS

1. Explain about Logistic regression?   
2. What is Sigmoid function & Squashing?   
3. Explain about Optimization problem in logistic regression.   
4. Expalain Importance of Weight vector in logistic regression.   
5. L2 Regularization: Overfitting and Underfitting   
6. L1 regularization and sparsity.   
7. What is Probabilistic Interpretation: Gaussian Naive Bayes?  
8. Explain about Hyperparameter search: Grid Search and Random Search?  
9. What is Column Standardization?   
10. Explain about Collinearity of features?  
11. Find Train & Run time space and time complexity of Logistic regression?

Chapter 28:	 INTERVIEW QUESTIONS ON LOGISTIC REGRESSION AND LINEAR REGRESSION

28.1	QUESTIONS & ANSWERS

1. After analysing the model, your manager has informed that your regression model is suffering from multicollinearity. How would you check if he’s true? Without losing any information, can you still build a better model?(https://google-interview-hacks.blogspot.in/2017/04/after-analyzing-model-your-manager-has.html)  
2. What are the basic assumptions to be made for linear regression?(https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/geometric-intuition-1-2-copy-8/)  
3. What is the difference between stochastic gradient descent (SGD) and gradient descent (GD)?(https://stats.stackexchange.com/questions/317675/gradient-descent-gd-vs-stochastic-gradient-descent-sgd)  
4. When would you use GD over SDG, and vice-versa?(https://elitedatascience.com/machine-learning-interview-questions-answers)  
5. How do you decide whether your linear regression model fits the data?(https://www.researchgate.net/post/What\_statistical\_test\_is\_required\_to\_assess\_goodness\_of\_fit\_of\_a\_linear\_or\_nonlinear\_regression\_equation)  
6. Is it possible to perform logistic regression with Microsoft Excel?(https://www.youtube.com/watch?v=EKRjDurXau0)  
7. When will you use classification over regression?(https://www.quora.com/When-will-you-use-classification-over-regression)  
8. Why isn't Logistic Regression called Logistic Classification?(Refer :https://stats.stackexchange.com/questions/127042/why-isnt-logistic-regression-called-logistic-classification/127044)

**External Resources:** 

1\. [https://www.analyticsvidhya.com/blog/2017/08/skilltest-logistic-regression/](https://www.analyticsvidhya.com/blog/2017/08/skilltest-logistic-regression/) 

2\. [https://www.listendata.com/2017/03/predictive-modeling-interview-questions.html](https://www.listendata.com/2017/03/predictive-modeling-interview-questions.html) 

3\. [https://www.analyticsvidhya.com/blog/2017/07/30-questions-to-test-a-data-scientist-on-linear-regression/](https://www.analyticsvidhya.com/blog/2017/07/30-questions-to-test-a-data-scientist-on-linear-regression/) 

4\. [https://www.analyticsvidhya.com/blog/2016/12/45-questions-to-test-a-data-scientist-on-regression-skill-test-regression-solution/](https://www.analyticsvidhya.com/blog/2016/12/45-questions-to-test-a-data-scientist-on-regression-skill-test-regression-solution/) 

5\. [https://www.listendata.com/2018/03/regression-analysis.html](https://www.listendata.com/2018/03/regression-analysis.html) 

