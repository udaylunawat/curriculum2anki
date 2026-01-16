**Module 4: Machine Learning – II (Supervised Learning Models)**

**Chapter 29:	Support Vector Machines (SVM)**

**29.1	Geometric Intuition**

	SVM – Popular Machine Learning Model – classification and regression problems

Let data points be as:

	![][image149]

The data points can be separated with many hyper-planes. Say, with 

In logistic regression, if a data point is very close to the hyper plane, then the probability of the point belonging to a class is very close to 0.5. For points away from hyper plane the probability will be close to 1\. 

The objective will be to separate the data points with a hyper plane which is as far as possible from the data points. 

π2 is better than  π1 as the hyper plane is as far as possible from all data points. Thus, π2 can be taken as a margin maximizing hyper plane. 

Take π\+ and π\- parallel to π2 such that it touches first data point of the respective group. We get a margin and with SVM we maximize the width of this margin. 

	**Objective: find π such that dist(π\+, π\-) (= margin) is maximum**

With this we will have lesser misclassifications and better generalization (s margin increases).

Support Vector: Say we have the margin maximizing hyper plane π, and we have π\+ and π\-, the data points through which these margin planes pass through are called support vectors. We have two support vectors in the following plot.

![][image150]

For classification we will use π, the margin maximizing hyper plane.

Alternative Geometric Intuition of SVM: Using Convex hulls:

![][image151]

Build smallest convex hulls for each set of data points, find the shortest line connecting these hulls, and draw a perpendicular bisector to this line to get the margin maximization hyper plane. 

**29.2	Mathematical derivation**

	![][image150]

Find hyper plane that does margin maximization; Let this hyper plane be WTX \+ b, where W is perpendicular to the hyper plane.

For π\+: WTX \+ b \= 1

For π\-: WTX \+ b \= \- 1

Margin \= 2/||W||

Objective: 	Find (W\*, b\*) \= argmax (W,b) 2/||W||

For yi (WTX \+ b) \>= 1 (2 cases: \+1 x \+1 and \-1 x \-1) this is the constraint

	This works very well for linearly separable data points. 

If the data points are mixed to some extent or some data points lie in the margin or some positive data points lie on negative side of the hyper plane. We will never find a solution for these cases. The above constraint imposes a hard margin on the SVM model. Thus the margin needs to be relaxed. 

Introduction of a new variable which tells how far the data point is in the opposite direction of the hyper plane. This variable introduced is ξ i , which is equal to zero if the data point is on the correct direction of the hyper plane and is equal to the distance from the hyper plane if it is on the other side. 

(W\*, b\*) \= argmin (W,b) ((||W||/2) \+ C \* (1/n) sum(ξ i)) | yi (WTX \+ b) \>= 1 \- ξ I for all i

This is an objective function which says to maximize the margin and minimize errors. 

As C increases, tendency to make mistakes decreases, overfitting on training data, high variance. As C decreases underfitting occurs, high bias. The above objective formulation is for soft margin optimization. 

**29.3	Why we take values \+1 and \-1 for support vector planes**

It does not matter, we can take \+k and –k and also want both the support vector planes be equidistant from the margin maximizing hyper plane. \+1 and \-1 are chosen for mathematical convenience.

**29.4	Loss function (Hinge Loss) based interpretation**

	Logistic Regression: 	Logistic loss \+ reg

	Linear Regression: 	Linear loss \+ reg

	SVM: 			Hinge loss \+ reg

	Zi \= yi f(xi) \= yi (WTXi \+ b)

	For 0-1 loss: if Zi \> 0 is correctly classified; if Zi \< 0 is incorrectly classified

	Hinge Loss: 

![Image result for hinge loss][image152]

	If Zi \>= 1, hinge loss \= 0;

	If Zi \<= 1, hinge loss \= 1 \- Zi;

Hinge loss \= max (0, 1 \- Zi)

Geometric formulation:

ξi \= 0 for correctly classified data points

For incorrectly classified data points:

![][image153]

ξi \= dist for xj to the correct hyper plane \= max(0,1- Zj)

Soft-SVM: min(W,b) ||W||/2 \+ C sum(ξi) such that ( 1- yi (WTXi \+ b)) \<= ξi

C increases \- Overfitting

Loss-minimization: min (W,b) sum(max(0, 1-yi(WTXi \+ b)) \+ λ||W||2

λ increases – Underfitting

(C when multiplied with loss function and λ for regularizing term)

**29.5	Dual form of SVM formulation**

	![][image154]

	Dual form:

		![][image155]

	αi \> 0 for support vectors and αi \= 0 for non support vectors

	The dual formulation of SVM does not depend only on Support Vectors.

	![][image156]

	SVM can include similarity between data points using the Dual form.

	K \- Kernel function tells about similarity about the two data points.

Link:	[http://cs229.stanford.edu/notes/cs229-notes3.pdf](http://cs229.stanford.edu/notes/cs229-notes3.pdf)

**29.6	Kernel Trick**

	We have:

![][image157]

Without kernel trick the formulation is called as Linear SVM; Task: find margin maximizing hyper plane; Results look similar for Linear SVM and Logistic Regression;

	A simple hyper plane cannot separate all types of data clusters;

	Logistic regression works on transformed space of data points;

	Kernel-SVM can solve non-linearly separable datasets; 

	Kernel logistic regression also exists;

**29.7	Polynomial Kernel**

	For dataset such as:	

The dataset is transformed to square of features during Logistic regression modeling;

Polynomial kernel: K(x1, x2) \= (x1Tx2 \+ c)d; 

Quadratic kernel:

![][image158]

Kernelization can be thought of application of feature transformation internally;

In logistic regression feature transformation is performed explicitly;

Mercer’s theorem: Kernel trick converts d dim dataset to d’ dim data set s.t. d’ \> d;

This will cater for non-linearity in the dataset; 

Finding the kernel is the challenge;

**29.8	RBF-Kernel**

	General purpose and most popular Kernel: Radial Basis Function:

		![][image159]

		Numerator: distance squared; sigma is a hyper parameter for RBF kernel

1. As distance increases, K(x1, x2) decreases and reaches 0  
2. As sigma reduces, the tolerance to distances reduces, i.e. data points need to be very near to have similarity values non-zero

![][image160]

Dissimilarity : Distance :: Similarity : Kernel

As sigma increase, variance increases. Sigma increment is similar to increment in K.

Qs: How RBF Kernel is related to KNN? How is SVM related to Logistic Regression? How is Polynomial Kernel related to feature transformations?

RBF-SVM is similar to KNN because of sigma;

KNN: Stores all k-pts;

RBF-SVM: only require Support Vectors;

RBF-SVM is an approximation to Kernel;

We have C and Sigma as hyper parameters;	

**29.9	Domain specific Kernels**

We have seen Polynomial and RBF kernels: RBF-SVM \~ KNN and RBF kernel is a general purpose kernel; Kernel-trick \~ Feature Transformation – Domain specific

String Kernels: Text classification

Genome Kernels: Genome prediction

Graph-based Kernels: Graph problems

RBF can be used as a general purpose kernel;

**29.10	Train and run time complexities**

	Train by using SGF, 

	Use Sequential Minimal Optimization (SMO) for specialized algorithm (dual formulation)

		Ex: libSVM

	Training Time: O(n2) for kernel SVMs

		SVMs are typically not use when ‘n’ is large

	Run Time: O(kd); k – Number of Support Vectors, d \- dimensionality

**29.11	nu-SVM: control errors and support vectors**

	C-SVM: C\>=0

	nu-SVM: nu belongs to \[0,1\] 

		Fraction of errors \<= nu; fraction of Support Vectors \>= nu

	Nu is upper bound on fraction of errors and lower bound on number of Support Vectors

**29.12	SVM Regression**

![][image161]

Epsilon is the hyper parameter;

With kernels we can fit non-linear datasets;

If epsilon is less; errors are low, overfitting increases;

RBF-Kernel SVR: similar to KNN Regression where nearest data points are calculated and a mean of the values of data points is calculated;

Links: 	[https://alex.smola.org/papers/2004/SmoSch04.pdf](https://alex.smola.org/papers/2004/SmoSch04.pdf), [https://youtu.be/kgu53eRFERc](https://youtu.be/kgu53eRFERc) 

**29.13	Cases**	

	Feature Engineering: 	Kernelization

	Decision Surfaces: 	Non- Linear surfaces

	Similarity or distance function: Easily kernelizable

	Interpretability and feature importance: no provisions with kernel SVM 

	Outliers have little impact as Support Vectors are important; 

\- RBF with a small sigma can be impacted with outliers as in case of kNN with small k

Bias Variance: 	As C increases – overfitting, high variance;

		As C decrease – underfitting, high bias

Large dimensionality: SVMs perform well with large d

If dataset size or number of Support Vectors is large, then Support Vectors are not zpreferred; Instead Logistic Regression is used

**29.14 	Code Sample**

	[https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)

**29.15	Revision Questions**

1. Explain About SVM?   
2. What is Hinge Loss?  
3. Dual form of SVM formulation?  
4. What is Kernel trick?  
5. What is Polynomial kernel?  
6.  What is RBF-Kernel?   
7. Explain about Domain specific Kernels?   
8. Find Train and run time complexities for SVM?   
9. Explain about SVM Regression. ? 

Chapter 30: Interview Questions on Support Vector Machine

**30.1	Questions and Answers**

1. Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm and vice-versa. (https://datascience.stackexchange.com/questions/6838/when-to-use-random-forest-over-svm-and-vice-versa)  
2. What is convex hull?(https://en.wikipedia.org/wiki/Convex\_hull)  
3. What is a large margin classifier?  
4. Why SVM is an example of a large margin classifier?  
5. SVM being a large margin classifier, is it influenced by outliers? (Yes, if C is large, otherwise not)  
6. What is the role of C in SVM?  
7. In SVM, what is the angle between the decision boundary and theta?  
8. What is the mathematical intuition of a large margin classifier?  
9. What is a kernel in SVM? Why do we use kernels in SVM?  
10. What is a similarity function in SVM? Why it is named so?  
11. How are the landmarks initially chosen in an SVM? How many and where?  
12. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?  
13. What is the difference between logistic regression and SVM without a kernel? (Only in implementation – one is much more efficient and has good optimization packages)  
14. How does the SVM parameter C affect the bias/variance trade off? (Remember C \= 1/lambda; lambda increases means variance decreases)  
15. How does the SVM kernel parameter sigma^2 affect the bias/variance trade off?  
16. Can any similarity function be used for SVM? (No, have to satisfy Mercer’s theorem)  
17. Logistic regression vs. SVMs: When to use which one? (Let n and m are the number of features and training samples respectively. If n is large relative to m use log. Reg. or SVM with linear kernel, if n is small and m is intermediate, SVM with Gaussian kernel, if n is small and m is massive, create or add more features then use log. Reg. or SVM without a kernel)  
18. What is the difference between supervised and unsupervised machine learning?

External Resource: [https://www.analyticsvidhya.com/blog/2017/10/svm-skilltest/](https://www.analyticsvidhya.com/blog/2017/10/svm-skilltest/)

**Chapter 31: Decision Trees**

**31.1	Geometric Intuition of decision tree: Axis parallel hyper planes**

	Classification and Regression methods:

		KNN			Instance based method

		Naïve Bayes		Probabilistic model

		Logistic Regression	Geometric model: Hyper plane based separation

		Linear Regression	Geometric model: Hyper plane based separation

		SVM			Geometric model: Hyper plane based separation

	Decision Trees:	Nested If – else classifier

	Ex: 

![][image162]![][image163]

Root – node: 		First node

Leaf node:		Terminating node

	Non- Leaf node:	Decision nodes

	Decision Trees are extremely interpretable

	![][image164]

	All hyper planes are axis parallel;

**31.2	Sample Decision Tree**

	![][image165]

	Given any query point predicting its class label is straight forward.

**31.3	Building a Decision Tree: Entropy**

	Task: Given a dataset build a Decision Tree

	Entropy: occurs in physics, Information Theory, etc.

	Given random variable y with k instances y1, y2, y3, …., yk

	Entropy(y) \= H(y) \= \- summ (p(yi)\*log2 (p(yi)))

	Let y \= Play Tennis: y \= {0, 1}

		P(y=1) \= 9/14, P(y=0) \= 5/14

		H(y) \= \- (9/14) log2(9/14) \- (5/14) log2(5/14)

		Case 1: 99% y+ and 1% y-: Entropy \= \-0.99lg 0.99 – 0.01 lg 0.01 \= 0.0801

		Case 2: 50% y+ and 50% y-: Entropy \= \- 0.5 log 0.5 – 0.5 log 0.5 \= 1

		Case 3: 0% y+ and 100% y-: Entropy \= 0

		

		Entropy has maximum value when all classes are equally probable;

		Entropy indicates a degree of randomness in the data.

**31.4	KL Divergence**

	Kullback Leibler Divergence:

	Two distributions P and Q:

		![][image166]     ![][image167]

	Dist(P, Q): is small when two distributions are similar and closer;

KS statistic: Maximum gap between P’ and Q’: KS statistic is not differentiable thus cannot be used as a part of loss function;

![][image168]

	Summation or integration depends on type of random variable (discrete or continuous);

	KL value will be increasing when there is dissimilarity in P and Q increasing;

	KL statistic is differentiable

**31.5	Building a Decision Tree: Information Gain**

	Information Gain \= Entropy(parent) – Weighted Average of Entropy of child nodes

	![][image169]

	E(Y, Outlook \= Sunny) 		\= \- (2/5) \*lg(0.4) – 0.6\*lg(0.6) \= 0.97

	E(Y, Outlook \= Overcast) 	\= \- 1\*lg(1) \- 0 			\= 0

	E(Y, Outlook \= Rainy) 		\= \- (3/5) \*lg(0.6) – 0.4\*lg(0.4) \= 0.97

	

	Information\_Gain	\= Entropy(parent) – Weighted average of entropy of child nodes

				\= 0.94 – ((5/14)\*0.97 \+ 4/14 \* 0 \+ (5/14)\*0.97)

**31.6	Building a decision tree: Gini Impurity**

	Gini Impurity similar to entopy;

	Advantages over Entropy:  

		IG(Y) \!= IG(Y)

		IG(Y) \= 1 – sum (p(yi))2

	Case 1: P(y+) \= 0.5 and P(y-) \= 0.5, IG \= 1 – (0.52 \+0.52) \= 0.5

	Case 2: P(y+) \= 1 and P(y-) \= 0, IG \= 1 – (1 – 0\) \= 0

		![][image170]

Calculating logarithmic values is computational expensive that computing squares, thus as Entropy and Gini impurity behave similar to each other, Gini impurity is preferred for its compute efficiency.

**31.7	Building a decision tree: Constructing a Decision Tree**

	Dataset: The play tennis case:	

1. At top we have dataset with 9+ and 5-: H(Y) \= 0.94  
2. Outlook: H(Y, Sunny), H(Y, Overcast), H(Y, Rainy), weighted \= 0.69, IG \= 0.94 – 0.69 \= 0.25  
3. Temp: High, Mild, Cool: weighted \= \_\_, IG \= \_\_  
4. Humidity: IG \= \_\_  
5. Windy: IG \= \_\_

   Result: 

   ![][image171]

   Stopping condition for growth of decision tree: Pure nodes or depth, if pure node not reached then use majority vote or mean;

   Depth of tree is a hyper parameter determined by cross validation.

**31.8	Building a decision tree: Splitting numerical features**

	Task: Split node: using Entropy or Gini Impurity and Information Gain

	Categorical feature is straight forward;

For Numerical features, sorting of data points is done based on feature values: and numerical comparisons are done to split nodes; 

Evaluate all possible threshold splits; Compute Information Gains;

**31.9	Feature Standardization**

	Mean centering and scaling;

	Threshold depends on order and does not depend on actual values in Decision Trees, No 	need for feature standardization

**31.10	Building a decision tree: Categorical Features with many possible values**

Ex: Pin code/ Zip code: Numerical but not comparable, thus these are taken as categorical features and there could be a lot of categories in each feature; which will result in data sparsity or tree sparsity

Transforming this into Numerical features will be useful: 

Calculate conditional probabilities of class labels given a certain categorical value

**31.11	Overfitting and Underfitting**

As depth increases, possibility of having very few data points at a node; possibility of overfitting increases and interpretability of the model decreases; as depth is less underfitting happens.

With Depth \= 1 the decision tree is called a decision stump,

Depth is determined using Cross Validation

**31.12	Train and Run time complexity**

	Train time: 		\~ O (n \* log n \* d)

	Space at run time:	\~ O (nodes)

	Time at run time:	\~ O(depth) \= \~ O(1)

	Decision Trees are suitable: Large data, small dimensionality, low latency requirement

**31.13	Regression using Decision Trees**

	Information Gain for classification; MSE or MAE for regression;

	yi\_hat \= mean/ median at each node;

	Take weighted errors for splitting, whichever feature gives lowest error is selected;

**31.14	Cases**

	Imbalanced data impacts Information Gain calculations; 

	Large dimensionality increases time complexities

Categorical features avoid one hot encoding; Use response encoding or Label encoding or probabilistic encoding

	Decision Trees cannot work with similarity matrix;

Multi-Class Classification: One versus Rest is not required as Entropy takes all categories while calculating them

Decision Surfaces: Non- Linear Axis parallel Hyper Cuboids

Feature Interactions: Between depths, F1 \< 2 AND F2 \> 5 are available or logical feature interactions are inbuilt to Decision Trees: Advantageous

Outliers: As depth is large, overfitting occurs, outliers make Trees unstable;

Interpretability: Nested if-else conditions; DTs are extremely interpretable;

Feature Importance: For every feature we get Information Gain or can sum up reductions in Entropy due to this feature. The more reduction, the more important;

**31.15	Code Samples**

	Link: [http://scikit-learn.org/stable/modules/tree.html](http://scikit-learn.org/stable/modules/tree.html) 

	Visualize: Dataset using Graphviz

	DTClassifier: Gini Impurity or Entropy

	DTRegressor: MSE or MAE (mean/ median IG)

**31.16	Revision Questions**

1. How to Build a decision Tree?  
2. What is Entropy?  
3. What is information Gain?  
4. What is Gini Impurity?  
5. How to Constructing a DT?  
6. Importance of Splitting numerical features?  
7. How to handle Overfitting and Underfitting in DT?  
8. What are Train and Run time complexity for DT?  
9. How to implement Regression using Decision Trees?

**Additional Material:**

**Decision Trees (DTs)** are a non-parametric supervised learning method used for [classification](https://scikit-learn.org/stable/modules/tree.html#tree-classification) and [regression](https://scikit-learn.org/stable/modules/tree.html#tree-regression). The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

For instance, in the example below, decision trees learn from data to approximate a sine curve with a set of if-then-else decision rules. The deeper the tree, the more complex the decision rules and the fitter the model;

![../\_images/sphx\_glr\_plot\_tree\_regression\_0011.png][image172]  
Some advantages of decision trees are:

* Simple to understand and to interpret. Trees can be visualised.  
* Requires little data preparation. Other techniques often require data normalisation, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.  
* The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.  
* Able to handle both numerical and categorical data. Other techniques are usually specialised in analysing datasets that have only one type of variable. See [algorithms](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms) for more information.  
* Able to handle multi-output problems.  
* Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.  
* Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.  
* Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

  The disadvantages of decision trees include:

* Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.  
* Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.  
* The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.  
* There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.  
* Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

**Chapter 32: Interview Questions on decision trees**

**32.1	Question & Answers**

1. You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. Can this happen? Why?(Refer :https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)  
2. Running a binary classification tree algorithm is the easy part. Do you know how does a tree splitting takes place i.e. how does the tree decide which variable to split at the root node and succeeding nodes?(Refer:https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/)

[https://vitalflux.com/decision-tree-algorithm-concepts-interview-questions-set-1/](https://vitalflux.com/decision-tree-algorithm-concepts-interview-questions-set-1/)

**Chapter 33: Ensemble Models**

**33.1	What are ensembles?**

	In Machine Learning, Multiple models are brought together to build a powerful model.

The multiple models may individually perform poorly but when combined they become more powerful, this combination to improve performance of several baseline models is called ensemble.

Ensemble (four of the Ensemble methods are as following):

* Bagging (**B**ootstrapped **Ag**gregation)  
  * Boosting  
  * Stacking  
  * Cascading

  High performing, Very powerful, Very useful in real world problems;

  Key aspect: the more different the base line models are, the better they can be combined

    Problem: 	Models can be thought of as experts in different aspects of the problem and when brought together we will get a better solution

**33.2	Bootstrapped Aggregation (Bagging) Intuition**

	Sample with replacements and train models on different samples: Bootstrapping

	Aggregation combining these different training models with a majority vote;

	At run time, the query point is passed to all the models and a majority vote is applied

Classification: Majority, Regression: Mean/Median

None of the models are trained on the whole dataset;

![][image173]

Bagging can reduce variance in modeling due to aggregation without impacting bias;

Bagging: Take Base models having low bias and high variance and aggregate them to get a low bias and reduced variance model

Decision Trees with good depth are low bias and high variance models: Random Forest

**33.3	Random Forest and their construction**

Bagging with Decision Trees \+ Sampling (Feature Bagging)

Bootstrap sampling \= Row sampling with replacement: Create k samples;

Column Sampling \= select a subset of features randomly

Xn x d to Xm x d’ where m\<n and d’ \< d; m and d’ are also different between samples;

We will have k Decision Trees trained on these k samples and Aggregation (majority weight or mean/median) is applied on the output of the k base learners (DTs);

The data points that are removed during selection for each sample (out of bag points) can be used for cross validation for the corresponding model (out of bag error is evaluated to understand performance of each base learner)

Random Forest: Decision Tree (reasonable depth) base learner \+ row sampling with replacement \+ column sampling \+ aggregation (Majority vote or Mean/Median)

Tip: Do not have a favorite algorithm

**33.4	Bias – Variance tradeoff**

Random Forest: low bias (due to base learners)

	As number of base learners increases, variance decreases (due to aggregation)

	Bias (individual models) \~ Bias (Aggregation)

	d’ / d \= column sampling ratio and m/n \= row sampling ratio

		If these two ratios are decreased, variance decreases

These ratios are fixed initially and the number of base learners is determined through hyper parameter tuning (cross validation)

**33.5	Train and run time complexity**

K base learners;

Train time complexity: O(n \* lg n \* d \* k); Trivially parallelizable, each base learner can be trained on different core of the computer;

Run time complexity: O(depth \* k): low latency

Space complexity at run time: O(Decision Trees \* k)

**33.6	Bagging: code sample**

Code: sklearn.ensemble.RandomForestClassifier()

**33.7	Extremely randomized trees**

For numerical features we apply thresholds (using sorting) to get Information Gain;

With extremely randomized trees, random selection of threshold values is done to check for Information Gain; In Decision Trees, we check for each of the values as threshold

Extreme Trees: Col sampling \+ Row sampling \+ Randomization when selecting thresholds \+ Decision Trees \+ Aggregation

Randomization is used to reduce variance; bias does not increase due to randomness;

**33.8	Random Forest: Cases**

Random Forest: Decision Tree \+ (Row samp. \+ Col samp \+ agg) (used to reduce variance)

1. Decision Trees cannot handle large dimensionality; with categorical features having many categories (fails even with one hot encoding), DTs and Random Forests have similar cases except for bias – variance tradeoff and feature importance  
   2. Bias – Variance Tradeoff of Random Forest is different from Decision Trees;

      	In Decision Trees we change depth;

      In Random Forests: \# of base learners is a hyper parameter and each decision tree has sufficient depth

   3. Feature Importance:  
      In DT, Overall reduction in Entropy of IG because of this feature at various levels in the DT (single DT)  
      In RF, overall reduction is checked across all base learners for feature importance

**33.9	Boosting Intuition**

Bagging: High variance and low bias models: with randomization and aggregation to reduce variance

Boosting: take low variance and high bias models: use additive combining to reduce bias;

Given a dataset: 

Stage 0: Train a model using the whole dataset: This model should have high bias and low variance: DT with shallow depth; Large Train error; 

Subtract: y\_i – the output of this Decision Tree;

Stage 1: In stage 1, we train a model on error of the previous model

	F1(x) \= a1 h0(x) \+ a1 h1(x) 

Stage k: Fk(x) \= summ (alpha\_i \* hi(x)): additive weighted model

Each of the models at each stage is trained to fit on the residual error at the end of the previous stage

		Methods: Gradient Boosted DT and Adaptive Boosting

**33.10	Residuals, Loss functions and gradients**

	Fk(x) \= summ (alpha\_i \* hi(x))

		Alpha\_i coefficients and hi residuals;

	Residual: erri \= yi \- Fk(x) 

	Modelk+1 is trained on {xi, erri}

	ML models are trained by minimizing loss functions: 

			Logistic loss – classification

			Linear regression – Square loss

			SVM – Hinge loss

	L(yi, Fk(x)) : loss is defined on target and prediction at end of stage k

	Let the problem be a regression problem: The loss function is say squared loss

	L(yi, Fk(x)) \= (yi \- Fk(x))2 

	Derivative of L wrt to F is \-1\*2\*( yi \- Fk(x)) \= 2\* residual of last stage

- Negative gradient of loss function at the end of stage k is proportional to residual at that stage; negative gradient \~ pseudo residual (this allows to use any loss functions)  
- RandomForest are limited with loss functions, Gradient Boosted DTs permit use of any loss function

  Link: [https://youtu.be/qEZvOS2caCg](https://youtu.be/qEZvOS2caCg) 

  	For non squared loss functions: Is pseudo residual equal to negative gradient?

  	Log loss: negative gradient \= probability estimate – actual class

  	But pseudo residuals cannot be interpreted as residual / error always

**33.11	Gradient Boosting**

	**![][image174]**

**33.12	Regularization by shrinkage**

	As number of base models increase, overfitting occurs with increase in variance;

	![][image175]

	This controls overfitting; shrinkage is a hyperparameter;

**33.13	Train and Run Time complexity**

	Decision Trees: Train: O(n \* lg n \* d\*M)

	Random Forests are trivially parallelizable;

While GBDTs are trained sequentially;

Run time complexity: O(depth \* M)

Space complexity at run time: O(store each tree \+ gamma)

GBDTs used for low – latency applications

	They can minimize any loss function;

**33.14	XGBoost: Boosting \+ Randomization**

	Extension of GBDT and RF

**33.15	AdaBoost: geometric intuition**

	Used in Computer Vision for face detection;

	In GBDT we used Pseudo Residuals;

Adaptive boosting: At every stage you are adapting to errors that were made before; more weight is given to misclassified points

![][image176]

Adaboost is highly prone to outliers;

**33.16	Stacking models**

	An ensemble learning meta-classifier for stacking;

Individual classification models are trained based on the complete training set; then a meta-classifier is trained on the outputs – Meta features of the individual classification models in the ensemble. The individual models should be as different as possible.

Train all models independent of each other; given a data point query it is passed through all models and all the outputs are passed through a meta-classifier;

Using outputs of base models as Meta features for a new classifier;

Real World: Stacking is least used on real world problems due to its poor latency performance

**33.17	Cascading classifiers**

Used when cost of making a mistake is high; Sequential model definition each model working to classify datapointso ne after the other; whichever data points are perfectly classified with high probability by a model are not shown to the next model; generally at the end of all the models a human is placed to make predictions on the data points the cascaded models are unsure about;

Used for fraud detections;

**33.18	Kaggle competitions vs Real World Case Studies**

	Bagging, Boosting, Stacking, Cascading;

	Kaggle competitions care about only one metric;

Due to this complex ensembles are generally experimented with and these models are not useful for low latency, training time and interpretability;

- Brilliant ideas for feature engineering are available;

**33.19	Revision Questions**

1. What are ensembles?   
2. What is Bootstrapped Aggregation (Bagging)?   
3. Explain about Random Forest and their construction?   
4. Explain about Boosting?   
5. What are Residuals, Loss functions and gradients?   
6. Explain about Gradient Boosting?   
7. What is Regularization by Shrinkage?   
8. Explain about XGBoost?   
9. Explain about AdaBoost?   
10. How do you implement Stacking models?   
11. Explain about cascading classifiers? 

