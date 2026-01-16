**Module 7: Data Mining (Unsupervised Learning) and Recommender Systems \+ Real – World Case Studies**

**Chapter 43:	Unsupervised Learning/ Clustering**

**43.1	What is Clustering?**

	Classification & Regression: Given a data point we have a target value;

	Clustering: No target values; Task is to group similar data points;

	Example:

		![][image183]

	Task: group similar points: we can have three groups here

Similarity: Points are close to each other in the same cluster and very far in different clusters

Measure of performance for Clustering Algorithms \= \_\_\_

For classification and regression we had metrics such as precision, MSE, etc.

Clustering: We have k-Means, Hierarchical Clustering, DBSCAN

**43.2	Unsupervised Learning**

Classification and Regression: Supervised Learning: We had target variable to train the models

Semi- Supervised Learning: Dataset which has a small size of supervised data points and unsupervised data points;

**43.3	Applications**

	Data Mining;

Ecommerce (group similar customers based on their location, income levels, purchasing behavior, product history)

Image segmentation: Grouping similar pixels, apply ML algorithm to do object detection

Review analysis (text): Manual labeling is time consuming and expensive; Clustering can be applied: into 10k groups based on word similarities syntactic and semantic similarity), review and label each cluster (pick some points and check for labels), Pick points that are closer to cluster centre and avoid outliers

**43.4	Metrics for Clustering**

Dataset: Given x and no y;

A good clustering result:

	Intra cluster: Within a cluster;

	Inter cluster: Between clusters;

Intra cluster should be small and inter cluster should be large: This is how we can effectively measure the performance of the clustering algorithm

Dunn index: 

D \= min d(i,j) / max d’(k) 	\= minimum inter-cluster distance/ maximum intra-cluster distance

	For a good clustering result Dunn index should be large.

**43.5	K-Means: Geometric intuition, Centroids**

	K \= number of clusters \= hyper parameter;

For every cluster K Means assigns a centroid and groups the data points into clusters around the centroid, no point belongs to two clusters and every data point belongs to at least 1 cluster;

Centroid is the geometric mean point of the cluster data points;

K Means is a centroid based clustering scheme;

Data points are assigned to a cluster based on nearness to a centroid;

**43.6	K-Means: Mathematical formulations: Objective function**

![][image184]

Above formulation has Exponential time complexity: NP hard problem;

An approximation algorithm is used to solve the optimization problem;

**43.7	K-Means Algorithm**

	Lloyd’s Algorithm:

1. Initialization: Randomly pick k data points from the Dataset and call them centroids  
2. Assignment: For each point in the Dataset, select the nearest centroid through the distance and add this data point to the centroid corresponding cluster  
3. Re-compute centroid: Update centroid to the mean of the cluster.

   ![][image185]

4. Repeat step 2 and step 3 until convergence

**43.8	How to initialize: K-Means++**

K-Means have initialization sensitivity: the final result changes when the initialization is changed

Link: [https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf](https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf) 

	![][image186]

Repeat k means with different initializations: pick best clustering based on smaller intra distances and larger inter cluster distances;

K-Means++: 	random init is replaced with smart init;

Pick first centroid randomly; pick further centroids with a probability proportional to the distance of the point from the previous nearest centroid; 

Probabilistic approach is preferred over deterministic approach where we can pick a point that is far from all centroids, to avoid outliers as K-Means++ gets impacted by outliers and probabilistic approach mitigates this effect;

**43.9	Failure Cases/ Limitations**

	Clusters with different sizes, densities and non-globular shapes and presence of outliers

	K-Means tries to cluster equal number of data points in each cluster;

	![][image187]

	Different densities:

	![][image188]

	K-Means tries to get similar densities across all clusters

	

	Non-Globular shapes:

	![][image189]

Use larger k to get many clusters and put together different clusters and avoid above problems;

Cluster evaluations are difficult as there is no supervised learning;

**43.10	K-Medoids**

	Centroids may not be interpretable for example bag of words vectorization of text;

Instead of giving centroids computed using means, if we output an actual data point that is just a review will be more interpretable; this review is a Medoid;

Each centroid is a data point \= Medoid for interpretation;

Partitioning around Medoids (PAM):

1. Initialization: Similar to K-Means++ (probabilistic approach)  
2. Assignment of data points to a cluster (closest Medoid)  
3. Update/ recomputed: No mean approach; Swap each Medoid with a non-Medoid point; If loss decreases keep the swap else undo swap; (K-Means formulation, minimizing distances) if swap is successful redo step 2, if a similarity matrix or distance matrix is given, K-Medoids can be easily implemented; K-Medoids is Kernelizable

   

**43.11	Determining the right K**

	Elbow method: plot loss function vs k; select k with elbow method;

**43.12	Code Samples**

	sklearn.cluster.KMeans

	Arguments: n\_clusters, n\_init

**43.13	Time and space complexity**

	K-Means: O(n\*k\*d\*i) \~ O(nd)

		n points, k clusters, d dimensionality, i iterations;

	Space: O(nd \+ kd) \~ O(nd)

**Chapter 44: Hierarchical clustering Technique**

**44.1	Agglomerative & Divisive, Dendrograms**

Agglomerative clustering: Takes each data point as a cluster and groups two nearest clusters into one cluster until the number comes to k; Stage wise Agglomeration of clusters;

Divisive starts in the reverse order: Takes all data points into 1 cluster and divides clusters stage be stage; division is a big question; Agglomerative is popular;

![][image190]Dendrogram

Link: [https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf](https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf)

**44.2	Agglomerative Clustering**

![][image191]

![][image192]

![][image193]

![][image194]

**44.3	Proximity methods: Advantages and Limitations**

	Inter cluster Similarity:

		Min: similarity \= minimum distance between clusters

		Max: maximum distance between clusters

Group Average: Summ(sim(pairs))/ (Multiplication of number of points in all clusters)

Distance between centroids;

Due to similarity presence in the calculations: Min, Max, Avg can be kernelized

		![][image195]

	\+ Can handle non-globular distributions and \- is sensitive to noise and outliers;

![][image196]

\+ Less prone to outliers and noise

\- Tends to break large clusters, biased towards globular clusters

![][image197]

![][image198]

\+ Less prone to outliers and noise

\- Biased towards globular clusters

Ward’s Method: squared distances are taken; everything else is similar to Group Avg

![][image199]

**44.4	Time and Space Complexity**

	Space: O(n2) n number of data points

	Time: O(n3)

**44.5	Limitations of Hierarchical Clustering**

	1\. No mathematical objective function we are solving in Hierarchical Clustering

	2\. Computational Complexity (Time and Space)

	3\. Once a technique is applied it cannot be undone

**44.6	Code Sample**

	sklearn.cluster.AgglomerativeClustering

	Hyperparameters: n\_clusters, linkage

**Chapter 45:	DBSCAN (Density based clustering) Technique**

**45.1	Density based clustering**

	K-Means: Centroid based

	Hierarchical: Agglomerative

	DBSCAN:	Dense regions: Clusters, Sparse: Noise: Separate dataset into dense regions from sparse regions

**45.2	MinPts and Eps: Density**

	Density at point P: Number of points in radius eps around p

	Dense region: Region which has minimum number of points MinPts in radius eps

**45.3	Core, Border and Noise points**

	Core point: if point has \>= Min pts in eps radius

Border point: point having \<= Min pts in eps radius and belongs to a neighborhood of core point

Noise point: None of above

	![][image200]

**45.4	Density edge and Density connected points**

	Density edge: Connection between two core points which are in a radius \<= eps

Density connected points: between two core points if there exists all density edges exist in the path between the core points

**45.5	DBSCAN Algorithm**

1\. 	For each data point: Label each point as core point, border pt or a noise pt (Given Minpts and Eps); This is done using Range Query (using kd-tree or kNN) 

2\. 	Remove all noise points as they do not belong to any cluster

3\. 	For each core pt p not assigned to a cluster:

	Create a new cluster with this core pt p

	Add all pts that are density connected to p into this new cluster

4\. 	For each border pts: assign it to the nearest cluster

	Important operation: RangeQuery: returns all data points that are in eps radius

**45.6	Hyper Parameters: MinPts and Eps**

	MinPts: \>= d \+ 1, typically 2\*d; for d dimensionality, larger Min pts takes care of noise

		Or ask an expert for Min Pts value

Eps:	For each data point calculate distance from kth nearest point where k \= Min pts Sort these distances and plot to get an elbow point

**45.7	Advantages and Limitations of DBSCAN**

DBSCAN: resistant to noise, can handle different sizes of cluster, does not require n\_clusters to be specified;

DBSCAN has trouble with: Varying densities and high dimensionality data, sensitive to hyper parameters: depends on distance measure which causes curse of dimensionality

**45.8	Time and Space Complexity**

	Time: O(n log n)

	Space: O(n) 

**45.9	Code Samples**

sklearn.clister.DBSCAN()

**45.10	Revision Questions**

1. What is K-means? How can you select K for K-means?  
2.  How is KNN different from k-means clustering?  
3. Explain about Hierarchical clustering?  
4. Limitations of Hierarchical clustering?   
5. Time complexity of Hierarchical clustering?   
6. Explain about DBSCAN?  
7. Advantages and Limitations of DBSCAN? 

**Chapter 46: Recommender Systems and Matrix Factorization**

**46.1	Problem formulation: Movie Reviews**

	Recommend relevant items to a user based on historical data of the item and the user;

	Given dataset A:

		Each row is for user and each column is for item 

		Cell values can be ratings or the usage of the item by the user

		Cell values which do not have values are left as nan rather than replacing with 0

		Matrix A is very sparse as each user can use a small set of items;

		Sparsity \= \# empty cells/ \# total cells; Density \= 1 \- sparsity

Given a user and his history of item usage, recommend a new item that he will most likely use

	Convert the problem into a regression or classification

	Task: Feature engineering for given dataset A (user – item pairs)

	Converting the problem into a matrix completion problem:

		Fill empty cells in the sparse matrix with non empty cell values;

**46.2	Content based vs Collaborative filtering**

	Collaborative Filtering: 

		U1:- M1, M2, M3

		U2:- M1, M3, M4

		U3:- M1

As M1 is common movie watched by user1, user2 and user3, and movie M3 is the common movie watched by user1 and user2, movie M3 can be recommended to U3

Idea: Users who agreed in the past tend to agree in the future (assumption for collaborative filtering)

Content based: uses rating information or the usage matrix values as target variable;

Uses representation of the item and the user (features), such as his preference, gender, location, item type, movie type, movie title, movie cast, etc

**46.3	Similarity based Algorithms**

	Item-item based similarity or user-user similarity;

	User-User similarity:

Given matrix with every row of the matrix is a user vector and every column is an item; 

Build user-user similarity matrix; Say we have U1, U2, and U3, who are most similar to U4, we can recommend items that U4 has not used yet and that U1, U2 and U3 have used already and recommend these items to U4

1. Build user vector based on ratings or item usage  
2. Build user – user similarity matrix using cosine similarity: UiTUj/(||Ui||\*||Uj||)  
3. Find similar users and the items they have in common  
4. Find items that are not watched by the user  
5. Recommend the un common item

   Limitation: User’s preference change over time: thus user-user similarity approach does not work well (when the change is very frequent)

   Item-item similarity: 

   Each item is a vector; and a similarity matrix is build using similarity between items;

   Ratings on a given item do not change significantly after the initial period;

   If we have more users than items and when item ratings do not change much over time after initial period, item – item similarity matrix is preferred

**46.4	Matrix Factorization: PCA, SVD**

	Also called Matrix decomposition:

		Decomposing a matrix into a product of other matrices:

		A \= B\*C\*D \= P\*Q

Principal Component Analysis: Dimensionality Reduction: Can be interpreted as an example of Matrix factorization

X nxd 🡪 Data Matrix; S dxd \= XTX \= co-variance matrix

Eigen values and Eigen vectors of S can be determined and when reducing d dimension to d’ dimension we take top eigen values and corresponding eigen vectors to project data points from d to d’;

![][image201]

- Eigen decomposition of Covariance matrix

  Singular Value Decomposition: Matrix Factorization related to PCA

  PCA:  Eigen value decomposition of covariance matrix;

  SVD: Any rectangular matrix (non square matrix)

  X nxd \= U Σ VT \= (nxn) \* (nxd) \* (dxd)

  	Σ \= diagonal matrix of singular values of X (related to eigen values)

  	Singular value si \= ((n-1) λi)2

		U: contains Left singular vectors

		UT: contains Right singular vectors

		Σ: contains Singular Vectors

**46.5	Matrix Factorization: NMF**

	Non-negative Matrix Factorization:

		Anxm \= Bnxd (CT)dxm

		Such that, Bij \>= 0 and Cij \>= 0

**46.6	Matrix Factorization for Collaborative filtering**

	Aij \= rating on Itemj by Useri

	![][image202]

	Let decomposition be possible:

		![][image203]

	Find B & C:

		argmin 	Σ 	(Aij – BiT Cj)2

		B, C		i, j where Aij is not emty

		Minimizing square loss: Actual Matrix and its Matrix Factorization

1. Solve optimization problem using SGD:

   ![][image204]

   Used non empty A values to get B and C;

2. Compute B and C  
3. Matrix completion: Fill the empty cells with B and C

**46.7	Matrix Factorization for feature engineering**

	Matrix A of user item ratings: Through Matrix factorization we get B and C

		d-dimensional representation of user and item: using A values;

Row vector of B matrix above can be used as useri vectorization and from C we can have itemj vectorization;

The d-dim representation arrived at using Matrox Factorization: if two users are similar then the distance between vectors will be small, similarly for items;

Matrix Factorization can be used for:

	Word vectors

	Eigen faces (face recognition) (through feature engineering)

**46.8	Clustering as Matrix Factorization**

	K-Means optimization: D \= {xi}

Find k- cluster centroids and corresponding sets of data points; Such that every data point belongs to only one set and the distance from the data points to the cluster centroid is minimum;

Define a matrix Z such that Zij \= 1 if xj belongs to Set Si  else 0; The Matrix Z is sparse and can be said to be an assignment problem;

![][image205]

![][image206]

	![][image207]

		If X is decomposed into C and Z through Matrix Factorization: 

such that Zij \= 0 or 1

	and sum of all elements of Zj \= 1

K-Means is Matrix Factorization \+ Col constraints \+ 0,1 values

**46.9	Hyperparameter tuning**

	Anxm \= B \* CT by Matrix Factorization

	d-dimensionality is a Hyperparameter; 

1. Problem specific  
2. Systematic way::  Optimization: min (A – BC)2

   Error plot with d dimensionality is generated and an elbow point is selected

**46.10	Matrix Factorization for recommender systems: Netflix Prize Solution**

	In 2009: Streams Video on demand;

	Given user – ratings and a loss metric (RMSE) 

	![][image208]

	RS 🡪 MF became popular only after Netflix prize;

	![][image209]

![][image210]

bu \= bias due to user, bi \= bias for item

Ratings for users and items are time dependent;

Link: [https://datajobs.com/data-science-repo/Recommender-Systems-\[Netflix\].pdf](https://datajobs.com/data-science-repo/Recommender-Systems-%5bNetflix%5d.pdf)

**46.11	Cold start problem**

If there is a new user joins the system or a new item that is added, then there is no ratings data;

Recommend top items based on meta-data such as geo location, browser, and device;

	We can make content based recommendation;

**46.12	Word Vectors as Matrix Factorization**

Word2Vec: Inspired by Neural Networks, ex: LSA\< PLSA\< LDA\< GLOVE can be interpreted as Matrix Factorizations

	GLOVE: related to W2V

	SVD related to PCA for W2V

1. Co-occurrence matrix:

   D \= {text documents}

   Compute X matrix:

   	Xij \= \# of time wj occurs in the context of Wi;

2. Xnxn \= U Σ VT (nxn each): Matrix Factorization (SVD)

   Using Truncated SVD by discarding (n-k) columns

![][image211]

Truncated SVD: top k singular values, top k left singular vectors and top k right singular vectors: discarding Eigen values with least information; k is the Hyperparameter;

Using co-occurrence matrix and applying truncated SVD over the matrix we will get U matrix. From this U matrix which is of (nxk) shape, we have each row as vector representation of each word of k dimensionality

	Instead of taking all words we can have top words that have good importance;	

**46.13	Eigen-Faces**

	Word vectors 🡪 MF (SVD) 🡪 Co-occurrence matrix)

Image data: Eigen faces for face recognition (PCA) to get feature vectors; (CNN replaced all techniques for image tasks) 

Link: [https://bugra.github.io/work/notes/2014-11-16/an-introduction-to-unsupervised-learning-scikit-learn/](https://bugra.github.io/work/notes/2014-11-16/an-introduction-to-unsupervised-learning-scikit-learn/)

Link: [http://scikit-learn.org/stable/auto\_examples/decomposition/plot\_faces\_decomposition.html\#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py](http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py)

Matrix construction with stacking images row wise where each row in the matrix contains image data which is flattened into a single vector;

From this matrix: Co-variance matrix is computed; dimensionality reduction is applied on this co-variance matrix through Matrix Factorization;

Multiply row wise images stacked matrix with the top k left singular vectors or column wise stacked Eigen vectors of top k eigen values of the co-variance matrix; This multiplication result is the Eigen Faces

Through Eigen values we can compute % of explained variance (ratio of Eigen values) to get good k;	

**46.14	Code example**

	sklearn.decomposition.TruncatedSVD()

	sklearn.decomposition.NMF()

[https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca) 

Elastic Net: min loss \+ alpha\*L1 norm \+ (1 – alpha) L2 norm

L1 norm induces sparsity

**46.15	Revision Questions**

1. Explain about Content based and Collaborative Filtering?  
2. What is PCA, SVD?  
3. What is NMF?  
4. How to do MF for Collaborative filtering?  
5. How to do MF for feature engineering?  
6. Explain relation between Clustering and MF?  
7. What is Hyperparameter tuning?  
8. Explain about Cold Start problem?  
9. How to solve Word Vectors using MF?  
10. Explain about Eigen faces?

**Chapter 47: Interview Questions on Recommender Systems and Matrix Factorization**

**47.1	Question & Answers**

1. How would you implement a recommendation system for our company’s users?(https://www.infoworld.com/article/3241852/machine-learning/how-to-implement-a-recommender-system.html)  
2. How would you approach the “Netflix Prize” competition?(Refer http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/)  
3. ‘People who bought this, also bought…’ recommendations seen on amazon is a result of which algorithm? (Please refer Apparel recommendation system case study, Refer:https://measuringu.com/affinity-analysis/)

**Chapter 48: Case Study 8: Amazon fashion discovery engine (Content Based recommendation)**

\--- Case Studies Notebook

**Chapter 49: Case Study 9: Netflix Movie Recommendation System (Collaborative based recommendation)**

\--- Case Studies Notebook

