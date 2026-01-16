**Module 2: Data Science: Exploratory Data Analysis and Data Visualization**

**Chapter 10: Plotting for exploratory data analysis (EDA)**

**10.1	Introduction to IRIS dataset and 2D scatter plot**

EDA: Simple analysis to understand data; tools include statistics, linear algebra and plotting tools

IRIS dataset: Hello World of Data Science; Collected in 1936; Classify a flower into 3 classes; 

Features: Sepal Length, Sepal Width, Petal length, Petal Width

		![][image7]

	Columns: Variables or features

	Rows: Data points or vector: n-dimensional numerical variable

	![][image8]

	Species: Target/ Label; A 1D vector: Scalar;

	These features are determined through domain knowledge

	Q: Number of data points in the data set? .shape

	Analysis is more important than code;

	Q: What are the features? .columns

Q: Number of points in each class? .value\_counts() (Also gives idea of the balance in class labels across number of data points)

	Q: 2D scatter plot? (iris.plot(kind \= ‘scatter’, x \= ‘sepal\_length’, y=’sepal\_width’)

		Use seaborn to separate different classes on above plots

![][image9]![][image10]

One of the labels can be separated easily by drawing a line;

	EDA gives us these neat characteristics of the data set;

**10.2	3D scatter plots**

	Use plotly;

	![][image11]

We cannot visualize 4D, 5D or nD (requires mathematical tools to reduce complexity)

**10.3	Pair plots**

	To allow visualization upto 6D sometimes upto 10D, Larger nD cannot be visualized;

	Seaborn.paiplot(iris, hue \= “species”)

	We can write straight forward rules (if else) to separate one of the class labels;

**10.4	Limitations of Pair plots**

	Pair plots are useful when the dimensionality is small

**10.5	Histogram and Introduction to PDF (Probability Density Function)**

	Uni-Variate analysis:

![][image12]

Histograms: 

![][image13]

PDF: smoothed histogram (Kernel Density Estimation)

**10.6	Univariate Analysis using PDF**

	Histograms plot on each variable (as the number of features is less, \= 4\) 

**10.7	CDF (Cumulative Distribution Function)**

	**![][image14]**

	CDF: Percentage of data points that are below an x value;

	PDF: Percentage of data points that lie between two x values;

Differentiate cdf to get pdf and integrate pdf to get cdf; Convert data into bins to get pdf, and use npp.cumsum to get cdf;

![][image15]

We can get intersections of CDF levels on x axis above to know how many points of each class intersect

**10.8	Mean, Variance and Standard deviation**

	Mean: 		Sum(data values)/number of values

			central value of the data, are majorly impacted by outliers

	Standard deviation:	The width of the spread of the values on an axis

	Variance: Summation squares of distance of values from mean whole divided by N

	Standard deviation \= sqrt(variance)

	These are corrupted if an outlier exists; use median

**10.9	Median**

Median does not get corrupted due to presence of outliers; better statistic for central tendency;

Sort the list in an increasing order and pick the middle value; 

**10.10	Percentile and Quantiles**

	Percentile: Number of values lesser than the percentile percentage of values;

		50th percentile: the value at which 50% of values are less than this value

	Quantiles: 25th percentile, 50th percentile, 75th percentile

**10.11	IQR (Inter Quartile Range) and MAD (Median Absolute Deviation)**

	Median Absolute Deviation: (1/n) \* sum(abs( valuei – median))

	IQR: 75th percentile value – 25th percentile value

**10.12	Box – plot with Whiskers**

The **box plot** (a.k.a. **box and whisker diagram**) is a standardized way of displaying the distribution of data based on the five number summaries: minimum, first quartile, median, third quartile, and maximum. (Google Search)

![What does a box plot tell you? | Simply Psychology][image16]

**10.13	Violin Plots**

A **violin plot** is a method of **plotting** numeric data. It is similar to a **box plot**, with the addition of a rotated kernel density **plot** on each side. **Violin plots** are similar to **box plots**, except that they also show the probability density of the data at different values, usually smoothed by a kernel density estimator.

![Violin Plots 101: Visualizing Distribution and Probability Density][image17]\[Google Search\]

**10.14	Summarizing Plots, Univariate, Bivariate and Multivariate analysis**

Write conclusion at end of each step or plots during data analysis, data analysis should align with project objective;

Univariate: Analysis considering only one variable (PDF, CDF, Box-plot, Violin plots)

Bivariate: two variables (Pair plots, scatter plots)

Multivariate: more than two variables (3D plots)

**10.15	Multivariate Probability Density, Contour Plot**

	**![][image18] ![][image19]**

Combining probability densities of two variables; dense regions are darker as if a hill coming out

**Chapter 11: Linear Algebra**

**11.1 	Why Learn it?**

We will apply it to solve specific type of problems in ML; we will learn things in 2d and 3d and extend it to nd

**11.2	Introduction to Vectors (2-D, 3-D, n-D), Row Vector and Column Vector**

	Point: 

		Let us have a Coordinate system with x1 and x2 axes;

		Point can be denoted by n-dimensional vector;

	Distance of a point from origin: 

		In 2D: sqrt(x12 \+ x22)

		In 3D: sqrt(x12 \+ x22 \+x32)

		In nD: sqrt(x12 \+ x22 \+x32\+….+xn2)

	Distance between two points: 

		In 2D: sqrt((x1-y1)2 \+ (x2-y2)2)

		In 2D: sqrt((x1-y1)2 \+ (x2-y2)2\+ (x3-y3)2)

		In 2D: sqrt((x1-y1)2 \+ (x2-y2)2\+ …+ (xn-yn)2)

	Row vector: A matrix with n columns and 1 row

	Column Vector: A matrix with 1 column and n rows

**11.3	Dot Product and Angle between 2 Vectors**

	a \= \[a1, a2, ……., an\]

	b \= \[b1, b2, ..…., bn\]

	c \= a \+ b \= \[a1 \+ b1, a2 \+ b2, ….., an \+ bn\]

Dot product: a.b \= a1b1 \+ a2b2 \+ ….. anbn \= aT \* b (by default every vector is a column vector)

Geometrically represents the angle between the two vectors;

![][image20]

	a.b \= ||a|| ||b|| cos(θab)  \= a1b1 \+ a2b2 \+…..

	||a|| \= sqrt ( a12 \+ a22 \+ …..)

If the dot product of two vectors is zero then the two vectors are orthogonal.	

**11.4	Projection and Unit Vector**

	Projection of a on b \= d \= a cos(θab) \= a.b/||b||

**![][image21]**				![][image22]

								    Unit Vector: â  \= a/||a||

	Unit vector â is in same direction as a; || â|| \= 1

**11.5	Equation of a line (2-D), Plane (3-D) and Hyperplane (n-D), Plane passing through origin, Normal to a Plane**

	Line 2D: y \= mx \+ c, ax \+ by \+ c \= 0, y \= – c/b – ax/b

	Plane 3D: ax \+ by \+ cz \+ d \= 0

	Hyper Plane nD: w0 \+ w1x1 \+ w2x2 \+ w3x3 \+ …..  \= 0

				Hyper plane π: WTx \+ W0= 0

Wn+1x1 \= \[W1, W2, …. \]; X n+1 x 1 \= \[x1, x2, … \] (W0 determines intercept of the hyper plane on y – axis\] (To pass through origin this intercept must be 0\)

	π: WTx \= 0: Hyper plane passing through origin

	W.x \= WTx \= ||w|| ||x|| cos(θxy) 

θxy\= 900; W is vector perpendicular to the plane π

**11.6	Distance of a point from a plane/ Hyperplane, half-spaces**

	π: WTx \= 0: Hyper plane passing through origin

θxy\= 900; W is vector perpendicular to the plane π

	Distance of a point P (p1, p2, …):	

d \= abs(WTP / ||W||)  

	A hyper plane divides the whole space into two half spaces;

	If W.P \> 0 then point P lies in the direction of W;

**11.7	Equation of a Circle (2-D), Sphere (3-D) and Hypersphere (n-D)**

	Circle: (x-a)2 \+ (y-b)2 \= c2

Given a point: if its distance from the center is less than the radius of a circle then the point lies inside the circle;

x12 \+ x22 \< r2 then the point p(x1,x2) lies inside the circle that is centered at origin;

x12 \+ x22 \> r2 then the point p(x1,x2) lies outside the circle;

x12 \+ x2 2 \= r2 then the point p(x1,x2) lies on  the circle;

Sphere: x12 \+ x22 \+ x32 \= r2 

Hyper sphere: x12 \+ x22 \+ x32 \+ .….. \+ xn2 \= r2 

**11.8	Equation of an Ellipse, Ellipsoid and Hyper ellipsoid**

	Ellipse: (x/a)2 \+ (y/b)2 \= 1

Ellipsoid: (x1/a)2 \+ (x2/b)2 \+ (x3/c)2 \= 1 

Hyper sphere: (x1/a1)2 \+ (x2/a2)2 \+ (x3/a3)2  \+ …… \+ (xn/an)2 \= 1

**11.9	Square, Rectangle**

	**![][image23]**

if x1 \< a2 and x1\>a1: if y1\>b1 and y1\<b2 then point P (x1, y1) lies inside the axis parallel rectangle, if a2 – a1 \= b2 – b1 then we get a square

Can be extended to 3D

**11.10	Hyper Cube, Hyper Cuboid**

if x \< a2 and x\>a1: if y\>b1 and y\<b2: if z\>c1 and z\<c2 then point P (x, y,z) lies inside the axis parallel cuboid, if a2 – a1= b2 – b1 \= c2 – c1 then we get a cube;

**11.11	Revision Questions**

	**Q1**. Define Point/Vector (2-D, 3-D, n-D)?

**Q2**. How to calculate Dot product and angle between 2 vectors?

**Q3**. Define Projection, unit vector?

**Q4**. Equation of a line (2-D), plane(3-D) and hyperplane (n-D)?

**Q5**. Distance of a point from a plane/hyperplane, half-spaces?

**Q6**. Equation of a circle (2-D), sphere (3-D) and hypersphere (n-D)?

**Q7**. Equation of an ellipse (2-D), ellipsoid (3-D) and hyperellipsoid (n-D)?

**Q8**. Square, Rectangle, Hyper-cube and Hyper-cuboid?

**Chapter 12: Probability and Statistics**

**12.1	Introduction to Probability and Statistics**

Datasets with well separable classes do not require ML methods to be applied, but classes that are well mixed cannot be separated easily;

The data points that lie in the mixed region cannot be clearly stated to belong to a certain class; instead we can have probability values of the data point belonging to each class;

Histograms, PDF, CDF, mean, variance, etc come under probability and statistics

Random Variable: can take multiple values in a random manner

	Ex: Dice with 6 sides: Roll a fair dice: the 6 outcomes of the dice are equally likely

	r.v. X \= {1, 2, 3, 4, 5, 6}

	Coin toss: r.v. X \= {H, T}

	Dice roll: Probability of X \= 1 is 1/6

	Probability of X \= even is {2,3,4}/{1,2,3,4,5,6} \= 3/6 \= ½

			\= prob(x=2) \+ prob(x=4) \+ prob(x=6)

	Height of a randomly picked student: Y \= \[120 to 190 cm\]

	Discrete random variable: a value from a set of values

	Continuous random variable: a value from a range of values

Outlier: Y: Height of students: 

	Let: Y \= {122.2, 146.4, 132.5, …, 12.26, 156.23, 92.6}

Outliers: 92.6 and 12.26: may have occurred due to input error or data collection error or may be a genuine value but does not indicate general trend of the population

**12.2	Population and Sample**

Population example: Set of heights of all people in the world;

Task: estimate the average height of a human;

	Mean \= (1/n) sum(h)

Collecting all the population height data is not possible; we will take a random sample (subset of the population) 

**12.3	Gaussian/Normal Distribution and its PDF (Probability Density Function)**

	Bell shaped curve: Gaussian distribution Probability density function curve;

	X: continuous random variable;

	Real world variables mostly follow Gaussian distribution; Height, Weight, Length 

We learn distributions to understand the underlying statistics or trends of the population

If X behaves as a Gaussian distribution: Given mean and standard variance:

	We can estimate the values of the population; by plotting the PDF

![Probability density function for the normal distribution][image24]

The red curve is standard normal distribution

Parameters of Gaussian distribution: Mean and variance:

X \~ N (0,1) (mean 0 and variance 1\)

	PDF(X \= x) \= (1/sqrt(2π) σ) exp(-(x-µ)2/2σ2)

	As x moves from mean, the pdf reduces;

	The PDF of a Normal distribution is symmetric;

	

**12.4	CDF (Cumulative Distribution Function) of Gaussian/Normal distribution**

	![Cumulative distribution function for the normal distribution][image25]

	N(µ,2σ2);

	As variance decreases the plot gets converted to a step function;

![https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Empirical\_Rule.PNG/350px-Empirical\_Rule.PNG][image26]

Standard deviation plot: 68 – 95 – 99.7 rule

**12.5	Symmetric distribution, Skewness and Kurtosis**

	One of the applications of these stats is to understand the distribution;

If the PDF has a mirror image of the curve of the either sides of the mean, then the distribution is symmetric over mean; as in above std dev plot;

F(x0 – h) \= F(x0 \+ h): function symmetric over x0 

Skewness: around the mean the distribution is not symmetric; one of the sides has a longer tail; can be said to be a measure of asymmetry;

![][image27]

Sample Skewness:

![][image28]

![][image29]

Kurtosis:

Excess kurtosis: kurtosis – 3

![][image30]

![][image31]

Kurtosis of Gaussian random variable \=3;

Through excess kurtosis we compare distributions with Gaussian random distribution

	Kurtosis is a measure of tailedness; it is not a measure of peakedness;

	This will give us an idea of outliers in the distribution; Smaller the better

**12.6	Standard normal variate (Z) and standardization**

	Z \~ N(0,1), mean \= 0 and variance \= 1

	Let X \~ N(µ,σ2); this can be converted to Z by:

Mean centering and scaling (X – mean) / std dev; To help understand the disb.

**12.7	Kernel density estimation**

	**![][image32]**

	Computing PDF from Histograms: using KDE;

	A Gaussian Kernel is plotted centered at each observation (of the histogram);

	Variance of this Kernel (bell curve) is called band width;

At every point in the range of the Gaussian Kernels we will add all PDF values to get a combined PDF; bandwidth selection is done by experts;

**12.8	Sampling distribution & Central Limit theorem**

	Let X be a not necessarily Gaussian distribution of a population (say incomes)

	Pick m random Samples independently of size **n** each 🡪 S1, S2, S3, …., Sm

	 For all Samples: compute means, x1m, x2m, x3m, ……

	These sample means will have a distribution: Sampling distribution of sample-mean

	Central Limit Theorem: If X (original population), X: finite mean and variance

Sampling distribution of sample means will be a Gaussian distribution with 		mean \= population mean and variance \= population variance/**n** as n increases

It is generally observed that if n is around 30 and m \= 1000, we can observe a Gaussian distribution;

**12.9	Q-Q plot: How to test if a random variable is normally distributed or not?**

Quantile – Quantile plot: Graphical way of determining similarity between two distributions

	X: a new distribution: Task: to understand whether X is Gaussian

1. Sort X values and compute percentiles;  
2. Y \~ N(0,1): take some k observations and sort \+ compute percentiles  
3. Plot: X percentile values on Y axis and Y percentile values on x axis

   If the plot is approximately a straight line then the distributions are similar;

   ![][image33]

   Code: stats.probplot(X, dist \= ‘norm’, plot \= pylab)

   If number of observations is small it is hard to interpret QQ plot;

   ![][image34]

   Plot where distributions are different;

	

**12.10	How distributions are used?**

	Probability is useful for data analysis;

Ex: Imagine your task is to order T shirts for all employees at your company. Let sizes be S, M, L and XL. Say we have 100k employees who have different size requirements; 

Q) How many XL T shirts should be ordered?

	Collect data for all 100k employees

	Let us have a relationship between heights and T shirt size; Domain knowledge;

	Collect heights from 500 random employees; Compute mean and std dev

	At gate of entry we can do this;

	From domain knowledge, let heights \~N(mean, variance)

We can extend the distribution of heights from 500 employees to 100k employees; We made many assumptions here; these may work in natural features

Q) Salaries: If salaries are Gaussian distributed, we can estimate how many employees make a salary \> 100k $;	

So distributions give us a theoretical model of a random variable; This will help us understand the properties of the random variable;

**12.11	Chebyshev’s inequality**

If we don’t know the distribution and mean is finite and standard deviation is non-zero and finite;

Task: to find the percentage of values of lie within a range;

Salaries of individuals (millions of values); distribution is unknown, but mean and std dev are known;

P(|x \- µ| \>= kσ) \<= 1/k2

**12.12	Discrete and Continuous Uniform distributions**

	PDF for continuous and PMF for discrete random variables;

	Roll a dice: uniform distribution; each observation is equally probable;

	Discrete Uniform: 

Parameters (a,b,n=b-a+1)

		Pmf \= 1/n

		Mean \= (a+b) /2 \= Median

		Variance \= ((b-a+1)2\-1)/12

		Skewness \= 0

	Continuous Uniform:

		 Parameter: a,b

		PDF: 1/(b-a)

		Mean: (a+b)/2 \= Median

Variance \= (b-a)2/12

**12.13	How to randomly sample data points (Uniform Distribution)**

	Using random functions in python;

	We can use a threshold value to generate a random number of random numbers;

	Probability of picking any value is equally probable;

**12.14	Bernoulli and Binomial Distribution**

	Bernoulli: discrete: Coin toss:{H, T}, P \= {0.5, 0.5}; parameter p

	Binomial: X \~ Bernoulli and Y \= n times X; parameters: n, p

	An event with n trials with success probability p in each trial and p \= 0 or 1

**12.15	Log Normal Distribution**

	X is log normal if natural logarithm of x is normally distributed;

	![][image35]![][image36]

In user reviews: most of comments are of small length and some of comments have large length of words;

We can employ QQ plot to check similarity of log of distribution to normal distribution;

**12.16	Power law distribution**

	![][image37]

	Green area is 80% values;

	In bottom 20% of values you can find 80% of mass;

When a distribution follows a power law then the distribution is called Pareto distribution; 

Parameters: scale and shape

![][image38]![][image39]

Applications: allocation of wealth in population, sizes of human settlements; 

 ![][image40]

We can also use a QQ plot against Pareto plot;

**12.17	Box cox transform**

	Power transform:

	In Machine Learning we assume generally that a feature follows a Gaussian distribution;

	Can we convert Pareto into Gaussian?

	Pareto: X: x1, x2, …., xn	&	Gaussian Y: y1, y2, ….., yn

	Box-cox(x) \= λ

	yi \= {(xiλ – 1\) / λ if λ\!=0 and lg(xi) if λ \= 0}

	If λ \= 0, then the power law distribution is actually a log normal;

	scipy.stats.boxcox()

	Box-cox does not work always. Use QQ-plot to check its results.

**12.18	Applications of non-Gaussian distributions?**

	Uniform for generating random numbers;

A well studied distribution gives a theoretical model for the behavior of a random variable

Weibull distributions: 

The upstream rainfall determines the height of a dam which stands for 100s of years without repairs; Probability of rainfall \> a value is required; 

This distribution is applied to extreme events such as annual maximum one day rainfalls and river discharges.

![][image41]

**12.19	Co-variance**

	Task: To determine relationships between different random variables; 

	Cov(x, y) \= (1/n) \* sum((xi \- µx) (yi \- µy))

	Cov(x, x) \= var(x) 

	Cov(x, y) will be positive if as x increases y increases

	Cov(x, y) will be negative if as x increases y decreases

Sign of co-variance is indicative of relationship, the value of co-variance does not indicate strength of relationship;

**12.20	Pearson Correlation Coefficient**

	To measure relationship between two random variables: ρ

	Ρx,y \= Cov(x, y) /(σx, σy)

	![][image42]

	ρ \= 0 🡪 no relation between the two random variables;

	Pearson relation coefficient assumes that the relationship is linear:

	![][image43]

	Does not perform well with non-linear relationships;

	![][image44]

	The above plot has a monotonically non decreasing curve

**12.21	Spearman Rank Correlation Coefficient**

	Pearson corr coeff: works for linear relationships

Spearman rank corr coeff:  	this is Pearson correlation coefficient of ranks of the random variables; Spearman correlation coefficient is much more robust to outliers than Pearson Corr coef

		![][image45] ![][image46]

	This allows us to understand whether two random variables increase at the same time;

	![][image47]

		Pearson				Spearman

**12.22	Correlation vs Causation**

	Correlation does not imply causation;

	Example: Number of Nobel Laureates vs Chocolate consumption per capita;

		The rank correlation coefficient \~ 0.8

		As x increased y increased;

But we know that as chocolate consumption and Number of Nobel Laureates do not relate;

Causations are studied with causal models which is a separate field of study;

**12.23	How to use correlations?**

	Q. Is salary correlated with sq footage of home?

	This data will be useful for real estate agents;

	Q. Is \# of years of education correlated with income?

	Useful for education ministry to encourage people to get more years of study;

Q. Is time spent on web page in last 24 hours correlated with money spent in the next 24 hours?

Useful for ecommerce to encourage people to spend more time on the website;

Q. Is number of unique visitors to the website correlated with the $ sales in a day?

The company will then take measures to increase number of unique user in a day;

Q. Dosage correlation with reduction in blood sugar;

Correlations can help us answer these questions;

**12.24	Confidence interval (C.I) Introduction**

	Let X be a random variable of height values;

	Distribution of X is unknown;

	Task: Estimate the population mean of X \= μ

	μ \= (1/n) \* summ(i \= 1 to n) (xi)

	As n increases sample mean reaches population mean;

With μ we have a point estimate and μ is sample mean that is calculated over a randomly selected sample of the population;

Instead we can also express mean in terms of confidence interval;

Say we have 170 cm as sample mean; we can say that sampling mean lies between 165 and 175 cm for 95% of the sampling experiments when the repeat the sampling multiple times; for 5% of the experiments the mean will not fall in this interval; This does not mean that population mean will lie in the interval with 95% probability;

**Confidence Interval: An N% confidence interval for parameter p is an interval that includes p with probability N%; \[Source: Tom Mitchell book\]**

**12.25	Computing confidence interval given the underlying distribution**

	Imagine we have a random variable of heights of population;

	Let this random variable follow a Gaussian distribution; 

	Let mean \= 168 cm and standard deviation \= 5

	![][image48]

We can get 95% CI from Gaussian distribution characteristics which range from μ \- 2σ to μ \+ 2σ, which is 158 to 178 cm;

**12.26	C.I for mean of a random variable**

	Random variable X \~ F(μ, σ)

	Let sample size be 10;

	Let samples be {180, 162, 158, 172, 168, 150, 171, 183, 165, 176}

	What is the 95% CI for population mean; the distribution can be anything; 

	Case 1: Let σ \= 5cm; 

CLT: sample mean x\_bar follows Gaussian distribution with mean \= population mean and standard deviation \= population standard deviation/sqrt(sample size);

We can say that μ ε \[x\_bar \- 2σ/sqrt(n), x\_bar \+ 2σ/sqrt(n)\] with 95% confidence;

Thus 95% CI for the samples:

	x\_bar \= 168.5

	n \= 10

Population Mean lies between 165.34 and 171.66 with 95% confidence;

	Case 2: if σ is unavailable: CLT cannot be applied;

		We can assume Student’s t distribution; 

![][image49]

ν \= sample size – 1

We can apply 95% characteristics of t distribution;

We have CI for mean; what if we like to have CI for standard deviation, median, 90th percentile and other statistics;

**12.27	Confidence interval using bootstrapping**

	Let X be a random variable with unknown distribution F;

	Let us compute confidence interval for median of X;

	S: sample size n {x1, x2, …, xn} n \= 10, 20, 40 or 50

	From sample we generate new samples: 

		S1: x1(1), x2(1), …., xm(1) and m\<=n;

			Random sample with replacement of size m generated from sample S;

			Using S as a uniform random variable;

		Generate k samples;

		K samples generated from Sample S with replacement;

		For each sample we can generate the statistic desired;

		Considering median: we can have k medians from k samples; Let k be 1000;

		Sort these k medians and consider central 95% of the medians;

		95% CI for population median is (median25, median975)

		We can use the same process for computing other statistics;

This is a non parametric technique to estimate CI for a statistic; and this process is called as bootstrap;

![][image50]

![][image51]

![][image52]

**12.28	Hypothesis testing methodology, Null-hypothesis, p-value**

	Let us have 2 classes; and let the number of students be 50; let us have height values;

	Q. Is there a difference in heights of students in cl1 and cl2?

	Say we plot histograms and saw that mean of class 2 is greater than mean of class 1;

	

	Hypothesis testing:

1. Choosing a test statistic: (μ2 – μ1)

   If μ2 – μ1 \= 0 then the classes have same means;

2. Define a Null hypothesis (H0): there is no difference in μ2 and μ1

   Alternative hypothesis (H1): there exists a difference in μ2 and μ1

3. Compute p-value: probability of observing μ2 – μ1 (that is evident from the observations) if null hypothesis is true;

   Say we have p value \= 0.9 for probability of observing a mean difference of 10cm when H0 is true;

   As p value is high than a threshold level (generally 5%) then we accept the null hypothesis else we reject null hypothesis; Also **this p value *does not* tell about the acceptance of alternate hypothesis or the probability of acceptance of null hypothesis;**

   Taken an observation of μ2 – μ1 from the classes which is equal to 10 that is μ2 – μ1 \= 10 the ground truth statistic that has already occurred; and computing the p value as the conditional probability of finding μ2 – μ1 \= 10 given H0 (which says that there is no difference in means); if the p value is less than threshold that is 5% we reject null hypothesis saying that we cannot accept that there is no difference in means and we reject null hypothesis in favour of the alternate hypothesis; if the p value is \> 5% we say that we don’t have sufficient  the null hypothesis saying that there is sufficient evidence that the observed value occurs under the null hypothesis with satisfactory probability;

   P value says that what is the probability of an observation statistic occurs under null hypothesis;

**12.29	Hypothesis testing intuition with coin toss example**

	Example: Given a coin determine if the coin is biased towards heads or not;

		Biased towards head: P(H )\> 0.5

		Not biased: P(H) \= 5;

	Design an experiment: flip a coin 5 times and count \# of heads \= X random variable;

		This X is the test statistic; (the number of heads)

	Perform experiment: Let that we got 5 heads; test statistic X \= 5 the observed value;

		Let the null hypothesis be “coin is not biased towards head”

		 P(X=5|coin is not biased towards head) \= ?

		Probability of observed value given null hypothesis;

		P(X=5|H0) \= (½)5 \~ 0.03

		P(X=5|H0) \= 0.03 \< 0.05

	Recap: p –value \= Probability of observing an observed value given a null hypothesis

If p-value \< 0.05 then null hypothesis is incorrect as observation is ground truth it cannot be incorrect; we then reject null hypothesis in favor of alternative hypothesis;

We reject that the coin is unbiased and accept that the coin is biased;

We have sample size as a choice; we have 5 coin flips, we can have 3 flips, 10 flips or 100 flips; and can perform the experiment;

If 3 flips are used and let us observe 3 Heads; 

	p-value \= ½\*½\*½ \= 1/8 \= 12.5% \> 5%

	Thus we fail to reject the null hypothesis; we accept null hypothesis;

Important criterion for hypothesis testing:

1. Design of the experiment  
2. Defining Null Hypothesis  
3. Design Test statistic

   Error: 	p-value \<5% 🡪 reject H0

   P-value is the probability of observation given H0; we cannot say anything about probability of H0 being true;

**12.30	Re-sampling and permutation test**

	We have the same student heights;

We have 2 classes with 50 height observations each; Take means and compute the difference between the means, let this difference be D;

We then mix all height values into a 100 value vector and randomly sample 50 points from the 100 points to form X vector and rest 50 into Y vector;

We have mean of X and mean of Y: the difference of these means is the mean difference of Sample 1;

Similarly we will generate n samples and compute mean difference; we will have n mean differences; (Note to make random sampling for picking 50 points)

Let n \= 10 000; 

The mean differences are sorted; We will now have D1, D2, …., D10000 in sorted order;

Now place the original mean difference before sampling in the sorted means list;

Computing the percentage of Di values that are greater D will generate a p-value;

Say D \~ D9500; then we have 500 D values (from D9501 to D10000) that are greater than D; thus we have 5% of values; hence p-value \= 5%; we can compute p-value with this process;

Why is this percentage considered as p-value; Initially while jumbling we assumed that there is no difference in the means of the 2 classes or 2 height lists; this is the assumption of null hypothesis; then we have experimented with sampling for 10000 iterations;

If the percentage of the sorted sample differences that are greater than original mean difference D is less than threshold value 5% then the sampling is not random; this implies that null hypothesis should be rejected;

p-value thresholds depends on problem; For medical purposes p-value of 1% is reasonable; generally p-value threshold is kept at 5%;

**12.31	K-S Test for similarity of two distributions**

	Kolmogorov-Smirnov test:

		Do two random variables X1 and X2 follow same distribution?

		We plot CDF for both random variables;

		We will be using hypothesis testing;

		Null hypothesis: the two random variables come from same distribution;

		Alternative hypothesis: both don’t come from same distribution;

		Test statistic; D \= CDF(X1) – CDF(X2) throughout the CDF range;

\= supremum (CDF(X1) – CDF(X2)) 

\= max of ( CDF(X1 distribution) – CDF(X2 distribution) ) 

(at same value on x axis)

		**Null hypothesis is rejected at level α, when D \> c(α) \* sqrt( (n+m)/nm )**

	c(α) \= sqrt(-0.5\*ln(α/2))

	D \> (1/n0.5)\* ( sqrt( \-0.5 \* ln(α) \* (1+ (n/m)) )

**12.32	Code Snippet K-S Test**

	Normal distribution:

	![][image53] ![][image54]

	

	![][image55]

	P- value is high thus X comes from Normal distribution;

	Uniform distribution:

	![][image56]

![][image57]

P-value is low thus X does not follow Normal distribution (we already know that X follows Uniform distribution)

**12.33	Hypothesis testing: another example**

	Difference of means:

Given heights list of two cities, determine if the population means of heights in these two cities is same or not;

Experiment: We cannot collect the whole population data; we will take a sample; say we collect height of 50 random people from both the cities;

Compute sample means from both cities μA (let \= 162 cm) and μB (let \= 167 cm); these are sample means as population data collection is infeasible;

Test statistic: Mean of city A – Mean of city B \= 167 – 62 \= 5 cm

Null hypothesis: μB – μA \= 0; there is no difference in means of height values of two cities;

Alternative hypothesis: μB – μA \!= 0

Compute: P(μB – μA \= 5 cm | H0) \= probability of observing a difference of 5 cm in sample mean heights of sample size 50 between two cities if there is no difference in mean heights;

Case 1: P(μB – μA \= 5 cm | H0) \= 0.2 \= 20%

There is a 20% chance of observing 5% difference in sample mean heights of C1 and C2 with 50 sample size if there is no population mean difference in heights;

As 20% is significant then null hypothesis must be true; We accept the null hypothesis;

Case 2: P(μB – μA \= 5 cm | H0) \= 0.03 \= 3%

There is a 3% chance of observing 5% difference in sample mean heights of C1 and C2 with 50 sample size if there is no population mean difference in heights;

As 3% is insignificant then null hypothesis must be False; we reject null hypothesis and accept the alternative hypothesis

**12.34	Re-sampling and Permutation test: another example**

	We have:

Test statistic: μB – μA \= 5 cm with sample size of 50; 

Null hypothesis: no difference in population means

List of C1 and C2 heights;

Step 1: S \= {C1 U C2} \= set of 100 heights

Step 2: From set S, randomly select 50 heights for S1 and 50 heights for S2

Compute means of these S1 and S2 heights: μ1 and μ2; we have re-sampled the data set; with this we are making an assumption that there is no difference in μB and μA, which is the null hypothesis;

Compute: μ2 \- μ1 

	Step 3: Repeat Step 2 for k times, let k \= 1000

Result: we will have 1000 (μ2 \- μ1); sort these; these are simulated differences 	under null hypothesis

		Result: we will have 1000 (μ2 \- μ1) (sorted) and we have μB – μA \= 5 cm

Compute the percentage of simulated means that are greater than test statistic; this is the p-value which can be checked for significance at α level;

	Note: simulated differences are computed using Null Hypothesis;

P(observed difference|H0) \= percentage of simulated means that are greater than test statistic \= p-value; if p-value \> 5% accept null hypothesis else reject null hypothesis;

Observed difference can never be incorrect as it is ground truth generated from the data; Acceptance or rejection happens with null hypothesis;

**12.35	How to use hypothesis testing?**

	Determining two random variables follow same distribution;

Drug testing: Effectiveness of a new drug over an old drug; claim new drug makes recovery fever from fever in comparison to old drug;

		Collect 100 patients: randomly split into two groups of 50 people each;

Administer old drug to group A and new drug to group B; record time taken for all the 100 people to recover;

		Let mean time for old drug people be 4 hours and for new drug people be 2 hrs

Mean tells that the new drug is performing well; note that the sample size is 50; thus hypothesis is applied;

H0: Old drug and new drug take same time for recovery;

Test statistic: μold – μnew \= 2;

P(μold – μnew \= 2 | H0) \= ? (let \= 1%)

If there is no difference in old drug and new drug then the probability of observing μold – μnew \= 2is 1%; this implies that the null hypothesis and observation do not agree with each other, thus null hypothesis is incorrect;

**P-value: agreement between null hypothesis and the test statistic;**

Probability value is computed using re-sampling and permutation test; combine the datasets into 1 large set and sample randomly and compute test statistics for samples and sort the results; find the percentage of values that are greater than the original test statistic value;

Significance level is problem specific; for problems such as medical tests significance level is 0.1% or1%; for ecommerce 3% or 5% are generally taken;

	Can be used for computing average spending of a set of people;

**12.36	Proportional Sampling**

	Let d: \[d1, d2, d3, d4, d5\] \= \[2.0, 6.0, 1.2, 5.8, 20.0\]

Task: pick an element from the list such that the probability of picking the element is proportional to the value; 

For random selection the probability of picking any value is equal to all other values;

But the task requires proportional sampling:

1. a. Compute sum of all the elements in the list: \= 35.0

   b. Divide list with the sum, list \= \[0.0571, 0.171428, 0.0343, 0.1657, 0.5714\]

   	Sum of all these values \= 1 and all are between 0 and 1

   c. Cumulate the list: 

   	C\_list \= \[0.0571, 0.228528, 0.262828, 0.428528, 1.00\]

2. Sample one value from Uniform distribution between 0 and 1; let r \= 0.6;  
3. Use C\_list; place r in the list and return the original value at the index at which r is placed;

   In C\_list the gap between values is proportional to the original values; as random number r is generated from a uniform distribution; the probability of picking any value is equal to the gap between the elements in C\_list in turn proportional to the original list values;

**12.37	Revision Questions**

1. What is PDF?  
2. What is CDF?  
3. Explain about 1-std-dev, 2-std-dev, 3-std-dev range?  
4. What is Symmetric distribution, Skewness and Kurtosis?  
5. How to do Standard normal variate (z) and standardization?  
6. What is Kernel density estimation?  
7. Importance of Sampling distribution & Central Limit theorem  
8. Importance of Q-Q Plot: Is a given random variable Gaussian distributed?  
9. What is Uniform Distribution and random number generators  
10. What Discrete and Continuous Uniform distributions?  
11. How to randomly sample data points?  
12. Explain about Bernoulli and Binomial distribution?  
13. What is Log-normal and power law distribution?  
14. What is Power-law & Pareto distributions: PDF, examples  
15. Explain about Box-Cox/Power transform?  
16. What is Co-variance?  
17. Importance of Pearson Correlation Coefficient?  
18. Importance Spearman Rank Correlation Coefficient?  
19. Correlation vs Causation?  
20. What is Confidence Intervals?  
21. Confidence Interval vs Point estimate?  
22. Explain about Hypothesis testing?  
23. Define Hypothesis Testing methodology, Null-hypothesis, test-statistic, p-value?  
24. How to do K-S Test for similarity of two distributions?

**Chapter 13: Interview Questions on Probability and statistics**

**13.1	Questions and Answers**

1. What is a random variable?  
2. What are the conditions for a function to be a probability mass function?(http://www.statisticshowto.com/probability-mass-function-pmf/)  
3. What are the conditions for a function to be a probability density function ?(Covered in our videos)  
4. What is conditional probability?   
5. State the Chain rule of conditional probabilities?(https://en.wikipedia.org/wiki/Chain\_rule\_(probability))  
6. What are the conditions for independence and conditional independence of two random variables?(https://math.stackexchange.com/questions/22407/independence-and-conditional-independence-between-random-variables)  
7. What are expectation, variance and covariance?(Covered in our videos)  
8. Compare covariance and independence?(https://stats.stackexchange.com/questions/12842/covariance-and-independence)  
9. What is the covariance for a vector of random variables?(https://math.stackexchange.com/questions/2697376/find-the-covariance-matrix-of-a-vector-of-random-variables)  
10. What is a Bernoulli distribution?   
11. What is a normal distribution?  
12. What is the central limit theorem?  
13. Write the formula for Bayes rule?  
14. If two random variables are related in a deterministic way, how are the PDFs related?  
15. What is Kullback-Leibler (KL) divergence?  
16. Can KL divergence be used as a distance measure?  
17. What is Bayes’ Theorem? How is it useful in a machine learning context?  
18. Why is “Naive” Bayes naive?  
19. What’s a Fourier transform?  
20. What is the difference between covariance and correlation?  
21. Is it possible capture the correlation between continuous and categorical variable? If yes, how?  
22. What is the Box-Cox transformation used for?  
23. What does P-value signify about the statistical data?  
24. A test has a true positive rate of 100% and false positive rate of 5%. There is a population with a 1/1000 rate of having the condition the test identifies. Considering a positive test, what is the probability of having that condition?  
25. How you can make data normal using Box-Cox transformation?  
26. Explain about the box cox transformation in regression models.  
27. What is the difference between skewed and uniform distribution?  
28. What do you understand by Hypothesis in the content of Machine Learning?  
29. How will you find the correlation between a categorical variable and a continuous variable?  
30. How to sample from a Normal Distribution with known mean and variance?

**Chapter 14: Dimensionality Reduction and Visualization**

**14.1 	What is Dimensionality Reduction?**

We can visualize 2D and 3D using Scatter Plots; up to 6D we can leverage Pair Plots. For nD \> 6D we should reduce the dimensionality to make it an understandable or a visualizable dataset. 

Techniques: tSNE and PCA

By reducing dimensionality, we transform the features into a new set of features which are less in number. The aim lies in preserving the variance which ensures least possible loss of information. 

**14.2	Row Vector and Column Vector**

X\_i belongs to R(d) 🡪 X\_i is a d dimensional column vector and several data points are stacked in row wise to form a dataset. 

**14.3	How to represent a data set?**

D \= {x\_i, y\_i}, x\_i belongs to R(d), y\_i \= R(1)

x\_i: data points, y\_i: class labels

**14.4	How to represent a dataset as a Matrix?**

Matrix is like a table

D \= {x\_i, y\_i}, i \= \[1 to n\], x\_i belongs to R(d), y\_i class labels

x\_i is a column vector

In a matrix: each row is a data point and each column vector is a feature (preferred form of representation, this is one type of representation scheme)

Y is a column matrix in which each row corresponds to the class that the data point  x\_i belongs to.

**14.5	Data Preprocessing: Feature Normalization**

Data preprocessing: Mathematical transformations on Data to make it Machine readable improving its quality and reducing the dimensionality (sometimes).

Column Normalization: Take each column, corresponding to a feature. Take all values and do minimum subtraction and range division for each feature separately. 

- a\_max, a\_min: Calculated or generated  
- From a1, a2, a3, …, an make transformations to a1’, a2’, a3’, ….., an’ such that 				ai’ \= (ai – a\_min)/(a\_max – a\_min)  
- The feature normalization method transforms all the features to a range from 0 to 1\.  
- Column Normalization is needed to avoid different feature scales, Some of the variables can take values between 1 Million and 2 Million, while some variables or features can only take values between 0 and 10\. Thus, the second feature becomes insignificant to a model as larger values will effect largely during the predictions stage of the model.  
- Through column normalization we get rid of feature scales. This squishes all the data into a unit hyper cube.

**14.6	Mean of a data matrix**

x\_mean \= (1/n) \* sum (I \= 1 to n) x\_i

Geometrically: Project all data points onto each feature axis. The central vector of all these projections is the mean on each feature axis. The set of the means is the mean vector of the data matrix.

**14.7 	Data Pre-processing: Column Standardization**

Column Normalization: values are squished to a unit hyper-cube \[0, 1\], this gets rid of scales of each feature

Column Standardization: In a data matrix, with rows as data points and columns as features

a1, a2, a3, a4, …., an for each feature: are transformed to a1’, a2’, a3’, ….., an’ such that			ai’ \= (ai – a\_mean) / a\_std

With standardization, we will have mean \=0 and std\_dev \= 1

Column Standardization \= Mean Centering \+ unit scaling

Geometrically: Project all data points on feature axes; we can get mean and standard deviation of the dataset. After applying column standardization the data points will have mean at origin and the data points are transformed to a unit standard deviation.

Why standard deviation is important?

**14.8	Co-variance of a Data Matrix**

	X \= n x d matrix, its co-variance matrix S is of d x d size

	fj \= column vector – jth feature

	Sij \= ith row and jth column in co-variance matrix, is a square matrix

xij \= value of ith datapoint and jth feature

Sij \= cov(fi, fj)

i: 1🡪 d, j: 1🡪 d 

cov(x,y) \= (1/n) \* sum(1 to n) (xi – mean\_x)\*(yi – mean\_y)

cov(x,x) \= var(x) 	\-- 1

cov(fi, fj) \= cov(fi, fi)	\-- 2

In co-variance matrix, we have diagonal elements as variance values and the matrix is symmetric.

Let  X: column standardized, mean(fi) \= 0, std\_dev(fi) \= 1

Cov(f1, f2) \= (1/n)\*sum(1 to n) (xi1 \* xi2) \= (f1 . f2)/n

S (X) \= (XT X) \*1 /n: if X has been column standardized

**14.9	MNIST dataset (784 dimensional)**

	D \= { x\_i, y\_i }  i \= 1 to 60k

Each data point: x\_i \= Image (28 x28), y\_i \= {0, 1…., 9}

Dimensionality reduction: from 2D to 1D using Flattening operation

(28 x 28\) is converted into (784 x 1\) shape; by stacking all elements from each row into a column vector

We can use PCA or tSNE to visualize the dataset using a plot

**14.10	Code to load MNIST data set**

	**i**mport numpy as np

	import pandas as pd

	import matplotlib.pyplot as plt

	d0 \= pd.read\_csv(‘’) \#csv – comma separated values

	print(d0.head())

	l \= d0\[‘label’\]

	d \= d0.drop(“label”, axis \= 1\)

	

plt.figure(figuresize \= (7, 7))

	idx \= 100

	grid\_data \= d.iloc(idx).as\_matrix.reshape(28, 28\)

	plt.imshow(grid\_data, interpolation \= “name”, cmap \= “gray”)

	plt.show()

	print(l(idx))

**Chapter 15: PCA (principal component analysis)**

**15.1	Why learn PCA?**

	PCA – for dimensionality reduction \- R(d) to R(d’) where d’ \<\< d

1) For visualization  
2) d’ \< d for model training

**15.2 	Geometric intuition of PCA**

	![][image58]

If we want to reduce dimensionality from 2D to 1D, we can skip feature f1 as the spread across the feature is very less than spread across the feature f2.

Another case: Let y \= x be the underlying relationship between two features and let X be column  standardized, and the plot is such that there is sufficient variance on both features. Say, we rotate the axes and reach an orientation where spread on f2’ \<\< f1’ (perpendicular to f2’) then we can drop f2’. So data transformation can also find maximum variance features. This will help us reduce dimensionality. 

We want to find a direction f1’ such that the variance of the data points projected onto f1’ is maximum and skip features with minimum spread.

**15.3	Mathematical objective function of PCA**

	![][image59]

u1 : Unit vector: in the direction where maximum variance of projection of x\_i exists

|| u1||2 \= 1

xi‘ 	\= proju1 xi (projection of xi on u1)

	\= u1 . xi / || u1||2 \= u1T xi

xi‘ 	\= u1T xi

xi‘ (mean)	\= u1T xmean 

Task: find u1, such that var(proju1 xi) is maximum

var{u1Txi} (i \= 1 to n) \= (1/n) \* sum (u1Txi \- u1Txmean)2

When column standardized: 

var{xi’ \= u1Txi} (i \= 1 to n) \= (1/n) \* sum (u1Txi)2

Optimization:	Objective: max (1/n)\*sum (u1Txi)2 such that 

Constraint: u1T u1 \= 1 \= || u1||2

**15.4	Alternative formulation of PCA: Distance Normalization**

	Variance minimization: find u1 such that projected variance is maximum

	Alternatively, find u1 such that distances of the data points to u1 is minimum

	min sum(di2) for u1: u1 \= unit vector

![][image60]

Variance maximization:

Objective: max (1/n)\*sum (u1Txi)2 such that 

Constraint: u1T u1 \= 1 \= || u1||2

Distance minimization:

	Objective: min (||xi||2 – (u1Txi)2) \= min sum(xiTxi – (u1Txi)2)

Constraint: u1T u1 \= 1 \= || u1||2

Same u1 optimizes both formulations

**15.5	Eigen values and Eigen vectors (PCA): Dimensionality reduction**

	Solution to the optimization problems: λ1, V1

X: \[ …… \]n x d : column standardized: mean of each column \= 0 and std\_dev of each column \= 1

	Co-variance matrix of X \= S

	S dxd \= XdxnT Xnxd : S is square and symmetric

	Eigen value of S: λ1, λ2, λ3, λ4, …. (λ1 \> λ2 \> λ3 \> λ4 …. )

Eigen vectors of S: V1, V2, V3, V4, …

Def: If λ1V1 \= Sdxd V1 

λ1 is a scalar which is the Eigen value of S and V1 is the Eigen vector corresponding to λ1

Eigen vectors are perpendicular to each other. This implies, that dot product of pairs of Eigen vectors is zero. ViTVj \= 0

u1 \= V1 \= Eigen vector of S (= XTX) corresponding to largest Eigen vector (=λ1)

Steps:

1. X: Column Standardized  
2. S \= XTX  
3. λ1, V1 \= eigen(s)  
4. u1 \= V1 direction with maximum variance

   λ1/ sum(λi) \= % of variance preserved on u1

	V1 max variance, V2 second max var, V3 third max var, …

**Eigen Vectors provide direction of maximum variance, Eigen values provides percentage of variance preserved with each Eigen value**

	![][image61]

**15.6	PCA for Dimensionality Reduction and Visualization**

	For X: Let us have S, V1, and V2; using eigen(s) function

Transform xi \= \[f1i, f2i\] to xi’ \= \[xiTV1, xiTV2\], we can drop xiTV2 as there is not much variance: 2D data point is transformed to 1D

If we want to preserve say k% of variance: 

Find d’ such that sum(i \= 1 to d’) λi / sum(i \= 1 to d) λi \= k% where d is dimensionality of the input data point.

**15.7	Visualize MNIST dataset**

	[https://colah.github.io/posts/2014-10-Visualizing-MNIST/](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)

**15.8	Limitations of PCA**

Loss of information: cases for hyper-spheres, separable data points on high dimensionality can get crowded at several locations making it inseparable at low dimensionality

![][image62]

**15.9	PCA code example**

	MNIST dataset:

	Step 1: Data preprocessing

		\- Standardization: using StandardScaler() \= X, mean centering and scaling

		\- Get (1/n)\*XTX – Co-variance matrix

\- values, vectors \= eigh(covar\_matrix, eigvals \= (782, 783)) \# gives top 2 values as co-variance matrix is of 784 x 784 size

		\- vectors \= vectorsT

		\- X’ \= vectors\*X

	Using PCA:

		\- from sklearn import decomposition

		\- pca \= decomposition.PCA()

		\- pca.n\_components \= 2

		\- pca\_data \= pca.fit\_transform(sample\_data)

		\- pca\_data \= np.vstack(pca\_data.T, labels)).T

**15.10	PCA for dimensionality reduction (not-visualization)**

d' \= 2 or 3 is for visualization, but if d’= 200 or any other value, then we are doing data reduction for non-visualization tasks.

Stack Eigen vectors horizontally to get Eigen matrix V. Multiply X with V to get X’ which is dimensionally reduced. d' can be determined by checking explained variance. The task is to maximize variance.

Explained variance \= % of variance retained \= 100\*sum(i \= 1 to d’) λi / sum(i \= 1 to d) λi

Take pca.n\_components \= 784

pca\_data \= pca.fit\_transform(sample\_data)

percentage\_var\_explained \= np.cumsum(percentage\_var\_explained)

**Chapter 16: (t-SNE) T-distributed Stochastic Neighborhood Embedding**

**16.1	What is t-SNE?**

	One of the best techniques for data visualization: tSNE

	PCA – basic, old: did not perform well on MNIST dataset visualization

Other techniques: Multiple Dimensional Scaling, Sammon Mapping, Graph based techniques

tSNE vs PCA: 

![][image63]

PCA preserves the global structure of the dataset and discards local structure. tSNE also preserves local structure of the dataset. 

**16.2 	Neighborhood of a point, Embedding**

	For a d-dimension space: 

Neighborhood points of a data point are the points that are geometrically close to the data point. 

Embedding: For every point in the original dataset, a new point is created in lower dimension corresponding to the original data point. Embedding means projecting each and every input data point into another more convenient representation space (Picking a point in high dimensional space and placing it in a low dimensional space).

**16.3	Geometric intuition of t-SNE**

Let us assume: x1, x2, x3, x4 and x5 are some data points, and let x1, x2 and x3 be in a neighborhood region, x4 and x5 be another neighborhood region and both neighborhood regions are very far from each other. When we reduce the dimensionality we follow the objective of preserving the distance between data points. 

		dist(x1, x2) \~ dist(x1’, x2’)

	For points which are not in neighborhood the distances are not preserved.

		Relation between dist(x1, x4) and dist(x1’, x4’) is ignored.

Link: [https://www.youtube.com/watch?v=NEaUSP4YerM](https://www.youtube.com/watch?v=NEaUSP4YerM)	

![][image64]

![][image65]

**![][image66]**

![][image67]

Link: [https://www.youtube.com/watch?v=ohQXphVSEQM](https://www.youtube.com/watch?v=ohQXphVSEQM)

![][image68]

pj|i \= conditional probability of two points being in neighborhood in high dimension, 

qj|i is for low dimension

![][image69]

KL divergence is asymmetric: gives large cost for representing nearby data points in high dimensional space by widely separated points in low dimensional space. 

Perplexity indicates number of neighbors considered during analysis. 

![][image70]

For optimization, we can use Gradient Descent. 

**16.4	Crowding Problem**

Non-neighbor data points can also be seen to crowd with neighborhood data points. 

t-SNE preserves distances in a Neighborhood. It is impossible to preserve neighborhood of all the data points. 

**16.5	How to apply t-SNE and interpret its output?**

	Link: [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)

Processes all the data in multiple iterations and tries to separate different neighborhoods. 

	Parameters: 

Step: 		Number of iterations

Perplexity: 	Loosely indicates number of neighbors to whom distances are preserved.

Perplexity: 	Run tSNE with multiple perplexity values. As perplexity increases from 1, the neighborhood tries to get good clusters and then with further increments the clustered profile becomes a mess (more number of data points are considered to belong to same neighborhood). 

Stochastic part of tSNE induces probabilistic embedding, results change every time tSNE is run on the same data points with same parameter values. 

tSNE also expands dense clusters and shrinks sparse clusters, cluster sizes cannot be compared. tSNE does not preserve distances between clusters.  

Never rely on the results of tSNE, experiment it with changing parameter values. You may need to plot multiple plots to visualize. 

![][image71]

**16.6	t-SNE on MNIST**

	MNIST: 784-dim data of images of hand written Numbers between 0 and 9\.

	tSNE grouped data points or images based on visual similarity;

**16.7	Code example of t-SNE**

	Model \= TSNE(n\_components \= 2, random\_state \= 0\)

	Tsne\_data \= Model.fit\_transform(data\_1000)

	Plot: perplexity \= 30, iter \= 1000

		![][image72]

	Perplexity \= 50, iter \= 5000, data points \= 1000 

		![][image73]

	

**16.8	Revision Questions**

1. What is dimensionality reduction?  
2. Explain Principal Component Analysis?  
3. Importance of PCA?  
4. Limitations of PCA?  
5. What is t-SNE?  
6. What is the Crowding problem?  
7. How to apply t-SNE and interpret its output? 

**Chapter 17:	Interview Questions on Dimensionality Reduction**

**17.1	Questions & Answers**

1. You are given a train data set having 1000 columns and 1 million rows. The data set is based on a classification problem. Your manager has asked you to reduce the dimension of this data so that model computation time can be reduced. Your machine has memory constraints. What would you do? (You are free to make practical assumptions.)([https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/](https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/))

   Ans.	Close all other applications, random sample dataset, reduce dimensionality (remove correlated features, Use correlation for Numerical and chi-square test for categorical, Can use PCA)

2. Is rotation necessary in PCA? If yes, Why? [https://google-interview-hacks.blogspot.com/2017/04/is-rotation-necessary-in-pca-if-yes-why.html](https://google-interview-hacks.blogspot.com/2017/04/is-rotation-necessary-in-pca-if-yes-why.html) 

   Ans.	Yes, rotation (orthogonal) is necessary because it maximizes the difference between variance captured by the component. This makes the components easier to interpret. Not to forget, that’s the motive of doing PCA where we aim to select fewer components (than features) which can explain the maximum variance in the data set. By doing rotation, the relative location of the components doesn’t change; it only changes the actual coordinates of the points.

   If we don’t rotate the components, the effect of PCA will diminish and we’ll have to select more number of components to explain variance in the data set.

3. You are given a data set. The data set contains many variables, some of which are highly correlated and you know about it. Your manager has asked you to run PCA. Would you remove correlated variables first? Why?([https://www.linkedin.com/pulse/questions-machine-learning-statistics-can-you-answer-saraswat/](https://www.linkedin.com/pulse/questions-machine-learning-statistics-can-you-answer-saraswat/))

   Ans.	Since, the data is spread across median, let’s assume it’s a normal distribution. We know, in a normal distribution, \~68% of the data lies in 1 standard deviation from mean (or mode, median), which leaves \~32% of the data unaffected. Therefore, \~32% of the data would remain unaffected by missing values.

