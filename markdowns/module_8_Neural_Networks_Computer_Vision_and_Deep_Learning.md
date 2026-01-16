**Module 8:		Neural Networks, Computer Vision and Deep Learning**

---

**50**

**DEEP LEARNING: NEURAL NETWORKS.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**50.1		HISTORY OF NEURAL NETWORKS AND DEEP LEARNING.**

---

* Neural Networks are an important Machine Learning Modelling framework  
* First simplest Neural Network Model built was a Perceptron. Built in 1957 by Rosenblatt  
* [https://en.wikipedia.org/wiki/Perceptron](https://en.wikipedia.org/wiki/Perceptron)  
* With small changes a perceptron can become a logistic regression  
* After WW2, in US people were trying to translate messages from Russain to English  
* Contributors \- Alan Turing(father of modern computing)  
  * Curiosity raised these questions:  
    * What is intelligence?  
    * How should we build it artificially?  
* There was a biological inspiration  
* Due to work in NeuroScience, a vague understanding of the working of brain has been developed

	**Biological Neuron vs Artificial Neuron**

![][image212]

	Biological Neuron						Artificial Neuron

A biological Neuron has Nucleus, Dendrites and a cell body. When a Neuron gets electrical signals as inputs, the neuron does computations inside and it sends the processed electrical signals as output maybe to other neurons.

An Artificial Neuron has some inputs that are important, thus weights are included on edges of the connections.

	Output \= f(W1x1 \+ W2x2 \+ W3x3 \+ ……)

Perceptron is this single neuron which is loosely inspired from the biological neuron and is not an exact replica but is powerful enough to solve many interesting problems.

In biology, a neuron does not exist on itself. It is connected to other neurons. A structure of neurons can be considered for imagination (network). First successful attempt was made in 1986 by a group of mathematicians (Hinton and others). They came up with backpropagation (chain rule around differentiation). A lot of hype has been generated.

Unfortunately around 1996, due to insufficient computational power, insufficient data and lack of algorithmic advancements AI experienced a long winter. This is shortly called AIWinter. Funding for AI got exhausted due to hype. Neural Networks couldn’t take off in the 90s. People shifted to SVM, RandomForest and other GBDTs between 1995 to 2009, which were giving solutions to many problems.

Hinton in 2006 released a paper on how to train a Deep Neural Network. Before this NN was limited to a small number of layers. As the number of hidden layers increased backpropagation failed.

DNN took developments with a competition on ImageNet Dataset. The task was to identify objects in images. DNN has performed very well by a large margin on this dataset compared to other classical ML algorithms.

Noticeable Applications: Siri, Cortana, Alexa, Google Assistant, Self Driving Cars, Health care purposes

The development is also driven by the availability if Data, computational power and new algorithms.

---

**50.2		HOW BIOLOGICAL NEURONS WORK?**

---

\- Biological Neuron: 

Biological Neuron

Simplified view of Neuron

Lof of internal chemical activity takes place in a biological neuron, but we can understand its working with a simple structure (as above). We have biochemistry generated electrical signals.

Each neuron gets connected to a bunch of other neurons. Some dendrites are thicker. This leads to more weight for that input. A neuron is activated or fired if there is enough input.

f(x) is the activation function that transforms input to a desired output. 

Output y \= f(sum(Wi\*xi)) \= f( W1x1 \+ W2x2 \+ W3x3 \+ …)

We are applying activation function on summation of weighted inputs.

Output dendrites will also have different weights.

The main purpose of the activation function is to add non-linearity in the network so that the model can learn difficult objective functions. 

In ensembles we are training different models and we are combining them on specific conditions. In NN, all neurons are learning at the same time based on the loss function we have.

Generally we use a nonlinear activation function. For regression problems we use a linear activation function in the last layer.

---

**50.3		GROWTH OF BIOLOGICAL NEURAL NETWORKS**

---

Let us consider the growth of a neural network in the human brain. At birth, there are far few connections between neurons. Missing connections can be thought of having weights equal to 0\. At age 6, the network gets dense, i.e. weights get trained. At age 60, weights or connections of some edges disappear or become thin, this process is termed as Neural degeneration. 

By age 6, humans learn: language, object recognition, sentence formation, speech, etc. (massive amounts of learning). Biological learning is basically connecting neurons with edges. New connections are formed based on data (not random).

---

**50.4		DIAGRAMMATIC REPRESENTATION: LOGISTIC REGRESSION AND				PERCEPTRON**

---

	Logistic regression: a plane separating positive points from negative points.

	LR: given x\_i predict y\_i

		y\_i\_hat \= sigmoid(W.T\*x\_i \+ b)

	Given a dataset: D \= {x\_i, y\_i}, we find W and b while training LR

	Let x\_i E R(d) → W E R(d)

	X\_i \= (x\_i1, x\_i2, ……, x\_id)

	W \= \[W1, W2, ….., Wd\]

y\_i\_hat \= sigmoid (summ(j \= 1 to d)(W\_j \* x\_ij \+ b))

	\= f(summ(j \= 1 to d)(W\_j \* x\_ij \+ b))

![][image213]

Sigmoid (logistic) activation function: Representation of Logistic Regression with a simple neuron.

If the activation function is sigmoid, a simple perceptron becomes an LR model.

Given dataset of {x\_i, y\_i}

	Task is to find W and b: trained by SGD

Training a neural network → finding weights on edges.

- **Perceptron** \- 1957 \- Rosenblatt

(difference between logistic regression and perceptron is in the activation function)

For perceptron: f(x) \=\[1 if W.T \* x \+ b \>0; 0 otherwise\]

![][image214]

Perceptron

![][image215]

Perceptron activation function

A perceptron can also be understood as a linear function trying to find a line that separates two classes of datapoint.

In logistic regression we have squashing through sigmoid. In perceptron we have no squashing. We just have a step function as an activation function.

LR, perceptron → Simple Neural network

		→ difference in activation functions

---

**50.5		MULTI-LAYERED PERCEPTRON (MLP).**

---

	Perceptron: 	a simplified single neuron

			Looks alike Logistic Regression

	An experiment with connecting multiple neurons was done.

	![][image216]

Bunch of single neurons stacked to form a layer and layers are stacked to form a network of neurons.

Q. Why should we care about the Neural network??

- Biological inspiration: connected structure like as in brains, to get true performance (making them work took decades)  
- Mathematical argument: Given: D: {x\_i, y\_i}  
  - task : find f where f(x\_i) \= y\_i (belongs to R)  
  - Ex: D \= {x\_i, y\_i}  
  - Let x\_i belong to R(1 dim) and y\_i belong to R(1 dim)  
  - Let f(x) \= 2\*​sin(​x^​2)+​(5\*​x)^​0.5

![][image217]

- Case: with a simple function y \= f(x) \= x  
  - We will get a 45 degree line passing through origin  
  - For  2\*​sin(​x^​2)+​(5\*​x)^​0.5 we require a complex function, a single linear regression is not sufficient  
- Understanding why MLPs are powerful:

Let f1 → add(); f2 → square(); f3 → sqrt(); f4 → sin(); f5 → mul()

		![][image218]

f (x)	\= 2sin(x^2) \+ (5x)^0.5 \= f1(2sin(x^2), (5x)^0.5 ) \= f1( f5(2,sin(x^2)), f3(5x)) 

\= f1(f5(2,f4(x^2)), f3(f5(5,x)) \= f1(f5(2,f4(f2(x))), f3(f5(5,x)))

This can be thought of as a function of functions. Thus with MLP we can have complex functions to act on x to get y. Having MLPs we can easily overfit, thus regularizers are applied to avoid overfit. MLP is a graphical way of representing functional compositions. 

---

**50.6		NOTATION**

---

	MLPs generate powerful models.

	Let D \= {x\_i, y\_i}; x\_i belongs to R(4) and the problem is a regression problem

	f\_ij → function of i layer j neuron

	o\_ij → output of i layer j neuron

	W\_k\_ij → from i neuron, to j neuron, to k layer

	![][image219]

---

**50.7		TRAINING A SINGLE-NEURON MODEL.**

---

Training → to find best edge weights in a neural network model.

Perceptron and Logistic Regression: Single neuron model for classification

Linear Regression: single neuron model for regression

- Linear Regression: y\_i\_hat \= W1\*x\_i1 \+ W2\*x\_i2 \+ W3\*x\_i3 \+ …… \= W.T\*x\_i

	Activation function f is identity in linear regression

	f(z) \= z

	For logistic regression f \= sigmoid, for perceptron f \= step function

Linear regression: Optimization problem:

min (Wi) summ(y\_i \- y\_i\_hat) ^2 \+ regularization

	Y\_i\_hat  W.T \* x\_i

D \= {x\_i, y\_i} (i \= 1 to n); x\_i E R(4), y\_i E R : regression

1. Define Loss function: L \= summ(y\_i \- y\_i\_hat) ^2 \+ regularization

			L-i \= (y\_i \- y\_i\_hat)^2

![][image220]

2. Optimization problem definition:

   min (W\_i) summ(y\_i \- f(W.T \* x\_i))^2 \+ reg

3. Solve the optimization  
- Usin SGD  
- Initialize the weights → random initialize and initialize eta  
- Key step \- **Compute derivative of loss function w.r.t weights**

  ![][image221]

- W\_new \= W\_old \- η(∇wL)W\_old

	Gradient Descent → Compute ∇wL using all x\_i’s and y\_i’s

	Stochastic Gradient Descent → Compute ∇wL using all x\_i and y\_i

Mini-batch SGD → Compute ∇wL using a batch of x\_i’s and y\_i’s

Computing ∇wL: Using chain rule:

Using path from W1 to L:

 ![][image222]

![][image223]

![][image224]

---

**50.8		TRAINING AN MLP: CHAIN RULE**

---

\- Backpropagation used for training

D \= {x\_i, y\_i}

x\_i belongs to R(4), y\_i belongs to R, square loss L(y\_i, y\_i\_hat)

![][image219]

![][image225]

(Consider path of flow of feature)

1. Define loss function: 	![][image226]  
2. Optimization: 			min L (find weights that minimize the loss function)  
3. SGD →   
   1. Initialize WK\_ij randomly  
   2. ![][image227] (η \= Learning rate)  
   3. Perform Updates (step 2\) upto convergence

      Until: W\_new \~ W\_old

In the notation (figure), we have W311 impacting O31 and O31 impacting L

*  ![][image228]  
* ![][image229]  
* ![][image230]  
* ![][image231]

  (When computing gradients for backpropagation using chain rule, consider path of flow of the feature)

---

**50.9		TRAINING AN MLP: MEMOIZATION**

---

In Computer Science we have a powerful idea called as Dynamic Programming. This helps us calculate the value of a variable only once. Compute anything only once and store that in a dictionary for reuse.

While computing gradients, gradients such as ![][image232]can be seen to occur multiple times. Without storing its value we end up re-calculating its value again and again. This will impact the time of computation of all the gradients  in an MLP. With memoization we calculate the value of each and every gradient only once. This will avoid repeated calculations while keeping run time minimum. 

	**Chain rule \+ Memoization → Back Propagation**

---

**50.10		BACKPROPAGATION.**

---

Given D \= {x\_i, y\_i} 

1. Initialize weights → randomly (there are effective initialization methods available)  
   * Input x\_i  
2. For each x\_i in F  
   * Pass x\_i forward through network  
     * Get y\_i\_hat   
     * This is forward propagation  
   * Compute loss 2(y\_i, y\_i\_hat)  
   * Compute derivatives using chain rule and memoization  
   * Update weights from end to start

     ![][image233]

	Forward propagation → Using x\_i to calculate y\_i and L

	Backward propagation → Using L to update weights

		Both combine to form an epoch

3. Repeat step 2 until convergence, i.e. W\_new \~ W\_old

		1 epoch of training → pass all of the data points through the network once.

Backprop: ‘init weights’, ‘in each epoch (forward prop, compute ;pss, compute derivative(chain rule \+ memoization), Update weights’, ‘repeat till convergence’

- Backprop works only if activation function is differentiable (required to update weights)  
  - Speed of training depends on computing the derivatives.  
  - A faster way is to pass a batch of input data points to speed up training.  
    - Batch size depends on RAM capacity  
    - Generally we use batch size \= 8, 16, 32, 64,... to take advantage of the RAM architecture

---

**50.11		ACTIVATION FUNCTIONS**

---

	Most popular (Pre-Deep Learning \- 1980s and 90s):

- Sigmoid and tanh

	Sigmoid: ![][image234] \[Used in Logistic regression for squashing values\]

	tanh: ![][image235]

Activation functions should be differentiable and should be faster to compute

1. ![][image236]

![][image237]

2. Tanh, derivative: 1- tanh sq.

![][image238]

Sigmoid and tanh:

	→ differentiable

	→ faster to compute gradients

ReLU became most popular activation function as sigmoid and tanh resulted in vanishing gradients

---

**50.12		VANISHING GRADIENT PROBLEM**

---

Due to chain rule, multiplication of derivatives which are \<1 will result in the overall derivative to be small.

	→ This will not help weight updates (no training)

	→ W\_new \~ W\_old

- This happened due to sigmoid and tanh activation functions  
- People were not able to train deep neural networks due  to this vanishing gradients problem.  
- Exploding gradients happen when each gradient is \>\>\>1  
  - Weight update will be noisy  
  - Large update → no convergence

    → training does not stop

	Vanishing gradients \- no weight updates

	Exploding gradient \- no convergence

(With sigmoid function, which results in values between 0 and 1, exploding gradients occur when weights are greater than 1\)

---

**50.13		BIAS-VARIANCE TRADEOFF.**

---

1. As number of layers increase, we will have more weights, leading to overfit → low bias, high variance  
2. Logistic Regression model \- fewer weights (compared to MLP) \- leads to underfitting \- high bias  
- MLPs typically overfit on train dataset  
- We avoid this using Regularizers (L1 or L2 or Dropouts)  
  - We will have ‘lambda’ coefficient of regularization term as a hyperparameter. Also number of layers is a hyperparameter  
- Using regularization, we reduce variance

---

**50.14		DECISION SURFACES: PLAYGROUND**

---

Site: playground.tensorflow.org \- A web browser interface to understand variations due to change in hyperparameters.

**51**

**DEEP LEARNING: DEEP MULTI-LAYER PERCEPTRONS**

---

**51.1		DEEP MULTI-LAYER PERCEPTRONS:1980s TO 2010s**

---

Until 2010, people were trying to build 2 to 3 layered networks due to: vanishing gradients, limited data (easy to overfit, no weight updates \-\> no training), limited compute power. 

By 2010 we got lots of data (Internet, labelled data (quality)), compute power has increased (Gaming GPUs \- NVIDIA \- found to be suitable for Deep Learning), advancements in algorithms. This paved way for modern Deep Learning achievements.

With classical ML (Mathematician approach), theory was first built and then proved through experiments. With DL it became possible to experiment (cheap) with different ideas first (Engineer Approach) and then develop theory. 

\>  Why not use Deep Learning for every problem?

Sometimes DL works better sometimes ML works better, it is generally experimented to see whose performance is good. 

Depending on data, if the data is simple (such as having an underlying Linear Relationship) using ML and DL will be the same and adding DL which is compute expensive does not give advantage over ML, while with complex data DL may work better. 

But EDA and other experiments are required to be performed to decide which Algorithm will perform well. 

\> So in short, mlp's are very complex mathematical functions with different variables(features) and weights are their coefficients such that we will find optimum values for weights which can accurately predict the result. Since weights are adjusted to the patterns in the train data and could predict the output if a similar pattern(data point) has been injected into it. 

Small dataset leads to overfit when: 

* Increasing the number of hidden units and layers  
* Increasing number of epochs

  \> Neural Networks handle outliers by using batch training and robust cost functions that include regularization.

---

**51.2		DROPOUT LAYERS & REGULARIZATION.**

---

Deep NN \-\> generally occurring problem in overfitting; Regularization such as L1 and L2 can be used, dropout is prefered regularization equivalent technique.

Concept of dropout can be linked to the sampling of a subset of columns in RandomForest. Each tree looks at a random subset of features. **Regularizing through a Random subset of features.** Base learners are high variance and low bias models. Through regularization we are reducing variance. 

![][image239]

Random Neuron Dropout per epoch with a dropout rate (percentage of neurons dropped)

At training time neuron is present with probability ‘p’, at test time all neurons are present but to account of probability of presence, multiply weights of dropout with p;

High dropout rate \-\> low keep probability \-\> underfitting \-\> Large regularization

Low dropout rate \-: high keep probability \-\> overfitting \-\> small regularization

---

**51.3		RECTIFIED LINEAR UNITS (RELU).**

---

* Classical NNs: Vanishing gradients: small updates  
* ReLU: becam by default activation function:

![][image240]

* f(Z) \= max(0,Z)   
* f’(Z) \= {0 if Z \<0 or 1 if Z\>0) (No vanishing gradient or No exploding gradients but there is dead activation below 0 (need to monitor dead activations))  
* Relu function is non-linear (as function changes at 0), not differentiable at 0\.  
* As derivative computing is easy and computationally   
* ReLU converges faster than tanh (due to absence of vanishing gradients)

* As ReLU is not differentiable at 0, a smooth approximation for ReLU is defined to take care of non-differentiability problem  
* Softplus: f(x) \= log(1 \+ exp x);   f’(x) \= sigmoid(x) ![][image241]  
* ReLU is simpler and computationally efficient. Derivatives can be computed with if else block.  
* Leaky ReLUs: for z\<=0 f(z) \= 0.01z, f’(z) \= {1 for z\>0, 0.01 else} (can result in vanishing gradients) used when a lot of relu activated neurons become dead. 

* ReLU Advantages:  
  * Faster convergence  
  * Easy to compute  
* ReLU limitations:  
  * Non-differentiable at zero  
  * Unbounded  
  * Dying relu  
  * Non-zero centered  
1. Why non-linear activations?

   If the activation functions are linear, a deep neural network will be similar to a single layer neural network which cannot solve complex problems. While non-linearity gives a good approximation of underlying relations.

2. When do we use the sigmoid function?

   Generally, at the last output layer in case of binary classification problems, Softmax in the output layer is used for multi-class classification problem, linear for regression. 

---

**51.4		WEIGHT INITIALIZATION.**

---

* Logistic Regression: initialization weights randomly, from a random normal distribution  
* If init all weights to zero or to same value: if activation functions across all neurons in a layer are same, all neurons compute the same thing, same gradient updates no training \-\> problem of symmetry  
* If weights distribution is asymmetric, all the neurons will learn different aspects of the input data  
* If init all weights to large \-ve numbers: ReLU(z) will be equal to 0 (normalizing data is mandatory: mean centering, variance scaling): we will have vanishing gradients  
* For sigmoid activation at extremities we have vanishing gradients

	a:	Normal distribution

* Weights should be small (not too small)  
* Not all zero   
* Good-variance (each neuron will learn differently)  
* Weights are random normally initialized

	

	Can we have better init strategies?

(fan\_in for a neuron \= number of input connections, fan\_out \= number of output connections: Weights are suggested to be initialized by fan\_in and fan\_outs, no concrete agreement which works well)

b:	Uniform initialization:

* Weights are initialized to a Uniform distribution   
  * Unif\[-1/sqrt(fan\_in),1/sqrt(fan\_in)\], selection of each value is equally likely

c:	Xavier/Glorot init (2010) \- useful for sigmoid activations

* Normal: Initialize to a mean centered normal distribution with variance (sigma sq.) \= 2/(fan\_in \+ fan\_out)  
* Uniform: Initialize to   
  * \~Unif\[(-sqrt(6)/sqrt(fan\_in+fan\_out), sqrt(6)/sqrt(fan\_in+fan\_out)\]  
* Using fan\_in and fan\_out also

	d. 	He init \- ReLU activations

* Normal: Initialize to a mean centered normal distribution with variance (= sigma sq.) \= 2/(fan\_in)  
* Uniform: \~Unif\[(-sqrt(6)/sqrt(fan\_in), sqrt(6)/sqrt(fan\_in)\]

---

**51.5		BATCH NORMALIZATION.**

---

* Let us take: an MLP fully connected,   
* Generally, we perform pre-processing of data by normalization (mean centering and variance scaling), even then for a deep MLP, we input a mini batch of data repeatedly, a small change in input leads to a large change in the last layer of a Deep Neural Network (though everything looks good near input layer)  
* Thus normalization is used to reduce changes deep in the network  
* This problem is called Internal co-variance shift: “change in the distribution of network activations due to the change in network parameters during training”  
* BN: normalization with batch mean and batch variance and then **scaling and shifting**. Parameters gamma, beta: gamma\*(x\_norm) \+ beta; these are learnt by backpropagation.   
* BN: faster convergence, allows using large learning rates, weak regularizer and avoids internal covariance shift

  BN on test set: for training: batch mean and variance, for test: train population mean and variance

**Additional material:** [https://arxiv.org/pdf/1502.03167v3.pdf](https://arxiv.org/pdf/1502.03167v3.pdf)

---

**51.6		OPTIMIZERS: HILL-DESCENT ANALOGY IN 2D**

---

- Objective was to minimise loss functions (Optimization approach)  
- In neural networks, Gradient Descent cannot solve all problems very well

	Working principle of optimizers using: Hill \-descent analogy

- Task: min L(w)

		Case 1: W \= Scalar

- If L(w) is y \= x\*\*2/4a:

				There is no maxima and has only one minima at 0

- If L(w) is a complex:

  We can have local minima, local maxima, global minima and global maxima, at minima or maxima we have zero gradient

			Update function: W\_new \= W\_old \- eta (grad(L,W))

Saddle point: gradient is also zero, but not a minima nor a maxima 

Zero gradient implies minima, maxima or saddle point

SGD could get stuck at saddle point

- Convex and Non-convex functions: Convex have one minima or maxima

				Non-convex have multiple local minima or local maxima

**Logistic Regression, Linear Regression and SVM functions: have loss functions that are convex, for DL-MLPs: loss functions are non-convex thus initialization of weights determine the minima or maxima we reach, depending on the initialization we can reach a global minima or local minima**

**Additional Material:**

- Squared loss is used as its derivative is a linear function which will have a unique solution, while derivatives of higher order loss functions will lead to having more than one solution or multiple minima or maxima locations, Squared loss can generally be preferred as it will only have one minima or maxima

Why does DL have non-convex loss functions?

Loss function of DL models: dice loss, cross-entropies, MSE. These are defined by the programmer, mostly non-convex functions are used as their performances are better and the choice depends on architecture and the network output or our desired target values. Loss functions represent the Network’s output function. 

	

---

**51.7		OPTIMIZERS: HILL DESCENT IN 3D AND CONTOURS.**

---

	Case 2: Let W be 2 dimensional (like a catching net)

		Let loss function be convex

Contour: project all points of a 3D plot at same height to a plane, visually indicates location of maxima and minima

(Graphically shown)

![][image242]

**Simple SGD will get stuck at saddle point, local maxima and local minima**

---

**51.8		SGD RECAP**

---

	Update: W\_k\_ij\_new \= W\_k\_ij\_old \- eta\*(grad(L,W\_k\_ij\_old))

	Generalized form: W\_t \= W\_t-1 \- eta\*grad(L, W\_t\_1)

	grad(): computed using all points: Gradient Descent

		computed using one point (random): Stochastic GD 

		computed using k points (random sample): mini-batch SGD (preferred)

grad(): mini-batch SGD \~ GD (mini batch SGD approximation to GD, and is compute efficient) (mini-batch SGD is a noisy descent)

	Task: denoise gradients for SGD

---

**51.9		BATCH SGD WITH MOMENTUM.**

---

SGD updates are noisy as compared to GD updates, 

Task: denoise SGD updates:

	With SGD we have gradients at each time step as: a1, a2, a3, …

- a\_i is an original gradient calculated using sgd, v\_i is exponential average of these gradients  
1. Applying average method of denoising (r \= gamma): 

   V1 \=a1, V2 \= gamma\*V1+a2, V3 \= gamma\*V2+a3, ...

   a1, r\*a1+ a2, r\*( r\*a1 \+ a2) \+ a3, …..: a.k.a exponential weighted average

   W\_t \= W\_t-1 \- eta\*grad (mini batch SGD)

   Exp. weighting: V\_t \= r\* V\_t-1 \+ eta \* g\_t

   Recent gradients has more influence

   If gamma \= 0, mini batch SGD

   If gamma \= 0.9, W\_t \= W\_t-1 \- eta\*V\_t-1 \- eta\*(g\_t). 

   Momentum adds to convergence by travelling more distance towards optimum by getting to know the sense of direction of movement

---

**51.10		NESTEROV ACCELERATED GRADIENT (NAG)**

---

		Previous Lecture: SGD \+ momentum: exponential gain to create momentum

		Gradient: eta \* g\_t, Momentum: gamma\*V\_t-1

SGD: g \= grad(L,W) \[W\_t-1\]

NAG: First move in momentum direction, then compute gradient and then move in gradient direction: W\_t \= W\_t-1 \- gamma \* V\_t-1 \- eta\*g’

NAG: g’ \= grad(L,W) \[W\_t-1 \- gamma \* V\_t-1\]

![][image243]

![][image244]

*Figure made using Paint*

NAG update ends up slightly closer to the optimum, convergence becomes faster compared to SGD \+ momentum.

---

**51.11		OPTIMIZERS:ADAGRAD**

---

* Adagrad: Introducing adaptive learning rate for each weight  
* Some features in data points can be sparse.  
* SGD: W\_t \= W\_t-1 \- eta\*g\_t: eta same for all weights  
* Adagrad: W\_t \= W\_t-1 \- eta**’**\_t\*g\_t: (eta**’**\_t \= different eta for each weight for each iter)  
* eta**’**\_t \= eta\_t/(sqrt(alpha\_t \+ epsil.)  
* alpha\_t \= sum\[i \= 1 to t-1\](g\_i\*\*2)     : g\_i \= grad(L,W)\[W\_t-1\]  
* aplha\_t \>= alpha\_t-1  → eta**’**\_t \<= eta**’**\_t-1  
* As ‘t’ increases, the learning rate (for each weight) decreases. Learning rates are different for each weight and this learning rate decay happens adaptively in the case of Adagrad

	Advantage: 

+ Adaptive learning rate: no manual decay needed  
+ Right learning rates for different features

	Disadvantage:

- alpha increases as t increases, becomes a large value which makes learning rate small, resulting in negligible updates → no training

	Learning AdaGrad makes it easy to understand other adaptive optimizers

Why different learning rates for sparse features?

	Link: [https://youtu.be/c86mqhdmfL0](https://youtu.be/c86mqhdmfL0) 

	Sparse features makes gradient small (due to summation on all data points)

Update does not happen, thus eta or the Learning rate requires to be larger for these sparse features, while for dense feature gradient will not be small. Simply put, constant learning rates will only help the weights to update with contributions from dense features and not from sparse features. Weights updates for sparse features will be negligible as gradients will be small which demands for a higher learning rate. If a constant higher learning rate is used, the weights are updated with large steps and convergence will never happen. Thus we require different learning rates. **Sparse features demand large learning rates, while dense features require smaller learning rates.**

---

**51.12		OPTIMIZERS : ADADELTA AND RMSPROP**

---

Previous Lecture: Adagrad’s alpha can become very large resulting in slow convergence.

* eta**’**\_t \= eta\_t/(sqrt(alpha\_t \+ epsil.)

		alpha\_t \= sum\[i= 1 to t\] (g\_i \*\*2)

	Adadelta: W\_t \= W\_t-1 \- eta**’**\_t \* g\_t

	eta**’**\_t \= eta / (sqrt(eda\_t \+ eps.))

eda(0) \= 0

eda\_t \= gamma \* eda\_t-1 \+ (1 \- gamma) \* (g\_t-1) \*\*2

	gamma \= 0.95, generally

Adadelta: take exponential weighted averages of gradient of squares instead of simple sum to avoid large alphas avoiding slow convergence. 

---

**51.13		ADAM**

---

	Adaptive Moment Estimation: most popular DL optimizer, 

	In Adadelta we are using eda(g\_i squares)

	Storing eda of (g\_i);

	Analogy in Statistics: Mean: first order of moment

			Variance: second order of moment

	m\_t \= beta\_1 \* m\_t-1 \+ (1 \- beta\_1) \*g\_t

	v\_t \= beta\_2 \* v\_t-1 \+ (1 \- beta\_2) \* (g\_t \*\*2)

	m\_t\_hat \= m\_t / (1 \- beta\_1\*\*t)

	v\_t\_hat \= v\_t / (1 \- beta\_2\*\*t)

	w\_t \= w\_t-1 \- alpha \* (m\_t\_hat/ sqrt( v\_t\_hat \+ eps.))

	t \= time step

		

![][image245]

![][image246]

---

**51.14		WHICH ALGORITHM TO CHOOSE WHEN?**

---

Type of Optimizers:

Mini batch SGD, NAG, Adagrad, Adadelta, RMSProp, Adam we came across

	Source: [Link](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f)

Considering getting stuck at saddle points or at local minima or local maxima (loss does not change):

- Mini Batch SGD works well for Shallow NN.  
- NAG works well for most cases but slower than Adagrad, Adadelta and Adam  
- Adagrad \- sparse data  
- Adam works well in practice \- favored optimizer \- fastest algorithm for convergence  
- For sparse data adaptive learning rates are required.

---

**51.15		GRADIENT CHECKING AND CLIPPING**

---

When we train MLPs, we need to monitor Weight updates. Thus we need to check gradients (each epoch and each weight). This will help us detect vanishing gradients which easily occur in the first few layers due to farness from the output layer (gradients become too small or very large as we move far from the output layer towards the input layer). 

For exploding gradients: we can use gradient clipping: All the weights or gradients are stored in a single vector. 

For example we can have an L2 norm clipping:

	All the gradients in the gradient vector are divided by the vector’s L2 norm (sum of squares of all gradients). This will clip the gradients to 1\. And multiplying these with a threshold value we clip all the gradients to the threshold value. 

	Rule of thumb: Monitor Gradients to ensure training is proper

---

**51.16		SOFTMAX AND CROSS-ENTROPY FOR MULTI-CLASS						CLASSIFICATION.**

---

Logistic Regression: Sigmoid activation: binary classification

For multi-class classification using Logistic Regression we use One versus Rest method. But can we do something else, like extending the basic math of Log. Reg. Extension of Logistic regression to Multi-Class classification results in softmax. Recap: Output of the Logistic regression network is the probability of y\_i \= 1 given x\_i. 

P(y\_i \= 1|x\_i) \= 1/(1+ e\*\*(-W.T \*x). 

![][image247]

	Summation of probabilities \= 1 in Softmax activation

	Softmax is a generalization of logistic regression for multi class problems. 

	![][image248]

sigma\_1(z\_1) \= P(y\_i=1|x\_i); sigma\_2(z\_2) \= P(y\_i=2|x\_i); 

sigma\_3(z\_3) \= P(y\_i=3|x\_i); sigma\_4(z\_4) \= P(y\_i=4|x\_i);

Summation \= 1

![][image249]

In Logistic Regression, we optimize log loss for binary classification, 

where L\_i \= y\_i\*log(p\_i) \+ (1-y\_i)\*log(1-p\_i)

In Softmax, we optimize multi-class log loss: ![][image250]

	y\_ij is 0 for all classes other than true class which has y\_ij \= 1

Regression: Squared Loss, 2 class classification: 2 class log loss, k-class classification: Multi class log loss

---

**51.17		HOW TO TRAIN A DEEP MLP?**

---

* **Preprocess data: Data Normalization**  
* **Weight Initialization:**   
  * **Xavier/Glorot for sigmoid or tanh**  
    * **He init for ReLU**  
      * **Gaussian with reasonable variance**  
* **Choose right Activation functions: ReLU (and its variations: avoids vanishing gradients)**  
* **Add Batch Normalization layers for Deep MLPs for later layers: helps you tackle internal co-variance shifts, Use Dropouts for regularization in deep MLPs**  
* **Optimizer choice: Adam general choice**   
* **Hyperparameters: Architecture:Number of layers, number of neurons, dropout rate, optimizer hyperparameters**  
* **Loss function: Log loss or Multi-class Log loss or Square loss (easy differentiable)**  
* **Monitor gradients (Apply gradient clipping if needed)**  
* **Plot Loss vs epoch and other metric plots**  
* **Remember to avoid Overfitting (Early Stopping, Dropouts, etc.)**

---

**51.18		AUTO ENCODERS.**

---

Neural network which performs dimensionality reduction (better than PCA and tSNE sometimes) (tSNE tries to preserve the neighborhood)

![][image251]

Dataset: D \= {x\_i}	x\_i E R(d) : an unsupervised problem

To convert D to D’ such that X\_i E R(d’ \<\< d)

Given x\_i: as we reduce the number of neurons in the next layer dimensionality reduces. 

Let us take a simple encoder with three layers: Input Layer: 6 neurons, Hidden: 3 neurons, Output: 6 neurons

x\_i with 6 dimensions is given as input, it is compressed to 3 dimensionality in the hidden layer, then at output layer it increases the dimensionality to original input. If the output is similar to the input then the representation of the input at the hidden layer is good. This implies that the reduction in dimensionality has preserved information of the input. 

Thus if we have input \~ output in an auto encoder, then we have the compression part reducing the dimensionality without losing variance or information. 

Denoising autoencoder: Even though we have noise in input, at the output we will get a denoised representation of the input as the dimensionality is reduced in the intermediate hidden layers. 

Sparse autoencoder: Getting sparse output by using L1 regularizer

If linear activations are used or a single sigmoid hidden layer, the optimal solution to an autoencoder is closely related to PCA.

---

**51.19		WORD2VEC : CBOW**

---

	Ways of Featurizing text data: BOW, TFIDF, W2V

	Ex: The cat sat on the wall.

		Focus word: sat: The, cat, on, the, wall are context words

		If Focus word: cat: The, sat, on, the, wall are context words

	Idea: context words are useful for understanding focus words and vice-versa.

	There are two algorithms: CBOW, SkipGram

	

CBOW: Continuous Bag of Words

We have a dictionary/vocab; Use one hot encoding for each word based on vocab length. Core idea behind CBOW: Given context words can we predict focus words. Linear or Identity activation is used. Softmax is used at the output layer. 

Structure:

![][image252]

Take all of the text dataset, create focus word \- context words dataset, train the neural network with above structure on this dataset. 

---

**51.20		WORD2VEC: SKIP-GRAM**

---

CBOW: Predict focus word given context words

Skip-gram: Predict context words given focus word

Input: v-dimensional one hot encoded focus word. Hidden layer with N dimension with linear activations. Use multi output Softmax layers which give context words as output corresponding to each output layer that are stacked to form a single output layer (k softmax layers). 

\# of weights: 	CBOW: (K+1) (NxV), 1 softmax: performs well for frequently occuring words

Skipgrams: (K+1) (NxV), k softmax: slow training: performs well for less occuring words.

---

**51.21		WORD2VEC :ALGORITHMIC OPTIMIZATIONS.**

---

	CBOW and SkipGrams: Millions of weights: time taking

Hierarchical Softmax: Using binary trees to predict a word.

Negative sampling: Update a sample of words: Update weights to all the target words and weights of some of non target words (selection of non target words is based on probability value corresponding to the occurrence of the word).

**52**

**DEEP LEARNING: TENSORFLOW AND KERAS.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.1		TENSORFLOW AND KERAS OVERVIEW**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

	Tensorflow/ Keras: Tool kits or libraries that enable us to code for Deep Learning

	Tensorflow: 	most popular DL library, open sourced by Google, by Nov, 2015

- Helps researchers, for developers and for deployment engineers also (two different tasks)

- Core of Tensorflow was written in C/C++ for speed, they made interfaces available in Java, Python and JavaScript

- Tensorflow Lite can run on Android for deployment on Phone.

- Tensor: Mathematical term for vectors, 1D vector : 1D tensor, 2D vector : 2D tensor, 3D vector : 3D tensor

- Deep Learning is all about tensor operations

- Flow may be inspired from forward and backward propagation (flow of data)

- It gives lot of low-level control of models

- With Keras: simple (similar to SKLearn); high level NN library: faster for developing and deploying models; less lines of code

- Keras is front end: Tensorflow is backend; other backend: Theano, Pytorch, Caffe, MXNET

- Tensorflow: Research easy, Keras: Development easy

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.2		GPU VS CPU FOR DEEP LEARNING.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

	Graphics Processing Unit: Became suitable for Deep Learning 

	CPU: Central Processing Unit

	

	GPUs: 1990s; for gaming (Graphic cards); 

![Image result for cpu vs gpu][image253]	

RAM is connected to the CPU through the motherboard, Cache is like RAM on chip, The processors can communicate with Cache faster than with RAM, With multiple cores multiple calculations are done at time parallely. Around 2GHz speed

GPU: Multiple processors: 1024, 512, etc.Each GPU core is slower than a CPU core, around 400MHz speed. Every unit has a processor and a cache.This is a distributed structure. Sum of all cache in GPU is called VRAM (Video RAM). It takes data from RAM, distributes it across all units, and the processor unit works very fast on its corresponding cache data. 

GPUs are fast if we have parallelizable tasks such as doing Matrix operations as each of the calculations are not dependent among themselves. Because of these characteristics of GPUs, Deep Learning is experiencing developments.

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.3		GOOGLE COLABORATORY.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Provided by Google for Machine Learning education and research;

Link: colab.research.google.com 

Similar to Jupyter Notebook	

Cloud computing: shared computing resources; bunch of computers tied together for ready access over internet

Google Colab computers are extremely powerful

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.4		INSTALL TENSORFLOW**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

	[https://www.tensorflow.org/install/install\_windows](https://www.tensorflow.org/install/install_windows) 

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.5		ONLINE DOCUMENTATION AND TUTORIALS**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

* [https://www.tensorflow.org/get\_started/](https://www.tensorflow.org/get_started/) 

* [https://learningtensorflow.com/](https://learningtensorflow.com/)

* [https://cloud.google.com/blog/products/gcp/learn-tensorflow-and-deep-learning-without-a-phd](https://cloud.google.com/blog/products/gcp/learn-tensorflow-and-deep-learning-without-a-phd)

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.6		SOFTMAX CLASSIFIER ON MNIST DATASET**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

	Using Tensorflow:

		MNIST dataset: Given image predict its class

1. Get data from Tensorflow datasets

   2. Import Libraries, check GPU, CPU capacities

   3. Placeholder: memory location 

   Placeholders and Variable:

   x \= tf.placeholder(tf.float32, \[None, 784\])

   Constant: cannot be changed

   Variable: Updated during training or other computations

   A placeholder can be imagined to be a memory unit; where values of some data points are stored for computations.

   W and b are variables

   * W \= tf.variable(tf.zeros(\[784,10\]))

   * b \= tf.variable(tf.zeros(\[10\]))

   * y \= tf.nn.softmax(tf.matmul(x,W)+b)

   * y\_ \= tf.placeholder(tf.float32, \[None.10\])

   * cross\_entropy \= tf.reduce\_mean(-tf.reduce\_sum(y\_\*tf.log(y), ...)

   * reduce\_sum: takes tensor of 2D reduces to 1D and applies sum

   * then training, etc.

   * sess \= tf.InteractiveSession() : initiates a computational graph

   * tf.global\_variables\_initializer().run()

   * Run loops over batches of training dataset

   * Metrics evaluations

   * Termination criteria: loss directed termination

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.7		MLP: INITIALIZATION**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

For MNIST dataset

a. Importing libraries

b. Dynamic plot: at every epoch

c. Architecture used: 784 (FC) – 512 (FC) – 128 (Softmax) – 10

d. Placeholders, Variables, Weight initializations, 

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.8		MODEL 1: SIGMOID ACTIVATION**

**52.9		MODEL 2: RELU ACTIVATION.**

**52.10		MODEL 3: BATCH NORMALIZATION.**

**52.11		MODEL 4 : DROPOUT.**

**52.12		MNIST CLASSIFICATION IN KERAS**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

[https://drive.google.com/file/d/1tIEOPJiJtMzStFai47UyODdQhyK9EQnQ/view](https://drive.google.com/file/d/1tIEOPJiJtMzStFai47UyODdQhyK9EQnQ/view) 

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**52.13		HYPERPARAMETER TUNING IN KERAS.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

1. **Definition of keywords**

   Hyperparameter: 	A parameter of learning algorithm that is not of the model, these remain constant during model training. The values of other parameters are derived from training. Example of hyperparameter: number of neurons in each layer. Example of model parameter: learning rate. Given hyperparameters the training algorithm learns the parameters from training.

   	[https://en.wikipedia.org/wiki/Hyperparameter\_(machine\_learning)](https://en.wikipedia.org/wiki/Hyperparameter_\(machine_learning\))

   Hyperparameter tuning: Through tuning we derive a tuple of hyperparameters that yields an optimal model which minimizes a predefined loss function on given test data. It is the procedure through which we derive hyperparameters that yields an optimal model. This model is further trained on the training set to update weights. 

2. **Detailed explanation**

Neural Networks are so flexible that they a drawback in terms of hyperparameters which can be tweaked. A simple MLP can have lots of hyperparameters: Number of layers, number of neurons per layer, type of activation function to use in each layer, weights initialization logic, and so on. One option is to try different combinations of hyperparameters and see which combination works best on validation set. Use SKLearn’s GridSearchCV or RandomizedSearchCV. Wrap keras models to mimic regular SKLearn classifiers. SKLearn tunes hyperparameters for maximum score by default. Thus loss functions needs to be transformed into a metric. This way of tuning is generally time consuming. Efficient toolboxes such as hyperopt, hyperas, skopt, keras-tuner, sklearn-deep, hyperband and spearmint are available for this purpose. As the search space is huge, follow the following guidelines to restrict search space:

	For number of hidden layers: Slowly add number of hidden layers from 1 to 50 or 100 until overfitting on training set. Above this use, transfer learning. 

Number of neurons per layer: For input and output layers it is predetermined by data. Randomly pick a number which is power of 2\. Use early stopping and regularization to prevent overfitting. It is generally preferred to increase number of layers over number of neurons per layer. Avoid having too few layers to have enough representational power of the data. 

Learning rate of loss function: Change learning rate from 10\-5 to 101 at an interval of epochs. Plot loss function and select the LR that is just behind the value which shows minimum loss value. If minimum is found at 1 use 0.1 as LR. 

Optimizers, batch\_size and activation functions have a fixed set of choices. While with number of epochs use a large number and use early stopping to stop training. 

Especially if after choosing LR, you tweak a hyperparameter, LR should again be tuned. A best approach is to update Learning Rate after tweaking all other hyperparameters.

3. **Video Lecture**

Multi Layered Perceptrons have a lot of hyperparameters.

1. Number of Layers  
2. Number of activation units in each layer  
3. Type of activation: relu, softmax  
4. Dropout rate, etc.

How do you do hyperparameter tuning for MLP?

* Scikit-learn has two algorithms for this purpose: GridSearchCV and RandomSearchCV, we used them a lot in Machine Learning assignments  
* Keras models require to be connected with the above algorithms. Build DL models in keras and use SKLearn’s hyperparameter tuning algorithms


  

| Code: \# Hyper-parameter tuning on type of activation of Keras models using Sklearn from keras.optimizers import Adam,RMSprop,SGD def best\_hyperparameters(activ):     ‘’’     Defining model: This function returns a model      with activation type \= input string     Input: string     Output: model     ‘’’     model \= Sequential()     model.add(Dense(512, activation=activ, input\_shape=(input\_dim,),                      kernel\_initializer=RandomNormal(mean=0.0,                      stddev=0.062, seed=None)))     model.add(Dense(128, activation=activ,                      kernel\_initializer=RandomNormal(mean=0.0,                      stddev=0.125, seed=None)))     model.add(Dense(output\_dim, activation='softmax'))     model.compile(loss='categorical\_crossentropy', metrics=\['accuracy'\],                     optimizer='adam')     return model | Procedure: We are tuning only activation type (comparing Relu and Softmax) Assume: all important libraries are downloaded, datasets loaded and everything is ready for hyperparameter tuning Model is defined using a function best\_hyperparameters(). This is a sequential model. We are using MNIST dataset which has 784 columns. This model has 4 layers. Input layer has 784 neurons. There are two hidden layers with 512 and 128 neurons respectively. The last layer is the output layer with number of neurons \= output\_dim. Type of Activation is the input to this function.  Model is compiled with categorical\_cross\_entropy with metrics \= accuracy and optimizer \= ‘adam’ The actication choices are softmax and relu. Softmax is just a multi-valued version of sigmoid.  |
| :---- | :---- |
|  \# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/ activ \= \['softmax','relu'\] from keras.wrappers.scikit\_learn import KerasClassifier from sklearn.model\_selection import GridSearchCV model \= KerasClassifier(build\_fn=best\_hyperparameters, epochs=nb\_epoch,                          batch\_size=batch\_size, verbose=0) param\_grid \= dict(activ=activ) \# if you are using CPU \# grid \= GridSearchCV(estimator=model, param\_grid=param\_grid,                     n\_jobs=-1) \# if you are using GPU dont use the n\_jobs parameter grid \= GridSearchCV(estimator=model, param\_grid=param\_grid) grid\_result \= grid.fit(X\_train, Y\_train)  |  Activation type choices: ‘softmax’ and ‘relu’. Wrapper makes connection between keras model and Scikit-learn tuner. We are using KerasClassifier function for this purpose. This makes a wrapping over Keras model and makes it readable by Scikit-learn. KerasClassifier takes model defined which gets compiled. Its arguments include epochs and batch\_size. A parameter grid is created in dictionary format. GridSearchCV is defined with KerasClassifier as estimator with a parameter grid. Fit function is called to evaluate the model on each choice of the hyperparameter. Here, activation is hyperparameter for hidden layers  and its choices are softmax and relu. The GridSearchCV hyperparameter tuning goes through each of the parameter in the parameter grid.  |
| print("Best: %f using %s" % (grid\_result.best\_score\_,                               grid\_result.best\_params\_)) means \= grid\_result.cv\_results\_\['mean\_test\_score'\] stds \= grid\_result.cv\_results\_\['std\_test\_score'\] params \= grid\_result.cv\_results\_\['params'\] for mean, stdev, param in zip(means, stds, params):     print("%f (%f) with: %r" % (mean, stdev, param))  | Best model is printed.  |
| \#Output: Best: 0.975633 using {'activ': 'relu'} 0.974650 (0.001138) with: {'activ': 'softmax'} 0.975633 (0.002812) with: {'activ': 'relu'}  | Relu gave a slightly better accuracy on this dataset. Keras is internally using Tensorflow |


  Other tools: Hyperopt, Hyperas(Variation of Hyperopt) – Hyperparameter tuning toolboxes which can work with keras and tensorflow. 

- SKLearn is used due to prior experience in GridSearchCV and RandomizedSearchCV. Sufficient for most practical purposes.   
- If you use CPU, take n\_jobs \= \-1 gives faster results. For GPU, ignore n\_jobs argument.

4. **Summary of comments**  
   1.  It is softmax and relu the choices for activation function.

   

   2. Keras comes with the new hyper parameter tuning library. It is better than keras.wrappers.  
      Please go through the video of sentdex and documentation  
      https://keras-team.github.io/keras-tuner/  
      https://github.com/keras-team/keras-tuner  
      https://www.youtube.com/watch?v=vvC15l4CY1Q

   

   3. In classical ML we plotted the train acc. and cv acc. both to check overfitting of our model. While in GridSearchCV or Talos, we just get the parameters for which CV acc. is highest. Then, how are we sure that we're not overfitting?  
      I mean suppose I get a hyperparameter set (h1) from GridSearch with highest cv acc. \= 86 % but the train acc. for h1 \= 99%. Now, if another hyperparameter set (h2) exists such that cv acc. \= 84% and train acc. \= 87%.  
      Now, acc. to what we learnt in classical ML, h2 is a better set of hyperparameters as cv acc. of h2 \~ cv acc. of h1 but the difference between the train and cv acc. for h2 (3%) is much less than that for h1 (13%). Hence, h2 performs better generalization on unseen data. But, from clf.best\_estimators() we're going to get h1 as it has the highest cv acc.  
      How to deal with this problem?

   **Ans.**  You have to consider that hyperparameter which gives highest CV accuracy in cross validation. Whichever gives highest CV score, that will be the best

5. **Additional material**  
1. [https://github.com/autonomio/talos](https://github.com/autonomio/talos)  
2. We have one of our students build a case-study and write a blog on this exact topic. It's a two-part blog that you can read here:  
   Part I: https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-i-hyper-parameter-8129009f131b  
   Part II: [https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7)  
3. About Hyperparameter Tuning in Keras: [https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53](https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53)  
4. [https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53](https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53)  
5. [https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

6. **QAs**  
   1. What can go wrong if you tune hyperparameters using the test set?  
   2. Can you list all hyperparameters you can tweak in a basic MLP? How could you tweak these hyperparameters to avoid overfitting?

**53**

**DEEP LEARNING: CONVOLUTIONAL NEURAL NETS.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.1		BIOLOGICAL INSPIRATION: VISUAL CORTEX**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Most popular for visual tasks: images; example: MNIST, Object Recognition...

In 1981, Nobel prize 🡪 Hubel and Wiesel 🡪 Research on Visual perception

They found that a certain neurons get fired to certain image features; in human brains certain visual areas have specialized functional properties. 

* Some neurons	in the visual cortex fire when presented with line at specific orientation

* Different regions of the brain are responsible for edge-detection, motion, depth, color, shapes, faces

* There is hierarchical structure amongst these different regions, layers stacked

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.2		CONVOLUTION: EDGE DETECTION ON IMAGES.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Primary visual cortex performs Edge detection. Here we are going to understand CNN through edge detection.

Say we have a 6x6 matrix with 3 rows black and 3 rows white. Thus we have an edge at third-fourth row interface. Grayscale image pixel intensities \- 0: black and 255: white. 		

  \*  

	Sobel/edge detection (horizontal)

Conv.

With convolutions, we have element wise multiplications and summation over them to get an element for a new matrix. The scope of the multiplications is determined by the size of the filter. 

We will have a new matrix of size (**n-k+1, n-k+1**): n \- size of input matrix, k – size of filter matrix

The above Sobel filter is used for detecting horizontal edges. Transpose of this matrix can be used to detect vertical edges. 

Output Matrix of above convolution: 

| 0 | 0 | 0 | 0 |
| :---: | :---: | :---: | :---: |
| \-1020 | \-1020 | \-1020 | \-1020 |
| \-1020 | \-1020 | \-1020 | \-1020 |
| 0 | 0 | 0 | 0 |

Normalization of this matrix: Min-max

| 255 | 255 | 255 | 255 |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 255 | 255 | 255 | 255 |

Link: [https://en.wikipedia.org/wiki/Sobel\_operator](https://en.wikipedia.org/wiki/Sobel_operator) 

Such distinguished filters are applied to get different features of images for visual tasks in CNN.

Machine Learning: dot products on vectors 

Convolution: Element wise multiplications and addition, dot products on matrices (generally)

Convolution can be applied on vectors also

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.3		CONVOLUTION: PADDING AND STRIDES**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

In previous topic, we had an input image of size 6x6 and the output array is of size 4x4. But if we want to have output array of a different size, we should go for padding which will add a row and column, top, bottom, left and right.  But zero padding will generate extra edges. We can have same value padding.  With padding of size p, we will have final matrix of size 		**(n-k+2p+1, n-k+2p+1)**. 

Strides will help us skip rows and columns by a value equal to strides (s). We will have an output matrix of size (**int( (n-k)/s) \+ 1, int( (n-k)/s) \+ 1**)

Convolution: 	element wise multiplication and addition

Padding:	add rows and columns to get a desired output size

Strides: 	reduce size by a large factor

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.4		CONVOLUTION OVER RGB IMAGES**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Each pixel in an RGB image contains a tuple of three values

![https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Beyoglu\_4671\_tricolor.png/800px-Beyoglu\_4671\_tricolor.png][image254]

It can also be thought of having three images stacked one over the other; resulting in a 3D Tensor; these multiple images are called as channels, So each image will have n x m x c size where c is the number of channels; Convolution filter will be a 3D tensor; care should be taken for image processing that the number of channels in filter will be same as the number of channels in the image; Convolution on a 3D image (n x n) with a 3D filter (k x k)  results in a 2D array (output image) of size (n-k+1, n-k+1)

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.5		CONVOLUTIONAL LAYER**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Link: [http://www.iro.umontreal.ca/\~bengioy/talks/DL-Tutorial-NIPS2015.pdf](http://www.iro.umontreal.ca/~bengioy/talks/DL-Tutorial-NIPS2015.pdf), [http://cs231n.github.io/convolutional-networks/\#pool](http://cs231n.github.io/convolutional-networks/#pool)

We learnt: Convolution, Padding and Strides

Recap: In MLP, we have weights; input and bias values, the dot product of these values are filtered through an activation function

* Convolution layer – biologically inspired to process visual tasks

* Multiple edge detectors: multiple kernels

* In CNN we train the models to learn kernel matrices by back-propagation

* At each convolution layer we will have multiple kernels to learn different features of the images. For each kernel we will have a 2D array output, multiple kernels result in multiple output arrays (padded to get input array size), so at the layer we will get an output array of size n x n x m, where m is number of kernels (m is a hyper parameter)

* For every element in the output of filtering, the activation function is applied

* Pad, convolve and activate to transform an input array to an output array in a convolution layer

* Multiple layers of convolutions are used, at each layer we will extract features:

  * In image recognition: Pixels 🡪edge 🡪….. 🡪 part 🡪 object

  * Text: character 🡪word 🡪 word group 🡪 sentence 🡪 story

  * Speech 🡪 sample 🡪 spectral brand 🡪 sound 🡪 … 🡪 phoneme 🡪 word

* MLP and convolution layers have similarity in terms of weights: kernels, while we train the models to learn weights in MLPs, we train the models to learn kernels in Conv Nets

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.6		MAX-POOLING.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Pooling introduces: Introduces small amount of Location invariance, Scale invariance and Rotational invariance

Pooling subsamples input arrays to reduce computational load, memory usage and number of parameters (limiting the risk of overfitting)

Pooling is destructive; sometimes invariance is not desirable, 

We can also have a goal of equivariance, where a small change in input array should also reciprocate a small change in output (invariance: a change in input image does not show changes in output array)

Global Average Pooling: computes mean of entire map; gives out a single scalar

Additional Material: [https://www.slideshare.net/kuwajima/cnnbp](https://www.slideshare.net/kuwajima/cnnbp)

![3   /14   Neural   Network   as   a   Composite   Func4on A   neural   network   is   decomposed   into   a...][image255]

![∇W J W,b;x, y( ) =∂∂WJ W,b;x, y( ) =∂J∂z∂z∂W= δ z( )xT∇bJ W,b;x, y( ) =∂∂bJ W,b;x, y( ) =∂J∂z∂z∂b= δ z( ...][image256]

![5   /14   Decomposi4on   of   Mul4-­‐Layer   Neural   Network n  Composite   func.on   representa.on   of   a...][image257]

![6   /14   Error   Signals   and   Gradients   in   Mul4-­‐Layer   NN n  Error   signals   of   the   square...][image258]

![7   /14   Backpropaga4on   in   General   Cases 1.  Decompose   opera.ons   in   layers   of   a   neural   ...][image259]

![9   /14   Deriva4ves   of   Convolu4on n  Discrete   convolu.on   parameterized   by   a   feature   w   and...][image260]

![10   /14   Backpropaga4on   in   Convolu4on   Layer Error   signals   and   gradient   for   each   example ...][image261]

![https://qphs.fs.quoracdn.net/main-qimg-fadd1fcd4f6a4788e6113e28a6c312bc][image262]

![12   /14   Backpropaga4on   in   Pooling   Layer Error   signals   for   each   example   are   computed   b...][image263]

![13   /14   Backpropaga4on   in   CNN   (Summary) Plug   in   δ(conv) Plug   in   δ(conv) … ∂J/∂Wn xn x...][image264]

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.7		CNN TRAINING: OPTIMIZATION**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

In MLPs, back propagation can be applied; 

Convolution layer: Similar to MLP \- del z/ del W \= x

Max pooling back propagation: gradient propagation to only the maximum value, say from a 2x2 matrix we max pool to get a 1x1 scalar, when back propagate through max pooling gradient in the 2x2 matrices are either 0 or 1, it 1 where the maximum is present and 0 everywhere else. Gradient \= 1 because the same value is pulled as output (maximum valued element and non-max values have no effect on the output).

Optimization is similar to an MLP

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.8		RECEPTIVE FIELDS AND EFFECTIVE RECEPTIVE FIELDS**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Receptive field: when passing the image through a filter at any time the pixels at which the filter is being applied is the receptive field at that instant. 

![][image265]

Effective Receptive field: In a Deep CNN, at a deeper layer a filter focuses on the pixels of input image. This region of focus on which layers of convolutions are applied is an effective receptive field. In layer 2, the filter has receptive field in the output of first layer and effective receptive field in the input image.

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.9		EXAMPLE CNN: LENET \[1998\]**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Link: [https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/Model%20&%20ImgNet/lenet.html](https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/Model%20&%20ImgNet/lenet.html)

Concepts of techniques are old but we didn’t have datasets and compute power;

![https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/img/lenet.png][image266]

LeNet: Small depth

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.10		IMAGENET DATASET**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Link: [https://en.wikipedia.org/wiki/ImageNet](https://en.wikipedia.org/wiki/ImageNet)

Contributed the most to Deep Learning; This dataset became the benchmark dataset for all new DL algorithms

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.11		DATA AUGMENTATION**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

We want CNN models to be robust to input image changes, such as translation, scaling, mirror, etc. Thus, the input image is passed through data augmentation generating new images. For example: (using matrix transformation)

![][image267]

Link: [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)

Through data augmentation, a new dataset is created (through transformations)

Can introduce invariance in CNN, create a large dataset when we have small datasets.

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.12		CONVOLUTION LAYERS IN KERAS**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Links:	[https://keras.io/layers/convolutional/](https://keras.io/layers/convolutional/)

	[https://keras.io/layers/pooling/](https://keras.io/layers/pooling/)

	[https://keras.io/layers/core /](https://keras.io/layers/core%20/)

Flatten: Converts a 2D array into a 1D vector

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.13		ALEXNET**

**53.14		VGGNET**

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Developments in Deep Learning training on Imagenet dataset;

LeNet: 2 Conv, 2 Mean Pool, 3 FC – 1000s trainable params, Sigmoid Activation: First Architecture for Hand written images classification

![Image result for lenet][image268]

AlexNet: 5 Conv, 3 Pool, 3 FC – 10M trainable params, ReLU activation, Includes Dropouts: Trained on ImageNet

![Image result for alexnet][image269]

VGGNet: (2Conv+1MaxPool)\*2 \+ (3Conv+1MaxPool)\*3 \+ 3FC \+ Softmax

![Image result for vggner][image270]

Additional: 

- The **Softmax classifier** gets its name from the **softmax** function, which is used to squash the raw class scores into normalized positive values that sum to one, so that the cross-entropy loss can be applied

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**53.15		RESIDUAL NETWORK.**

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

ResNets: Residual Networks: Regular networks as depth increases both training error and test error are increasing. This was tackled using ResNets. Skip connections: Output of previous layer is given as input to the next layer also in addition to giving it as input to the current layer.

ReLU(x) \= x

So as increasing number of layers has an effect of increasing error, ResNets skip connection concept will ensure that some of the layers that are useless will be neglected (type of regularization). With ResNets we can add additional layers such that performance does not get effected. If the new layers are useful, performance will increase as skipping will not happen. 

Input array dimensions should match while using skip connections. ResNets are used to avoid reduction of performance when number of layers increases.

![Image result for resnet block][image271]

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**53.16		INCEPTION NETWORK.**

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Inception Network: Using multiple filters at a layer and stacking output of each filter to result in a single output. This will help us take advantage of different filters or Kernel sizes. Number of computations is large for each filter. As we increase number of kernels at each layer we will have very large computations. It is of the order of Billion computations at each layer. Having an intermediate layer of size 1x1 before each filter will reduce the number of computations.

[http://www.ashukumar27.io/CNN-Inception-Network/](http://www.ashukumar27.io/CNN-Inception-Network/) 

![Image result for inception block][image272]

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**53.17		WHAT IS TRANSFER LEARNING**

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Transfer Learning: Utilizing performance of an already trained neural network to work on a new dataset instead of building an NN from scratch to solve the Visual tasks. Pre-trained models are readily available on Keras and Tensorflow. The pre-trained model can be trained on one dataset and can be applied on to another related dataset. A model trained on cars can be applied to recognize trucks.  Note that it takes a lot of time to train a model from scratch. 

Link: [http://cs231n.github.io/transfer-learning/](http://cs231n.github.io/transfer-learning/) 

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. The three major Transfer Learning scenarios look as follows:

* **ConvNet as fixed feature extractor**. Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features **CNN codes**. It is important for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.  
* **Fine-tuning the ConvNet**. The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.  
* **Pretrained models**. Since modern ConvNets take 2-3 weeks to train across multiple GPUs on ImageNet, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. For example, the Caffe library has a [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) where people share their network weights.

**When and how to fine-tune?** How do you decide what type of transfer learning you should perform on a new dataset? This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images). Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, here are some common rules of thumb for navigating the 4 major scenarios:

1. *New dataset is small and similar to original dataset*. Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.  
2. *New dataset is large and similar to the original dataset*. Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.  
3. *New dataset is small but very different from the original dataset*. Since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.  
4. *New dataset is large and very different from the original dataset*. Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

**Practical advice**. There are a few additional things to keep in mind when performing Transfer Learning:

* *Constraints from pretrained models*. Note that if you wish to use a pretrained network, you may be slightly constrained in terms of the architecture you can use for your new dataset. For example, you can’t arbitrarily take out Conv layers from the pretrained network. However, some changes are straight-forward: Due to parameter sharing, you can easily run a pretrained network on images of different spatial size. This is clearly evident in the case of Conv/Pool layers because their forward function is independent of the input volume spatial size (as long as the strides “fit”). In case of FC layers, this still holds true because FC layers can be converted to a Convolutional Layer: For example, in an AlexNet, the final pooling volume before the first FC layer is of size \[6x6x512\]. Therefore, the FC layer looking at this volume is equivalent to having a Convolutional Layer that has receptive field size 6x6, and is applied with padding of 0\.  
* *Learning rates*. It’s common to use a smaller learning rate for ConvNet weights that are being fine-tuned, in comparison to the (randomly-initialized) weights for the new linear classifier that computes the class scores of your new dataset. This is because we expect that the ConvNet weights are relatively good, so we don’t wish to distort them too quickly and too much (especially while the new Linear Classifier above them is being trained from random initialization).

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.18		CODE EXAMPLE: CATS VS DOGS**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

[https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.19		CODE EXAMPLE: MNIST DATASET**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

IPYNB

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**53.20	Interview question: How to build a face recognition system from scratch?**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

From scratch: we need around 1000 images per person. 

If dataset size is less, use pre-trained models with fine tuning on around 100 images per person

Collect data in various conditions 🡪 Data augmentation 🡪 Transfer learning 🡪 Use Categorical log loss (cross entropy with Softmax) 🡪Try cloud APIs for faster training (Microsoft Azure, Google Images, etc.) 

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54**

**DEEP LEARNING: LONG SHORT-TERM MEMORY (LSTMS)**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.1		WHY RNNS?**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Suited for sequences of data, of words, etc.; in most sentences the sequence becomes important in addition to presence of a word

In all vectorization methods of textual data which work on the occurrence of a word, they discard the semantic meaning or sequence information; Machine Translation: sequence of French sentence to English; Speech recognition

In time series data, at new instant we have an observation. (Stock market, ride pickups)

In image captioning, we have output as sequence

Also sentences can have different lengths; we can zero pad all sentences to a common length. But this is memory inefficient, high number of parameters. 

RNNS: Need: each input is of different length, number of parameters should be less

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.2		RECURRENT NEURAL NETWORK**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

Recurrent: repeating

Ex: ![Image result for RNN][image273]

Let us take a Task: Classify sequence of words into 0 or 1

X1 – (X11, X12, X13, X14, X15)

At t \=1, X11 is given input to a layer, this gives an output O1 that is taken as input for the same layer at t \= 2\. O1 depends on input at t \=1 that is on X11. O2 depends on input at t \= 2, X12 and also on O1. 

O1 \= f(W\*X11), O2 \= f(W\*X12 \+ W’\*O1), O3 \= f(W\*X13 \+ W’\*O2), …

Let O5 be the output of final time stamp. On O5 we will apply an activation function to get y\_i\_hat. This activation function will have a Weight Matrix. 

Three weight matrices: W for input, W’ for previous state output and W’’ before the final activation function. To create this as a repetitive structure, we can have a dummy output (O0) before the first input. Weights can be initiated using Xavier/ Glorot method.

**To get output we apply an activation function to the output of the desired time step cell.**

![][image274]🡪RNN

The same neurons are used over the whole sequence of words of the sentence, the RNN layer is generally confused to be of a number of layers at different time steps, which is wrong.

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.3		TRAINING RNNS: BACKPROP**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

For RNNs Back propagation through time is used. Back propagation is unrolled over time.

RNN: Only one layer repeating over time

For gradient computations follow the arrows

![][image275]

![][image276]

All the weights are same (there are only three weight matrices)

At the end of back propagation, multiplications are large which results in vanishing or exploding gradients. The number of layers may be low but the because of recurrence of the computations which depends on the length of sequences.

Note \- We’re not running into the vanishing gradient/exploding gradient problems because of lots of layers, it’s because **we’re performing backprop over time.**

**Our sequences are long and we’re taking one word after another as input.**

So I got the hang of these RNN Back prop through time concept: for cases such as one to many or many to many. It is just that we calculate del l/ del w at each time step and update W. While calculating the gradient we follow all the paths from Loss function to the weight matrices. For ex. Del l / del w \= **(** del l / del y2\_hat \* del y2\_hat/ del On \* del On/ del O(n-1) … \* del Ok/ del W **)** \+ **(** del l / del y1\_hat \* del y1\_hat/ del O(n-1) \* del O(n-1)/ del O(n-2) … \* del Ok/ del W **)** \+ other outputs on which Loss function is depended on.

**54.4		TYPES OF RNNS**

Many to one RNN (sentiment classification); 

Many to many: same length: Parts of speech, 

Many to many: different length: Machine Translation; 

One to many: Image captioning (Input: Image matrix, Output: Sequence of words)

One to one: MLP

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.5.		NEED FOR LSTM/GRU.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**Video Lecture**

What is the problem with Simple RNNs?

            Many to many same length RNN:

![][image277]

            yi4 depends a lot on xi4 and O3 and depends less on xi1 and O1 due to vanishing 	gradients

            Simple RNNs **cannot take care of long term dependency**, when 4th output depends a lot on 1st input

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.6		LSTM.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

![][image278]

**Lecture:**

LSTM: takes care of long term dependencies as well as short term dependencies

LSTM: Long Short Term Memory

![][image279]

Simple RNN:           	

Ot \= f(Wxit \+ W’Ot-1) (sum)

yit hat \= g(W’’ Ot) 

![][image280]

LSTM: 

Ot \= f(\[W,W’’\] \[xit Ot-1\])  \- concatenation

yit hat \= g(W’’ Ot)

 ![][image281]

* 3 inputs: previous cell state, input, output of previous cell  
* 2 outputs: current output, current cell state  
* Using identity we can pass same cell states  
* Amount of previous cell state you want to pass can be manipulated

 

 

**Additional material**

a.       [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.7		GRU**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**Detailed explanation**

 ![][image282]

Source: [http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

- GRU cell is a simplified version of the LSTM cell and it performs as well as LSTM  
- Cell states from previous cell are merged into a single vector  
- Gate controller controls the forget gate and the input gate  
- There is no separate output gate, the full cell state is given as output  
- Usage: keras.layers.GRU()  
- Still it is difficult to retain very-long term patterns with these RNNs

**Video Lecture**

- GRU: Gated Recurrent Unit  
- LSTMS were created in 1997, GRUs \- 2014  
- LSTM have 3 gates: input, output, forget  
- GRUs: simplified version inspired from LSTMs, faster to train, as powerful as LSTM  
- GRUs: 2 gates: reset, update  
- The short-circuit structure is necessary for having long term dependencies

**Summary of comments**

Applications of long term dependencies: [https://en.wikipedia.org/wiki/Long\_short-term\_memory\#Applications](https://en.wikipedia.org/wiki/Long_short-term_memory#Applications)

- Predicting sales, finding stock market trends  
- Understanding movie plots  
- Speech recognition  
- Music composition

**Additional material**

1. [https://www.slideshare.net/hytae/recent-progress-in-rnn-and-nlp-63762080](https://www.slideshare.net/hytae/recent-progress-in-rnn-and-nlp-63762080)  
2. [https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)   
3. [https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm](https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm) 

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.8		DEEP RNN.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**Video Lecture**

- In MLPs, built one layer and then extended it to Deep MLPs  
- In CNN, built one layer and then extended it to Deep CNN  
- Similarly, with GRUs/LSTMs we will extend it to Deep RNNs

  ![][image283]

- Stacking one layer one over the other to build multiple layers  
- Deep RNN structure as other NNs are problem specific  
- Backpropagation and Forward propagation can be determined by the direction of arrows  
- Back propagation occurs across time and across depth also

**Summary of comments**

- The number of units in each layer of RNN is a hyperparameter that we need to tune. They're not dependent on the number of words in a sentence.    
- Number of words in sentence \== Number of time a cell will be unfolded along time axis \== Maximum length of sentence among all sentences \!= number of units  
- Seems like you are getting confused between the embedding and padding. Please note that padding is different from embedding. We make padding to sentences whereas here embedding is done to words. Suppose we have a sentence with 8 words and we pad them to 10 by adding two zeros in the end, we now have a 1\*10 vector. Now using embedding layer, if we want to represent each word using some 5 dimensional matrix, then we will have 1\*10\*5 . At each time step we send in 1\*5 vector and we send 10 such vectors to the LSTM unit. We can send sentences without padding as well. But we can send only one sentence at a time if we don't use padding. But we cannot send two words of different dimensions into LSTM as the length of cell state, hidden state are fixed.  
- Forget gate parameters are learnt by backpropagation.  
- Memoization is taken care by Tensorflow backend  
- During Backprop, weights are updated after all time steps. 

**Additional material**

1. [https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/)

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.9		BIDIRECTIONAL RNN.**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**Detailed explanation**

In a regular RNN, at each time step the layer just looks at past and present inputs. In Machine Translation tasks it is important to look ahead in the words that come next. This is implemented by using two recurrent layers which start from the two extremities of the sentence, and combine their outputs at each time step.

Usage: keras.layers.Bidirectional(keras.layers.GRU())

**Video Lecture**

Bi- directional RNNs:

NLP: input: x, output: y

In simple RNNs, yi3  is assumed to be dependent on x11, x12, x13

	But if yi3 is dependent on x14, x15 dependent which comes after yi3’s time step.

	To cater this we have bi-directional RNNs.

		![][image284]

	

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**54.10		CODE EXAMPLE: IMDB SENTIMENT CLASSIFICATION**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**Video Lecture**

* Dataset: IMDB reviews  
  * LSTMs are tricky to get them to work.  
  * Dataset: xi, yi pairs, xi \- sequence of words:  
    * xi is rank encoded list of the sentence, when you use top\_words \= k, the dataset will be loaded with k most frequent words  
    * Each xi is represented with a list of indices of the top\_words.  
    * All xi are padded to length 600 (a common length for sequences). Pre-padded to have sequence at the last of time steps. If post-padded, the sequence will be far from the output time step and leads to the requirement of a strong Long-term memory layer.  
    * For sequences with length \>this padded length, the sequence is truncated  
    * Without padding: SGD operation will be with batch\_size \= 1, which will be very slow. Sequences with different lengths cannot be formed into a batch as some xi require more time steps. To make batch type training possible, we use padding, else the training will be slow.  
    * Embedding: Turns positive integers {indexes} into dense vectors of fixed size, can only be used as first layer in a model  
    * Embed layer parameters: 5000\*32 \= 160k  
    * LSTM: Number of parameters: (4\*100(100+32+1))  
    * Dense layer: 1 output: 1\*(100+1) parameters  
    * Each LSTM cell can learn different aspects of the sequence leading to having different weight matrices and each cell is unrolled over time.

**Additional material**

\- Derivation of parameters in an LSTM layer ([https://www.youtube.com/watch?v=l5TAISVlYM0\&feature=youtu.be](https://www.youtube.com/watch?v=l5TAISVlYM0&feature=youtu.be))

* LSTM layer with m units:  
  * There are four gates:  
    * “f” First gate: Bias, input n-dimensional vector, cell state m-dimensional, Weights vector m+n dimensional: Number of parameters: (n+m+1)   
    * “i” Second gate: similar number of params  
    * “c” Third gate: similar  
    * “o” Fourth gate: similar  
    * Parameters: 4(n+m+1) per cell  
    * Total: 4m(n+m+1) per LSTM layer with m units, n \= input size  
    *  \= 4(nm+m\*\*2 \+ m)

**55**

**DEEP LEARNING: GENERATIVE ADVERSARIAL NETWORKS (GANs)**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

55.1		LIVE SESSION ON GENERATIVE ADVERSARIAL NETWORKS

**56**

**LIVE: ENCODER-DECODER MODELS**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

56.1 		LIVE: ENCODER-DECODER MODELS

**57**

**ATTENTION MODELS IN DEEP LEARNING**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

57.1		ATTENTION MODELS IN DEEP LEARNING

	\--- Live Sessions notebook

**58**

**INTERVIEW QUESTIONS ON DEEP LEARNING**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

58.1		QUESTIONS AND ANSWERS

Basics of NLP:

1. Explain about Bag of Words?  
2. Explain about Text Preprocessing: Stemming, Stop-word removal, Tokenization, Lemmatization.  
3. Explain about uni-gram, bi-gram, n-grams.?  
4. What is tf-idf (term frequency- inverse document frequency)  
5. Why use log in IDF?  
6. Explain about Word2Vec.?  
7. Explain about Avg-Word2Vec, tf-idf weighted Word2Vec?  
8. Explain about Multi-Layered Perceptron (MLP)?  
9. How to train a single-neuron model?  
10. How to Train an MLP using Chain rule ?  
11. How to Train an MLP using Memoization?  
12. Explain about Backpropagation algorithm?  
13. Describe about Vanishing and Exploding Gradient problem?  
14. Explain about Bias-Variance tradeoff in neural Networks?

  **Deep Learning:**

1. What is sampled softmax?  
2. Why is it difficult to train a RNN with SGD?  
3. How do you tackle the problem of exploding gradients? (By gradient clipping)  
4. What is the problem of vanishing gradients? (RNN doesn't tend to remember much things from the past)  
5. How do you tackle the problem of vanishing gradients? (By using LSTM)  
6. Explain the memory cell of a LSTM. (LSTM allows forgetting of data and using long memory when appropriate.)  
7. What type of regularization do one use in LSTM?  
8. What is the problem with sigmoid during back propagation? (Very small, between 0.25 and zero.)  
9. What is transfer learning?  
10. What is back propagation through time? (BPTT)  
11. What is the difference between LSTM and GRU?  
12. Explain Gradient Clipping.  
13. Adam and RMSProp adjust the size of gradients based on previously seen gradients. Do they inherently perform gradient clipping? If no, why?

Additional Source: [https://www.analyticsvidhya.com/blog/2017/01/must-know-questions-deep-learning/](https://www.analyticsvidhya.com/blog/2017/01/must-know-questions-deep-learning/)
