# CMSC-25400-Homework-3-solution

Download Here: [CMSC 25400 Homework 3 solution](https://jarviscodinghub.com/assignment/cmsc-25400-homework-3-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

1. In class we have derived an explicit formula for least squares regression when the hypothesis class is
the class of linear functions h(x) = θ0 + θ1×1 + θ2×2 + . . . θdxd. However, not all data can be well
modeled by just a linear equation. An alternative, generally richer hypothesis class can be defined by
defining a set of n basis functions {ϕ1(x), . . . , ϕn(x)}, and considering regressors of the form
h(x) = ∑n
i=1
θi ϕi(x).
Now each θi parameter is the coefficient of the corresponding ϕi
in the linear combination of basis
functions. The {ϕi} can really be any collection of functions, but as an illustration in the d = 1 case
you might consider something like a family of Gaussian bumps
ϕi = e
(x−i)
2/(2σ
2
)
i = 1, 2, . . . , n.
In principle one could also consider a infinite sequence of basis functions, but for simplicity here we
only consider a finite set. As before, let the loss function be the squared error loss
J(θ) = 1
2
∑m
j=1
(h(xj ) − yj )
2
.
Show that similarly to the linear case, the optimal solution can be found in the form
θ = (A
⊤A)
−1A
⊤⃗y,
where ⃗y = (y1, . . . , ym)
⊤, and derive the form of the matrix A. This problem illustrates that the
least squares technique (including its SGD version) has much broader applicability than just linear
regression.
2. In class we have derived that given data {(x1, y1),(x2, y2), . . . ,(xm, ym)}, the log-likelihood for logistic
regression is
ℓ(θ) = ∑m
i=1
[
ui
log(h(xi)) + (1 − ui) log(1 − h(xi))]
, (1)
where h(x) is the logistic function
h(x) = 1
1 + e−θ·x
= g(θ · x) g(z) = 1
1 + e−z
,
and the ui
’s are just the 0/1 analogs of the yi
’s, i.e., ui = (1 + yi)/2. There is no closed form solution
for the MLE of logistic regression.
(a) For simplicity consider (1) for a single data point (x, u). Derive the form of the gradient ∇ℓ(θ).
The formula g
′
(z) = g(z) (1 − g(z)) that we found in class might come in handy.
(b) Conclude that the SGD step based on a single datapoint (xi
, ui) in the dataset is
θ ← θ − α
[
(h(xi) − ui) xi
]
.
3. An online algorithm is said to be conservative if it changes its hypothesis only when it makes a mistake.
Let C be a concept class and A be a (not necessarily conservative) online algorithm which has a finite
mistake bound M on C. Prove that there is a conservative algorithm A′
for C which also has mistake
bound M.
1
4. Recall that the k–class perceptron maintains k separate weight vectors w1, w2, . . . , wk, and predicts
yb = arg max
i∈{1,2,…,k}
(wi
· x).
If this prediction is incorrect, and the correct label should have been y, it updates the weights by
setting
wy ← wy + x/2
wyb ← wyb − x/2.
Let {(x1, y1),(x2, y2), . . .} be the training data. Assume that ∥ xt ∥ = 1 for all t, and that this dataset
is separable with a margin δ, which in this case means that there exist unit vectors v1, v2, . . . , vk such
that for each example (xt, yt)
vyt
· xt − vy · xt ≥ 2δ y ∈ {1, 2, . . . , k} \ {yt} .
(a) Show that in the k = 2 case this notion of margin is equivalent to the margin that we saw in class.
(b) In the k = 2 case we saw that the number of mistakes that the perceptron can make is upper
bounded by 1/δ2
. Derive a similar bound for the k = 3 case. Hint: Two quantities that you may
wish to consider are a = v1 · w1 + v2 · w2 + v3 · w3 and b = ∥ w1 ∥
2 + ∥ w2 ∥
2 + ∥ w3 ∥
2
. Part of
your derivation might involve showing that a ≤ 3
√
b.
5. The file train35.digits contains 2000 images of 3’s and 5’s from the famous MNIST database of
handwritten digits in text format. The size of each image is 28 × 28 pixels. Each row of the file is a
representation one image, with the 28 × 28 pixels flattened into a vector of size 784. A value of 1 for a
pixel represents black, and value of 0 represents white. The corresponding row of train35.labels is
the class label: +1 for the digit 3, or −1 for the digit 5. The file test35.digits contains 200 testing
images in the same format as train35.digits.
Implement the perceptron algorithm and use it to label each test image in test35.digits. Submit the
predicted labels in a file named test35.predictions. In the lectures, the perceptron was presented
as an online algorithm. To use the perceptron as a batch algorithm, train it by simply feeding it
the training set M times. The value of M can be expected to be less than 10, and should be set by
cross validation. Naturally, in this context, the “mistakes” made during training are not really errors.
Nonetheless, it is intructive to see how the frequency of mistakes decreases as the hypothesis improves.
Include in your write-up a plot of the cumulative number of “mistakes” as a function of the number of
examples seen.
Since the data is fairly large, for debugging purposes it might be helpful to run your code on just
subsets of the 2000 training test images. Also, it may be helpful to normalize each example to unit
norm.

