"""-**
This is my practice module I'm just building intuition for Regression Models for better understanding the solving problems. Initially I found difficulty to understand why we are using the Machine learning and what we are predicting. Now I have clear Understanding Its not Fancy In depth Project For that I will add the Capstone Project Later On.

For this intuition Building I'm clearly using the Basic California House Price Prediction dataset because I found It simple and easy to understand rathe moving forward to any fancy and difficult dataset.
**-"""
Problem Statement : Imagine you are a real estate agent trying to price a new house that just came on the market. You know its size (square footage). How do you guess the price?

Geometric Intuition: Lines, Planes, and Beyond:
1D (One Feature): If you are predicting price based only on size, your model is a line on a flat graph.

2D (Two Features): What if you use size and the number of bedrooms? Now your data lives in a 3D room. Your model is no longer a line, but a flat plane (like a sheet of stiff cardboard) angled through the cloud of floating dots.

High Dimensions: If you use 100 features (size, bedrooms, age, distance to schools, etc.), your model is a "hyperplane." Our brains can't picture a 100-dimensional space, but the geometry works exactly the same way. It's just a flat, straight surface slicing through data.

The vertical distance between the actual dot (reality) and your line (the prediction) is called the error or residual.
Our goal in linear regression is to position the line so that the total tension of all these springs is as small as possible. We want to minimize the error.

A line is defined by two things:

Intercept: Where the line starts on the vertical axis (e.g., the base value of the land even if the house size is zero).

Slope: How steep the line is (e.g., how much the price jumps for every single extra square foot).

In the real world, a zero slope means the feature you are measuring (size) has absolutely zero impact on the outcome (price). A 500 sq ft house would be predicted to cost the same as a 5,000 sq ft house!

In algebra, you probably learned the equation of a line as $y = mx + c$. In machine learning, we write this slightly differently, usually calling it our "hypothesis" function (our prediction):$$\hat{y} = wx + b$$

$\hat{y}$ (pronounced "y-hat"): Our predicted price.$x$: Our input feature (house size).$w$: The weight (the slope). It tells us how much influence $x$ has.$b$: The bias (the y-intercept). Our base baseline prediction if $x$ was zero.If we have multiple features (size, age, bedrooms), it simply expands into a hyperplane:$$\hat{y} = w_1x_1 + w_2x_2 + w_3x_3 + b$$



We need a mathematical way to calculate the total "tension of the springs" (the error) so we know if our line is good or bad. We use a function called Mean Squared Error (MSE). We often denote this cost function as 

$J$.$$J(w,b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$


Linear regression isn't magic; it is a mathematical tool that comes with a specific set of rules. If your data doesn't follow these rules, your model's predictions will be unreliable. We call these the L.I.N.E. assumptions.

1. The L.I.N.E. Assumptions
L - Linearity: The relationship between X and Y must actually be a straight line.

What it looks like: A cloud of points moving in a general straight direction.

When it breaks: If your data actually curves like a U-shape (e.g., age vs. energy level), drawing a straight line through it is useless. Your model will constantly under-predict or over-predict at the extremes.

I - Independence: The errors of your data points should not influence each other.

When it breaks: This usually happens in time-based data. If predicting stock prices, today's price is heavily dependent on yesterday's. Linear regression assumes each house (or data point) is completely independent of the others.

N - Normality of Residuals: The errors (residuals) should follow a normal distribution (a bell curve).

What it looks like: Most of your predictions should be very close to the actual price (small errors). You should have very few predictions that are wildly off (huge errors).

When it breaks: If your errors are heavily skewed, it means your model is systematically failing for a certain chunk of your data.

E - Equal Variance (Homoscedasticity): The spread of the errors should be constant across all values of X.

What it looks like: If you plot your errors, they should look like a random, evenly thick tube of static.

When it breaks (Heteroscedasticity): The errors form a cone shape. For example, predicting the price of a small 1-bedroom house is easy (low error). But for a 10,000 sq ft mansion, the price could be wildly different based on luxury features, meaning your predictions get vastly more inaccurate as the houses get larger.

2. The Bias-Variance Tradeoff
Since you already brought up overfitting, let's formally define the most important concept in machine learning: the tug-of-war between Bias and Variance.

Imagine looking at a target board where the bullseye is the true relationship of the data.

High Bias (Underfitting): The model is too simple. It completely ignores the underlying patterns in the data and just draws a dumb, rigid line. It has strong "bias" towards its own simple assumption.

Visual: A flat straight line trying to fit data that clearly curves.

Result: High error on the training data, and high error on new data.

High Variance (Overfitting): The model is too complex and incredibly sensitive. It chases every single noisy data point, zigzagging wildly to get an MSE of 0.

Visual: A line that looks like a rollercoaster, perfectly connecting every single dot.

Result: Zero error on the training data, but massive error on new data because the model memorized the noise instead of learning the trend.

The Tradeoff: We want to find the sweet spot in the middle—a model complex enough to capture the true trend (low bias), but simple enough that it doesn't freak out over random noise (low variance).


When a model overfits, it usually does so by assigning massive, extreme weights ($w$) to specific features to perfectly capture every tiny fluctuation in the training data.Regularization is like adding a "budget" or a "speed limit" to your weights. We tell the model: "You are allowed to minimize the Mean Squared Error (MSE), but you will be heavily fined if you use large weights to do it."

Total Cost = MSE + Penalty for Large Weights
The parameter that controls how harsh this penalty is is called Lambda ($\lambda$) or Alpha ($\alpha$) in Scikit-Learn.$\lambda = 0$: No penalty. Standard Linear Regression. (Prone to overfitting).$\lambda \rightarrow \infty$: Infinite penalty. All weights are crushed to zero. (Prone to underfitting).

L2 Regularization(Weight Shrinkage) (Ridge Regression)Instead of just MSE, we add the squared sum of the weights.$$J(w,b) = \text{MSE} + \lambda \sum_{j=1}^{p} w_j^2$$

L1 Regularization {Sparsity/Feature Selection} (Lasso Regression)Instead of squaring, we add the absolute value of the weights. (Lasso stands for Least Absolute Shrinkage and Selection Operator).$$J(w,b) = \text{MSE} + \lambda \sum_{j=1}^{p} |w_j|$$
