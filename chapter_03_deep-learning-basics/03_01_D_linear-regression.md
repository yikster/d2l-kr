# Linear Regression

To get our feet wet, we'll start off by looking at the problem of regression.
This is the task of predicting a *real valued target* $y$ given a data point $x$.
Regression problems are extremely common in practice. For example, they are used for predicting continuous values, such as house prices, temperatures, sales, and so on. This is quite different from classification problems (which we study later), where the outputs are discrete (such as apple, banana, orange, etc. in image classification).

시작하는 단계로, 회귀 문제(regression)를 살펴보겠습니다. 회귀 문제는 주어진 데이터포인트 $x$에 해당하는 실제 값 형태의 타겟 $y$를 예측하는 과제입니다. 회귀 문제는 실제로 매우 일반적입니다. 예를 들면, 주택 가격, 온도, 판매량 등과 같은 연속된 값을 예측하는데 사용됩니다. 이는 결과 값이 이미지 분류와 같은 과일의 종류를 예측하는 이산적인(descrete)인 구분 문제(classification)와는 다릅니다. 

## Basic Elements of Linear Regression

In linear regression, the simplest and still perhaps the most useful approach,
we assume that prediction can be expressed as a *linear* combination of the input features
(thus giving the name *linear* regression).

가장 간단하지만 가장 유용한 접근 방법인 선형 회귀에서, 예측 함수가 입력 피처들의 선형 조합으로 표한된다고 가정합니다. 이 때문에, 선형 회귀(linear regression)이라고 불립니다.

### Linear Model

For the sake of simplicity we we will use the problem of estimating the price of a house based (e.g. in dollars) on area (e.g. in square feet) and age (e.g. in years) as our running example. In this case we could model

간단한 예를 들기 위해서, 집의 면적(제곱 미터)과 지어진지 몇년이 지났는지를 입력으로 받는 주택 가격을 예측하는 문제를 들어보겠습니다. 이 경우 모델은 다음과 같은 수식으로 표현할 수 있습니다.

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b$$

While this is quite illustrative, it becomes extremely tedious when dealing with more than two variables (even just naming them becomes a pain). This is what mathematicians have invented vectors for. In the case of $d$ variables we get

이 공식은 명확해 보이는데, 두개 이상의 입력 변수가 사용되는 경우는 굉장히 긴 공식이 됩니다. (변수 이름을 지정하는 것조차 지루한 일입니다.) 하지만, 수학자들이 발명한 백터를 사용하면 간단하게 표한이 가능합니다. 예를 들어, $d$ 개의 변수가 있다고 하면, 모델을 아래와 같이 표현됩니다.

$$\hat{y} = w_1 \cdot x_1 + ... + w_d \cdot x_d + b$$

Given a collection of data points $X$, and corresponding target values $\mathbf{y}$,
we'll try to find the *weight* vector $w$ and bias term $b$
(also called an *offset* or *intercept*)
that approximately associate data points $x_i$ with their corresponding labels $y_i$.
Using slightly more advanced math notation, we can express the long sum as $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$. Finally, for a collection of data points $\mathbf{X}$ the predictions $\hat{\mathbf{y}}$ can be expressed via the matrix-vector product:

데이터 포인트들을 모아서 $X$ 로 표현하고, 타겟 변수는  $y$ 로 표현하고, 각 데이터 포인트 $x_i$ 와 이에 대한 label 값인 $y_i$ 를 대략적으로 연관시켜주는 $가중치(weight)$ 백터 $w$ 와 $bias$ $b$ 를 찾아보는 것을 해볼 수 있습니다. 이를 조금 전문적인 수학 기호로 표현하면, 위 긴 공식은 $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ 로 표현됩니다. 즉, 데이터 포인드들의 집합 $X$ 와 예측 값 $\hat{\mathbf{y}}$ 은 아래와 같은 행렬-백터 곱의 공식이 됩니다.

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b$$

It's quite reasonable to assume that the relationship between $x$ and $y$ is only approximately linear. There might be some error in measuring things. Likewise, while the price of a house typically decreases, this is probably less the case with very old historical mansions which are likely to be prized specifically for their age. To find the parameters $w$ we need two more things: some way to measure the quality of the current model and secondly, some way to manipulate the model to improve its quality.

$x$ 와 $y$ 의 관계가 대략 선형적이라고 가정하는 것은 상당히 합리적입니다. 측정하는데 다소 오류가 발생할 수 있습니다. 마찬가지로, 주택 가격은 일반적으로 하락하지만, 오래될 수록 가치가 더 해지는, 오래된 역사적인 주택의 경우는 해당되지 않을 수 있습니다. 파라메터 $w$ 를 찾기 위해서는 두 가지를 더 필요합니다. 하나는, 현재 모델의 품질(quality)를 측정하는 방법과 두번째는 품질을 향상시킬 수 있는 방법입니다.

### Training Data

The first thing that we need is data, such as the actual selling price of multiple houses as well as their corresponding area and age. We hope to find model parameters on this data to minimize the error between the predicted price and the real price of the model. In the terminology of machine learning, the data set is called a ‘training data’ or ‘training set’, a house (often a house and its price) is called a ‘sample’, and its actual selling price is called a ‘label’. The two factors used to predict the label are called ‘features’ or 'covariates'. Features are used to describe the characteristics of the sample.

Typically we denote by $n$ the number of samples that we collect. Each sample (indexed as $i$) is described by $x^{(i)} = [x_1^{(i)}, x_2^{(i)}]$, and the label is $y^{(i)}$.

우선 필요한 것은 데이터입니다. 예를 들면, 여러 집들의 실제 판매가과 그 집들의 크기와 지어진지 몇년이 되었는지가 데이타가 됩니다. 우리가 하고자 하는 것은 모델이 예측한 집 가격과 실제 가격의 차이(오류)를 최소화하는 모델 파라메터를 찾는 것입니다. 머신러닝의 용어로는, 데이터셋은 '학습 데이터' 또는 '학습 셋'이라고 하고, 하나의 집 (집과 판매가)를 '샘플', 그 중에 판매가는 '레이블'이라고 합니다. 레이블을 예측하기 위해서 사용된 두 값는 '피처(feature)' 또는 '공변량(covariate)'라고 부르기도 합니다. 피처는 샘플의 특징을 표현하는데 사용됩니다.

일반적으로 수집한 샘플의 개수를 $n$ 으로 표기하고, 각 샘플은 인덱스 $i$ 를 사용해서 $x^{(i)} = [x_1^{(i)}, x_2^{(i)}]$ 와 레이블은  $y^{(i)}$ 로 적습니다.

### Loss Function

In model training, we need to measure the error between the predicted value and the real value of the price. Usually, we will choose a non-negative number as the error. The smaller the value, the smaller the error. A common choice is the square function. The expression for evaluating the error of a sample with an index of $i$ is as follows:

모델 학습을 위해서는 모델이 예측한 값과 실제 값의 오차를 측정해야합니다. 보통 오차는 0 또는 양수값을 선택하고, 값이 작을 수록, 오차가 적음을 의미합니다. 제곱 함수가 일반적으로 사용되고, index $i$ 의 샘플에 대한 오차 계산은 다음과 같이 합니다.

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2,$$

The constant $1/2$ ensures that the constant coefficient, after deriving the quadratic term, is 1, which is slightly simpler in form. Obviously, the smaller the error, the closer the predicted price is to the actual price, and when the two are equal, the error will be zero. Given the training data set, this error is only related to the model parameters, so we record it as a function with the model parameters as parameters. In machine learning, we call the function that measures the error the ‘loss function’. The squared error function used here is also referred to as ‘square loss’.

수식에 곱해진 1/2 상수값는 2차원 항목을 미분했을 때 값이 1이되게 만들기 위한 값입니다.

To make things a bit more concrete, consider the example below where we plot such a regression problem for a one-dimensional case, e.g. for a model where house prices depend only on area.

조금 더 구체화해보면, 집값이 집 크기에만 의존한다는 모델을 가정해서 일차원 문제로 회귀 문제를 도식화한 것을 예를 들어보겠습니다. 

![Linear regression is a single-layer neural network. ](../img/linearregression.svg)

As you can see, large differences between estimates $\hat{y}^{(i)}$ and observations $y^{(i)}$ lead to even larger contributions in terms of the loss, due to the quadratic dependence. To measure the quality of a model on the entire dataset, we can simply average the losses on the training set.

보는바와 같이 이차 의존성 (quadratic dependence)로 인해서 예측값  $\hat{y}^{(i)}$ 과 실제값 $y^{(i)}$ 의 큰 차이는 loss 로는 더 크게 반영됩니다. 전체 데이터셋에 대해서 모델의 품질을 측정하기 위해서는 학습셋에 대한 loss의 평균값을 사용할 수 있습니다.

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

In model training, we want to find a set of model parameters, represented by $\mathbf{w}^*, b^*$, that can minimize the average loss of training samples:

학습 샘플들의 평균 loss를 최소화하는 모델 파라메터  $\mathbf{w}^*$ 와 $b^*$ 를 찾는 것이 모델을 학습시키는 것입니다.

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$


### Optimization Algorithm

When the model and loss function are in a relatively simple format, the solution to the aforementioned loss minimization problem can be expressed analytically in a closed form solution, involving matrix inversion. This is very elegant, it allows for a lot of nice mathematical analysis, *but* it is also very restrictive insofar as this approach only works for a small number of cases (e.g. multilayer perceptrons and nonlinear layers are no go). Most deep learning models do not possess such analytical solutions. The value of the loss function can only be reduced by a finite update of model parameters via an incremental optimization algorithm.

모델과 loss 함수가 상대적으로 간단하게 표현되는 경우에는 앞에서 정의한 loss를 최소화하는 방법은 역행렬을 포함한 명확한 수식으로 표현할 수 있습니다. 이 수식은 다양하고 좋은 수학적 분석을 적용할 수 있어서 좋지만, 캐이스가 적은 경우에만 적용할 수 있는 제약이 있습니다. 대부분 딥러닝 모델은 위 분석 방법을 적용할 수 없습니다. loss 함수의 값은 점진적인 최적화 알고리즘을 사용해서 모델 파라메터를 유한한 회수로 업데이트하면서 줄이는 방법을 적용해야만 합니다.

The mini-batch stochastic gradient descent is widely used for deep learning to find numerical solutions. Its algorithm is simple: first, we initialize the values of the model parameters, typically at random; then we iterate over the data multiple times, so that each iteration may reduce the value of the loss function. In each iteration, we first randomly and uniformly sample a mini-batch $\mathcal{B}$ consisting of a fixed number of training data examples; we then compute the derivative (gradient) of the average loss on the mini batch the with regard to the model parameters. Finally, the product of this result and a predetermined step size $\eta > 0$ is used to change the parameters in the direction of the minimum of the loss. In math we have

딥러닝에서는 산술적인 솔루션으로 미니 배치를 적용한 stocahstic gradient descent 방법이 널리 사용되고 있습니다. 사용되는 알고리즘은 간단합니다: 우선, 일번적으로는 난수를 이용해서 모델 파라메터를 초기화합니다. 그 후, 데이터를 반복해서 사용해서 loss 함수의 값을 줄입니다. 각 반복에서는 학습 데이터에서 미리 정한 개수만큼의 샘플을 임의로 또 균일하게 뽑아서 미니 배치 $\mathcal{B}$ 를 구성하고, 미니 배치의 값들에 대한 평균 loss 값의 모델 파라메터에 대한 미분을 구합니다. 마지막으로 이 결과와 미리정의된 스텝 크기  $\eta > 0$ 를 곱해서 loss 값이 최소화되는 방향으로 파라메터를 변경합니다. 수식으로 표현하면 다음과 같습니다.

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)$$

For quadratic losses and linear functions we can write this out explicitly as follows. Note that $\mathbf{w}$ and $\mathbf{x}$ are vectors. Here the more elegant vector notation makes the math much more readable than expressing things in terms of coefficients, say $w_1, w_2, \ldots w_d$.

이차원 loss 및 선형 함수에 대해서는 아래와 같이 명시적으로 이를 계산할 수 있습니다. 여기서  $\mathbf{w}$ 와 $\mathbf{x}$ 는 백터입니다. 벡터를 잘 사용하면 $w_1, w_2, \ldots w_d$ 와 같은 계수를 읽기 쉬운 수식으로 표현됩니다.
$$
\begin{aligned}
\mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && =
w - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\
b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  && =
b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} - y^{(i)}\right).
\end{aligned}
$$

In the above equation $|\mathcal{B}|$ represents the number of samples (batch size) in each mini-batch, $\eta$ is referred to as ‘learning rate’ and takes a positive number. It should be emphasized that the values of the batch size and learning rate are set somewhat manually and are typically not learned through model training. Therefore, they are referred to as *hyper-parameters*. What we usually call *tuning hyper-parameters* refers to the adjustment of these terms. In the worst case this is performed through repeated trial and error until the appropriate hyper-parameters are found. A better approach is to learn these as parts of model training. This is an advanced topic and we do not cover them here for the sake of simplicity.

위 수식에서  $|\mathcal{B}|$ 는 각 미니 배치의 샘플 개수를 의미하고, $\eta$ 는 '학습 속도(learning rate)'를 뜻 합니다. 학습 속도는 양수 값을 사용합니다. 여기서 중요한 점은 미니 배치 크기와 학습 속도는 모델 학습을 통해서 찾아지는 값이 아니라 여러분이 직접 선택해야하는 값들 입니다. 따라서, 우리는 이 값들을 *hyper-parameters* 라고 불립니다. 우리가 흔히 *hyper-parameters* 튜닝이라고 하는 일은 이 값들을 조정하는 것을 의미합니다. 아주 나쁜 경우에는 적당한 *hyper-parameters* 를 찾기까지 반복된 실험을 수행해야 할 수 있습니다. 더 좋은 방법으로는 모델 학습의 일부로 이 값들을 찾는 것이 있습니다만, 심화 주제이기 떄문에 여기서는 다루지 않겠습니다.

### Model Prediction

After model training has been completed, we then record the values of the model parameters $\mathbf{w}, b$ as $\hat{\mathbf{w}}, \hat{b}$. Note that we do not necessarily obtain the optimal solution of the loss function minimizer, $\mathbf{w}^*, b^*$ (or the true parameters), but instead we gain an approximation of the optimal solution. We can then use the learned linear regression model $\hat{\mathbf{w}}^\top x + \hat{b}$ to estimate the price of any house outside the training data set with area (square feet) as $x_1$ and house age (year) as $x_2$. Here, estimation also referred to as ‘model prediction’ or ‘model inference’.

모델 학습이 끝나면 모델 파라메터  $\mathbf{w}, b$ 에 해당하는 값 $\hat{\mathbf{w}}와 \hat{b}$ 을 저장합니다. 학습을 통해서 loss 함수를 최소화 시키는 최적의 값 $\mathbf{w}^*, b^*$ 를 구할 필요는 없습니다. 다만, 이 최적의 값에 근접하는 값을 학습을 통해서 찾는 것입니다. 이 후, 학습된 선형 회귀 모델  $\hat{\mathbf{w}}^\top x + \hat{b}$ 을 이용해서 학습 데이터셋에 없는 집 정보에 대한 집 값을 추정할 수 있습니다. "추정"을 "모델 예측(prediction)" 또는 "모델 추론 (inference)" 라고 합니다.

Note that calling this step 'inference' is actually quite a misnomer, albeit one that has become the default in deep learning. In statistics 'inference' means estimating parameters and outcomes based on other data. This misuse of terminology in deep learning can be a source of confusion when talking to statisticians. We adopt the incorrect, but by now common, terminology of using 'inference' when a (trained) model is applied to new data (and express our sincere apologies to centuries of statisticians).

"추론(inference)"라는 용어는 실제로는 잘못 선택된 용어지만, 딥러닝에서는 많이 사용하는 용어로 자리잡았습니다. 통계에서 추론은 다른 데이터를 기반으로 파라메터들과 결과를 추정하는 것을 의미하기 때문에, 통계학자들과 이야기할 때 이 용어로 인해서 혼동을 가져오기도 합니다. 하지만, 이미 보편적으로 사용되고 있기 때문에, 학습된 모델에 새로운 데이터를 적용하는 것을 추론이라는 용어를 사용하겠습니다. (수백년을 걸친 통계학자들에게 미안함을 표합니다.)


## From Linear Regression to Deep Networks

So far we only talked about linear functions. Neural Networks cover a lot more than that. That said, linear functions are an important building block. Let's start by rewriting things in a 'layer' notation.

지금까지 선형 함수만을 이야기했는데, 뉴럴 네트워크는 이 보다 많은 것을 다룹니다. 물론 선형 함수는 중요한 구성 요소입니다. 이제 모든 것을 '층(layer)' 표기법으로 다시 기술해 보겠습니다.

### Neural Network Diagram

While in deep learning, we can represent model structures visually using neural network diagrams. To more clearly demonstrate the linear regression as the structure of neural network, Figure 3.1 uses a neural network diagram to represent the linear regression model presented in this section. The neural network diagram hides the weight and bias of the model parameter.

딥러닝에서는 모델의 구조를 뉴럴 네트워크 다이어그램으로 시각화할 수 있습니다. 뉴럴 네트워크 구조로 선형 회귀를 좀 더 명확하게 표현해보면, 그림 3.1에서와 같이 뉴럴 네트워크 다이어그램을 이용해서 이 절에서 사용하고 있는 선형 회귀 모델을 도식화 할 수 있습니다. 이 뉴럴 네트워크 다이어그램에서는 모델 파라메터인 가중치와 bias를 직접 표현하지 않습니다.

![Linear regression is a single-layer neural network. ](../img/singleneuron.svg)

In the neural network shown above, the inputs are $x_1, x_2, \ldots x_d$. Sometimes the number of inputs is also referred as feature dimension. In the above cases the number of inputs is $d$ and the number of outputs is $1$. It should be noted that we use the output directly as the output of linear regression.  Since the input layer does not involve any other nonlinearities or any further calculations, the number of layers is 1. Sometimes this setting is also referred to as a single neuron. Since all inputs are connected to all outputs (in this case it's just one), the layer is also referred to as a 'fully connected layer' or 'dense layer'.

위 뉴럴 네트워크에서 입력값은  $x_1, x_2, \ldots x_d$ 입니다. 때로는 입력값의 개수를 피처 차원(feature dimension)이라고 부르기도 합니다. 이 경우이는 입력값의 개수는 $d$ 이고, 출력의 개수는 1 입니다. 선형 회귀의 결과를 직접 결과로 사용하는 것을 기억해두세요. 입력 레이어는 어떤 비선형이나 어떤 계산을 적용하지 않기 때문에 레이어 개수는 1개입니다.  **Sometimes this setting is also referred to as a single neuron** 모든 입력이 모든 출력과 연결되어 있기 때문에, 이 레이어는 fully connected layer 또는 dense layer라고 불립니다.

### A Detour to Biology

Neural networks quite clearly derive their name from Neuroscience. To understand a bit better how many network architectures were invented, it is worth while considering the basic structure of a neuron. For the purpose of the analogy it is sufficient to consider the *dendrites* (input terminals), the *nucleus* (CPU), the *axon* (output wire), and the *axon terminals* (output terminals) which connect to other neurons via *synapses*.

뉴럴 네트워크라는 이름은 신경과학으로부터 나왔습니다. 얼마나 많은 네트워크 구조가 만들어졌는지 잘 이해하기 위해서, 우선 뉴론(neuron)의 기존적인 구조를 살펴볼 필요가 있습니다. 비유 하자면, 입력 단자인 수상돌기(dendrities), CPU인 핵(nucleu), 출력연결인 축삭(axon), 그리고, 시냅스를 통해서 다른 뉴런과 연결하는 축삭 단자(axon terminal)라고 하면 충분합니다.

![The real neuron](../img/Neuron.svg)

Information $x_i$ arriving from other neurons (or environmental sensors such as the retina) is received in the dendrites. In particular, that information is weighted by *synaptic weights* $w_i$ which determine how to respond to the inputs (e.g. activation or inhibition via $x_i w_i$). All this is aggregated in the nucleus $y = \sum_i x_i w_i + b$, and this information is then sent for further processing in the axon $y$, typically after some nonlinear processing via $\sigma(y)$. From there it either reaches its destination (e.g. a muscle) or is fed into another neuron via its dendrites.

수상돌기는 다른 뉴론들로 부터 온 정보 $x_i$ 를 받습니다. 구체적으로는 그 정보는 시텝틱 가중치 $w_i$ 가 적용된 정보값입니다. 이 가중치는 입력에 얼마나 반응을 해야하는지 정의합니다. (즉, $x_i w_i$ 를 통해서 활성화 됨) 이 모든 값들은 핵에서 $y = \sum_i x_i w_i + b$,  로 통합되고, 이 정보는 축삭(axon)으로 보내져서 다른 프로세스를 거치는데, 일반적으로는 $\sigma(y)$ 를 통해서 비선형 처리가 됩니다. 이 후, 최종 목적지 (예를 들면 근육) 또는 수상돌기를 거처서 다른 뉴론으로 보내집니다.

Brain *structures* can be quite varied. Some look rather arbitrary whereas others have a very regular structure. E.g. the visual system of many insects is quite regular. The analysis of such structures has often inspired neuroscientists to propose new architectures, and in some cases, this has been successful. Note, though, that it would be a fallacy to require a direct correspondence - just like airplanes are *inspired* by birds, they have many distinctions. Equal sources of inspiration were mathematics and computer science.

뇌의 구조는 아주 다양합니다. 어떤 것들은 다소 임의적으로 보이지만, 어떤 것들은 아주 규칙적인 구조를 가지고 있습니다. 예를 들면, 여러 곤충들의 시각 시스템은 아주 구조적입니다. 이 구조들에 대한 분석을 통해서 신경과학자들은 새로운 아키텍처를 제안하는데 영감을 받기도 하고, 어떤 경우에는 아주 성공적이어 왔습니다. **하지만, 비행기가 새로 부터 영감을 받아서 만든 것처럼 직접 관계를 찾아보는 것은 오류가 되기도 랍니다. 수학과 컴퓨터 과학이 영감의 같은 근원이라고 볼 수 있습니다**

### Vectorization for Speed

In model training or prediction, we often use vector calculations and process multiple observations at the same time. To illustrate why this matters, consideer two methods of adding vectors. We begin by creating two 1000 dimensional ones first.

모델 학습 및 예측을 수행할 때, 백터 연산을 사용하고 이를 통해서 여러 값들은 한번에 처리합니다. 이것이 왜 중요한지 설명하기위해서 백터들을 더하는 두 가지 방법을 생각해봅시다. 우선 1000 차원의 백터 두개를 생성합니다.

```{.python .input  n=1}
from mxnet import nd
from time import time

a = nd.ones(shape=10000)
b = nd.ones(shape=10000)
```

One way to add vectors is to add them one coordinate at a time using a for loop.

두 백터값을 더하는 방법 중에 하나는 for loop을 이용해서 백터의 각 값들을 하나씩 더하는 것입니다.

```{.python .input  n=2}
start = time()
c = nd.zeros(shape=10000)
for i in range(10000):
    c[i] = a[i] + b[i]
time() - start
```

Another way to add vectors is to add the vectors directly:

다른 방법으로는 두 백터들을 직접 더할 수 있습니다.

```{.python .input  n=3}
start = time()
d = a + b
time() - start
```

Obviously, the latter is vastly faster than the former. Vectorizing code is a good way of getting order of mangitude speedups. Likewise, as we saw above, it also greatly simplifies the mathematics and with it, it reduces the potential for errors in the notation.

당한하게도 백터를 직접 더하는 방법이 훨씬 더 빠릅니다. 코드를 백터화하는 것은 연산 속도를 빠르게 하는 좋은 방법입니다. 마찬가지로, 연산식을 간단하게 하고, 표기에 있어서 잠재적인 오류를 줄여주는 효과도 있습니다.

## The Normal Distribution and Squared Loss

The following is optional and can be skipped but it will greatly help with understanding some of the design choices in building deep learning models. As we saw above, using the squred loss $l(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2$ has many nice properties, such as having a particularly simple derivative $\partial_{\hat{y}} l(y, \hat{y}) = (\hat{y} - y)$. That is, the gradient is given by the difference between estimate and observation. You might reasonably point out that linear regression is a [classical](https://en.wikipedia.org/wiki/Regression_analysis#History) statistical model. Legendre first developed the method of least squares regression in 1805, which was shortly thereafter rediscovered by Gauss in 1809. To understand this a bit better, recall the normal distribution with mean $\mu$ and variance $\sigma^2$.

이번 내용은 옵션이니, 다음으로 넘어가도 됩니다. 하지만, 딥러닝 모델을 구성에 있어서 디자인 선택에 대한 이해를 높이는데 도움이 됩니다. 위에서 봤듯이, squred loss  $l(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2$ 는 간단한 편미분  $\partial_{\hat{y}} l(y, \hat{y}) = (\hat{y} - y)$ 과 같은 좋은 특징들을 가지고 있습니다. 즉, gradient가 예측값과 실제값의 차이로 계산됩니다. 선형 회귀는 전통적인 통계 모델입니다. Legendre가 처음으로 least squres regression을 1805년에 개발했고, 1809년에 Gauss에 의해서 재발견되었습니다. 이를 조금 더 잘 이해하기 위해서 평균이  $\mu$ 이고 편차가 $\sigma^2$ 인 정규 분포(normal distribution)를 떠올려 봅시다. 

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right)$$

It can be visualized as follows:

이는 다음과 같이 시각화될 수 있습니다.

```{.python .input  n=2}
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
from mxnet import nd
import math

x = nd.arange(-7, 7, 0.01)
# mean and variance pairs
parameters = [(0,1), (0,2), (3,1)]

# display SVG rather than JPG
display.set_matplotlib_formats('svg')
plt.figure(figsize=(10, 6))
for (mu, sigma) in parameters:
    p = (1/math.sqrt(2 * math.pi * sigma**2)) * nd.exp(-(0.5/sigma**2) * (x-mu)**2)
    plt.plot(x.asnumpy(), p.asnumpy(), label='mean ' + str(mu) + ', variance ' + str(sigma))

plt.legend()
plt.show()
```

As can be seen in the figure above, changing the mean shifts the function, increasing the variance makes it more spread-out with a lower peak. The key assumption in linear regression with least mean squares loss is that the observations actually arise from noisy observations, where noise is added to the data, e.g. as part of the observations process.

위 그림에서 보이 듯이, 평균을 변경하면 함수를 이동시고, 편차를 증가시키면 피크는 낮추고 더 펼쳐지게 만듭니다. least mean sequre loss를 적용한 선형 회귀에서 중요한 가정은 관찰들은 노이즈가 있는 관찰에서 얻어지고, 이 노이즈들은 데이터에 더해진다는 것입니다.

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2)$$

This allows us to write out the *likelihood* of seeing a particular $y$ for a given $\mathbf{x}$ via

이는 주어진 $x$ 에 대해서 특정 $y$ 를 얻을 가능성(likelihood)는 다음과 같이 표현됩니다.

$$p(y|\mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right)$$

A good way of finding the most likely values of $b$ and $\mathbf{w}$ is to maximize the *likelihood* of the entire dataset

가장 근접한 $b$ 와 $\mathbf{w}$ 값을 찾는 좋은 방법은 전체 데이터셋에 대한 likelihood를 최대화하는 것입니다.

$$p(Y|X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)})$$

The notion of maximizing the likelihood of the data subject to the parameters is well known as the *Maximum Likelihood Principle* and its estimators are usually called *Maximum Likelihood Estimators* (MLE). Unfortunately, maximizing the product of many exponential functions is pretty awkward, both in terms of implementation and in terms of writing it out on paper. Instead, a much better way is to minimize the *Negative Log-Likelihood* $-\log P(Y|X)$. In the above case this works out to be

파라매터들에 대해서 데이터의 likelihood를 최대화하는 것은 *Maximum Likelihood Principle* 로 잘 알려져 있고, 이런 estimator들은  *Maximum Likelihood Estimators* (MLE)라고 불립니다. 아쉽게도, 많은 지수 함수들의 곱을 최적화하는 것은 구현하는 것이나, 종이에 적어보는 것이나 아주 어렵습니다. 대신, 더 좋은 방법은 *Negative Log-Likelihood* $-\log P(Y|X)$ 를 최소화하는 것입니다. 위 예는 다음 수식으로 표현됩니다.

$$-\log P(Y|X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2$$

A closer inspection reveals that for the purpose of minimizing $-\log P(Y|X)$ we can skip the first term since it doesn't depend on $\mathbf{w}, b$ or even the data. The second term is identical to the objective we initially introduced, but for the multiplicative constant $\frac{1}{\sigma^2}$. Again, this can be skipped if we just want to get the most likely solution. It follows that maximum likelihood in a linear model with additive Gaussian noise is equivalent to linear regression with squared loss.

이 공식을 잘 살펴보면  $-\log P(Y|X)$ 를 최소화할 때는 첫번째 항목을 무시할 수 있습니다. 왜냐하면, 첫번째 항목은 $w$, $b$ 그리고 데이터와도 연관이 없기 때문입니다. 두번째 항목은 우리가 이전에 봤던 objective와 상수 $\frac{1}{\sigma^2}$ 가 곱해진 것을 빼면 동일합니다. 이 값은 가장 근접한 솔루션을 찾는 것만 원한다면 무시할 수 있고, 이렇게 하면 additive Gaussian noise를 갖는 선형 모델의 likelihood를 최대화하는 것은 squred loss를 적용한 선형 회귀와 동일한 문제로 정의됩니다.

## Summary

* Key ingredients in a machine learning model are training data, a loss function, an optimization algorithm, and quite obviously, the model itself.
* Vectorizing makes everything better (mostly math) and faster (mostly code).
* Minimizing an objective function and performing maximum likelihood can mean the same thing.
* Linear models are neural networks, too.
* 머신러닝에서 중요한 요소는 학습 데이터, loss 함수, 최적화 알고리즘, 그리고 당연하지만 모델 자체입니다.
* 백터화는 모든 것(수학)을 좋게 만들고, (코드를) 빠르게 만들어 줍니다.
* objective 함수를 최소화하는 것과 maximum likelihood 구하는 것은 같은 것입니다.
* 선형 모델도 뉴럴 모델이다.

## Exercises

1. Assume that we have some data $x_1, \ldots x_n \in \mathbb{R}$. Our goal is to find a constant $b$ such that $\sum_i (x_i - b)^2$ is minimized.
    * Find the optimal closed form solution.
    * What does this mean in terms of the Normal distribution?
1. Assume that we want to solve the optimization problem for linear regression with quadratic loss explicitly in closed form. To keep things simple, you can omit the bias $b$ from the problem.
    * Rewrite the problem in matrix and vector notation (hint - treat all the data as a single matrix).
    * Compute the gradient of the optimization problem with respect to $w$.
    * Find the closed form solution by solving a matrix equation.
    * When might this be better than using stochastic gradient descent (i.e. the incremental optimization approach that we discussed above)? When will this break (hint - what happens for high-dimensional $x$, what if many observations are very similar)?.
1. Assume that the noise model governing the additive noise $\epsilon$ is the exponential distribution. That is, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    * Write out the negative log-likelihood of the data under the model $-\log p(Y|X)$.
    * Can you find a closed form solution?
    * Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint - what happens near the stationary point as we keep on updating the parameters). Can you fix this?
1. Compare the runtime of the two methods of adding two vectors using other packages (such as NumPy) or other programming languages (such as MATLAB).

## Discuss on our Forum

<div id="discuss" topic_id="2331"></div>
