## **Mike Heddes**[1] **Adel Javanmard**[2 3] **Kyriakos Axiotis**[3] **Gang Fu**[3] **MohammadHossein Bateni**[3] **Vahab Mirrokni**[3]

## **Abstract**

Transformer networks have achieved remarkable success across diverse domains, leveraging a variety of architectural innovations, including residual connections. However, traditional residual connections, which simply sum the outputs of previous layers, can dilute crucial information. This work introduces DeepCrossAttention (DCA), an approach that enhances residual learning in transformers. DCA employs learnable, inputdependent weights to dynamically combine layer outputs, enabling the model to selectively focus on the most relevant information in any of the previous layers. Furthermore, DCA incorporates depth-wise cross-attention, allowing for richer interactions between layers at different depths. Our language modeling experiments show that DCA achieves improved perplexity for a given training time. Moreover, DCA obtains the same model quality up to 3x faster while adding a negligible number of parameters. Theoretical analysis confirms that DCA provides an improved trade-off between accuracy and model size when the ratio of collective layer ranks to the ambient dimension falls below a critical threshold.

## **1. Introduction**

Residual connections play an important role in modern neural network architectures because they stabilize the training of deep neural networks and improve model convergence and quality. Since their usage in the ResNet architecture (He et al., 2016), residual connections have been widely adopted in both convolutional neural networks and transformer architectures across various domains, including natural language

> 1Department of Computer Science, University of California, Irvine, USA[2] Marshall School of Business, University of Southern California, Los Angeles, USA[3] Google Research, New York, USA. Correspondence to: Mike Heddes _<_ mheddes@uci.edu _>_ , Gang Fu _<_ thomasfu@google.com _>_ .

_Proceedings of the 42[nd] International Conference on Machine Learning_ , Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

processing (Vaswani, 2017), audio recognition (Gong et al., 2021), and computer vision (Dosovitskiy et al., 2021).

A residual neural network (ResNet) is constructed by stacking layers known as residual blocks. Each residual block is characterized by the recursive equation _**x** t_ +1 = _f_ ( _**x** t_ ) + _**x** t_ , which contains a residual function _f_ along with an identity shortcut (also called an identity loop or skip connection). The residual functions typically used in these blocks include multi-layer perceptrons (MLPs), convolutional neural networks (CNNs), and attention. By unrolling the recursion, we equivalently see that each layer’s input is the sum of all its previous layers’ outputs (including the model’s input). Figure 2 provides a schematic illustration of this concept.

**Information dilution in residual networks.** Residual connections increase the flow of information across the neural network. However, they also come with a potential limitation: Taking a straight sum of previous layer outputs implicitly treats all previous layers as equally important. This can dilute useful information present in a select few layers (including the model’s input) with potentially less useful information. We hypothesize that, because of this dilution, even though residual networks mitigate the problem of neural network bottlenecks, they do not sufficiently resolve it. One way to resolve the issue of dilution would be to allow each layer to _choose its inputs_ .

In order to confirm the existence and significance of the dilution phenomenon we ask a simple question: _Can residual networks easily learn to recover the input?_ This should be a basic task expected of any generative model — otherwise there would be information loss. However, if our dilution hypothesis is true, the answer would be negative. To test this, we create a neural network consisting of a number of low-rank layers, and add residual connections in order to mitigate the bottlenecks introduced by the low ranks. The resulting model is full-rank. We compare this model with another model that employs _learnable_ residual connections, as in DenseFormer (Pagliardini et al., 2024), which we later also call _GRN-v1_ , since it is the starting point of our generalizations. In Figure 1 we see the results of the two models on two tasks: learning the identity transformation and learning a random linear transformation. Perhaps surprisingly, we observe that the residual network is unable

**==> picture [470 x 172] intentionally omitted <==**

10 [1]<br>10 [2] ResNet<br>10 2 GRN-v1<br>10 5<br>10 8<br>ResNet 6 × 10 [1]<br>10 11 GRN-v1<br>10 14<br>4 × 10 [1]<br>10 17<br>10 20 3 × 10 [1]<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>Number of steps 1e3 Number of steps 1e3<br>(a) Learning the identity transformation by minimizing (b) Minimizing the loss ∥f ( x ) − y ∥ [2] 2 [, where] [ x] [ is a][ 100][-d i.i.d.]<br>∥f ( x ) − x ∥ [2] 2 [, where] [ x] [ is a][ 100][-dimensional i.i.d.] [normal input] normal input, f is a low-rank linear network, and y = Ax + b ,<br>and f is a low-rank linear network. where A , b have i.i.d. standard normal entries.<br>MSE (log scale) MSE (log scale)<br>**----- End of picture text -----**<br>

_Figure 1._ Training low-rank linear models to learn the identity and a random transformation. Each model consists of 10 linear layers, each of rank 3, and is trained using mini-batch SGD.

to fully reconstruct the input even after seeing 10[3] batches (10[5] examples), while the model with learnable residual weights is able to reach extremely small loss values, even with 100x fewer examples. This confirms that ResNet does not address neural network bottlenecks in a satisfactory way, even though it learns a full-rank transformation, and underscores the importance of using learnable residual weights to increase model capacity.

**Our contribution.** In this work, we propose _DeepCrossAttention (DCA)_ , a new transformer architecture that generalizes residual networks by employing learnable, inputdependent weights to dynamically combine layer outputs, enabling the model to selectively focus on the most relevant information in any of the previous layers and thereby prevent dilution of information in the hidden representations. Furthermore, DCA incorporates depth-wise cross-attention by enabling the queries, keys, and values in each transformer block to independently combine layer outputs, allowing for richer interactions between layers at different depths. This is all achieved with a negligible number of additional parameters, making DCA more effective than increasing the model size (for instance by increasing its width or depth).

DCA can be viewed as a mechanism to adapt the model architecture dynamically for each input token. By optimizing the added parameters, DCA learns to effectively combine the outputs of earlier residual blocks. This allows the model to rearrange the residual blocks from purely sequential to fully parallel and any intermediate combination, without the need for explicit architectural design choices.

We analyse our generalization of the residual network theoretically by focusing on a linear low-rank model. We show that DCA achieves a better trade-off between accuracy and model size when the ratio of the collective ranks of the lay-

ers to the ambient dimension is below a threshold, which depends on the complexity of the target task. In addition, the improvement in this trade-off can itself be characterized as a function of the collective ranks of the layers, ambient dimension and the complexity of the target task. We extend this insight to nonlinear models by working with the notion of bottleneck rank, proposed by Jacot (2023).

We additionally provide empirical results to support the theoretical findings and demonstrate the effectiveness of DCA. Experiments on language modeling and image classification tasks demonstrate that DCA consistently outperforms the standard transformer architectures in terms of perplexity, accuracy and training efficiency. DCA achieves lower perplexity for a given parameter budget and training time. Furthermore, DCA exhibits improved training stability, mitigating the occurrence of loss spikes frequently observed while training large models.

## **2. Related Work**

Highway networks enable each layer to interpolate dynamically between its output _f_ ( _**x**_ ) and its input _**x**_ using a gating mechanism (Srivastava et al., 2015). Residual connections (He et al., 2016) popularized the direct flow of information from earlier to later layers using identity shortcuts. These innovations proved crucial in stabilizing training and allowing for the construction of significantly deeper networks. Building upon this concept, DenseNet (Huang et al., 2017) further enhanced information flow by concatenating the outputs of all preceding layers to each layer’s input.

The following methods are the ones most similar to ours. They all build on the idea of DenseNet but apply an efficient aggregation of the previous layer outputs instead of

concatenating them. DenseFormer (Pagliardini et al., 2024) performs the aggregation as a learned linear combination of the previous layer outputs. To reduce the computational load, they propose to apply their method only on a subset of the possible layer connections. Building on DenseFormer, LAuReL (Menghani et al., 2024) presents three aggregation functions, the best performing one applies a learned low-rank transformation to the previous layer outputs before the learned linear combination. Zhu et al. (2024) take a different approach with Hyper-Connections, they consider a fixed-size stack where layer outputs are added into with a learned weight for every slot of the stack. Before each layer, the stack is mixed by a matrix multiplication with a learned weight matrix. The input to a layer is then obtained by a learned linear combination of the stack, instead of accessing the previous layer outputs directly. They also present a dynamic version of their method where the weights are derived from the inputs.

## **3. Method**

We start with a detailed exposition of our proposed generalizations to the residual network architecture. We present three distinct proposals, each incrementally augmenting the complexity of the network structure. Building upon these proposals, we subsequently introduce DeepCrossAttention (DCA), a novel approach to enhance residual learning capabilities of the transformer architecture.

**Notation.** We denote a residual function by _ft_ : R _[d] →_ R _[d]_ , where _t_ is the layer index and _d_ the feature dimension. As an example, in a multi-layer perceptron residual network (MLPResNet), we have _ft_ ( _**x**_ ) = _**V** tσ_ ( _**W** t_ _**x**_ ) with _**W** t ∈_ R _[k][×][d]_ , _**V** t ∈_ R _[d][×][k]_ and _σ_ is a nonlinear function, such as sigmoid or ReLU, that is applied component-wise. Then, the _t_ -th residual block outputs _gt_ +1( _**x**_ ), are defined recursively as

**==> picture [125 x 11] intentionally omitted <==**

Using this recursion, the output of the _T_ -th residual block is given by

**==> picture [109 x 30] intentionally omitted <==**

with the conventions that _g_ 0( _**x**_ ) = **0** and _f_ 0( _g_ 0( _**x**_ )) = _**x**_ . We refer to Figure 2 for a schematic illustration.

An alternative description, which we will use to introduce our generalizations, is the following. For every _t_ , define the stack of layer outputs _**G** t ∈_ R _[d][×][t]_ as

**==> picture [197 x 14] intentionally omitted <==**

We then have _gt_ ( _**x**_ ) = _**G** t_ **1** and _**y**_ = _**G** T_ **1** in the standard residual network, where **1** denotes the all ones vector.

**==> picture [200 x 82] intentionally omitted <==**

_Figure 2._ Two alternative schematic representations of standard ResNet. The top represents the recursive form, the bottom represents the explicit sum.

**==> picture [165 x 156] intentionally omitted <==**

_Figure 3._ Computation diagram of GRN-v3.

## **3.1. Generalized Residual Networks (GRN)**

We propose three generalizations of ResNets by considering weighted linear combinations of previous layer outputs. The parameters of the modules and the generalizations are all optimized during training using the AdamW optimizer (Loshchilov & Hutter, 2017).

**Dimension-independent weights (GRN-v1)** . We consider simple linear combinations as

**==> picture [138 x 11] intentionally omitted <==**

with _**b** t ∈_ R _[t][×]_[1] which is initialized as all ones and optimized with the rest of the model parameters during training. This setting has been previously explored in the DenseFormer paper (Pagliardini et al., 2024).

**Dimension-dependent weights (GRN-v2).** In this proposal, we allow _**b** t ∈_ R _[d][×][t]_ and consider

**==> picture [195 x 11] intentionally omitted <==**

where _⊙_ indicates the entry-wise (Hadamard) product. Note that in GRN-v1 the same weight vector _**b** t_ is used for each of the _d_ features. GRN-v2 generalizes this by using different weight vectors for different features, which are all stacked together in a matrix _**b** t ∈_ R _[d][×][t]_ .

**Input-dependent weights (GRN-v3).** In the next generalization, we allow the weights to be input dependent. Specifi-

cally, the weights are given by _**b** t_ + _**w**_ ¯ _t_ with _**b** t,_ _**w**_ ¯ _t ∈_ R _[d][×][t]_ . The first component acts similar to the weights in GRNv2, it puts different weights on different dimensions of the input. The second component _**w**_ ¯ _t_ is a nonlinear mapping of the input features vector _**x**_ , but is the same for all the _d_ dimensions. This combination gives us flexibility to have both dimension-dependent and input-dependent weights for a slight increase in the number of parameters. GRN-v3 is expressed as

**==> picture [207 x 28] intentionally omitted <==**

where _**w** t_ : R _[d][×]_[1] is initialized as all zeros and optimized with the rest of the model parameters during training and _σ_ : R _→_ R is a non-linearity which is applied entry-wise. In this proposal we consider _σ_ to be the ReLU activation. The computation diagram of GRN-v3 is illustrated in Figure 3.

**Reducing memory and computation.** Since the stack of layer outputs _**G** t_ grows linearly with the depth of the model, this could lead to significant memory and computational overhead for deep models. Our experiments reveal that GRNs tend to weight inputs and the last few layer outputs the most. An example weight distribution is provided in Appendix H. Therefore, to increase efficiency, we propose to include only the first and last- _k_ layers explicitly in _**G** t_ . On the intermediate layers we apply standard ResNet, only involving simple addition. For example, if we set _k_ = 2, then _**G** t_ contains at most 4 vectors: the model inputs, the sum of the intermediate layers’ outputs, and the last two layers’ outputs _ft−_ 1( _gt−_ 1( _**x**_ )) and _ft−_ 2( _gt−_ 2( _**x**_ )). The GRNs then take this modified _**G** t_ as their input.

## **3.2. DeepCrossAttention**

The generalizations introduced thus far are generally applicable to any ResNet. We now describe our main method which is specific to the transformer architecture. DeepCrossAttention (DCA) generalizes self-attention by adding three independent instances of a GRN in each decoder block. In this proposal we consider the GRN to be GRN-v3. These three GRN instances are given the same stack of previous layer outputs as their input but return the queries, keys, and values for the attention module, respectively. This enables richer interactions between layers at different depths. Figure 4 shows the computation diagram of a DCA decoder block inside a transformer, where the remaining skip connections ensure that the inputs are not added to the outputs of the decoder block, but are included in the inputs of both the attention and the feed forward module. Notably, DCA does not modify the underlying attention mechanism, but instead uses GRNs to dynamically compose attention inputs.

**==> picture [153 x 217] intentionally omitted <==**

_Figure 4._ Computation diagram of a DCA decoder block.

## **4. Theoretical analysis**

Motivated by language modeling tasks, we focus on the regime where the size of the training set ( _n_ ) significantly exceeds the input dimension ( _n ≫ d_ ). As we increase the number of model parameters, the representation capacity of the network improves, which helps with reducing the test error. We will be focusing on the the trade-off between the test error and the number of parameters, and argue that our proposed generalizations achieve a better trade-off than the standard ResNet.

We will first study a “stylized” low-rank linear model for which we characterize the test error-model complexity tradeoff and demonstrate the benefits of our proposed generalizations. Our analysis elucidates the role of various factors on this trade-off, such as collective widths of layers, complexity of the target task, and input dimension. We then discuss how some of these results can be extended to non-linear models and empirically demonstrate that the insights gained from our analysis are applicable to more complex models.

Due to space constraint, proof of theorems are deferred to the supplementary material.

## **4.1. Low-rank linear model**

Consider the setting where for each sample the response _**y** ∈_ R _[d]_ is given by

**==> picture [52 x 11] intentionally omitted <==**

with _ϵ ∈_ R _[d]_ representing the noise. Here _**A** ∈_ R _[d][×][d]_ is a full rank matrix.

We consider a network with _T_ layers where _ft_ ( _**z**_ ) = _**V** t_

(there is no activation). We let _rt_ := rank( _**V** t_ ) and define the collective rank _r∗_ :=[�] _[T] t_ =1 _[r][t]_[.][We][assume] _[r][∗][<][d]_[,] i.e., the collective rank of all layers still is lower than the ambient dimension _d_ .

We next focus on four architectures: Baseline (where there is no residual connection), ResNet, GRN-v1 and GRN-v2 and characterize the class of models which can be expressed by each of these architectures. We assume each architecture to have _T_ layers.

**Baseline.** In this architecture, there is no residual connection � and so the model is given by _**y**_ =[�] _[T] t_ =1 _**[V]**[t]_ _**[x]**_[.][We denote by] _C_ base the class of functions that can be represented by such architecture.

**ResNets.** In this case, we have � _**y**_ =[�] _[T] t_ =1[(] _**[I]**_[+] _**[V]**[t]_[)] _**[x]**_[. Denote] by _C_ res as the class of functions that can be represented by such architecture.

� **GRN-v1.** In this case, we have _**y**_ = _**G** T_ +1 _**b** T_ +1, with _**b** T_ +1 a ( _T_ + 1)-dimensional vector as described in Section 3. Denote by _C_ GRN _−_ v1 the class of functions that can be represented by such architecture.

� **GRN-v2.** In this case, we have _**y**_ = ( _**G** T_ +1 _⊙_ _**b** T_ +1) **1** , where _**b** T_ +1 is _d ×_ ( _T_ + 1) matrix as described in Section 3. We denote by _C_ GRN _−_ v2 the class of functions that can be represented by such architecture.

� **GRN-v3.** In this case, we have _**y**_ = ( _**G** T_ +1 _⊙_ ( _**b** T_ +1 + _**w**_ ¯ _T_ +1)) **1** , where _**b** T_ +1 is _d ×_ ( _T_ + 1) matrix and _**w**_ ¯ _T_ +1 is _d_ dimensional vector as described in Section 3. We denote by _C_ GRN _−_ v3 the class of functions that can be represented by such architecture.

**Theorem 4.1.** _For the low rank linear model we have:_

- _C_ base = _{_ _**x** �→_ _**Mx**_ : rank( _**M**_ ) _≤_ min( _rt_ ) _[T] t_ =1 _[}][.]_

- _C_ res = _{_ _**x** �→_ ( _**I**_ + _**M**_ ) _**x**_ : rank( _**M**_ ) _≤ r∗}._

- _C_ GRN _−_ v1 = _{_ _**x** �→_ ( _α_ _**I**_ + _**M**_ ) _**x**_ : rank( _**M**_ ) _≤ r∗}._

- _C_ GRN _−_ v2 = _{_ _**x** �→_ ( _**D**_ + _**M**_ ) _**x**_ : rank( _**M**_ ) _≤_

- _r∗,_ _**D** is diagonal}._

- _C_ GRN _−_ v3 _⊃ {_ _**x** �→_ ( _**D**_ + _**M**_ ) _**x**_ + _σ_ ( _**w**_[T] _**x**_ ) _**x**_ :

- rank( _**M**_ ) _≤ r∗,_ _**D** is diagonal,_ _**w** ∈_ R _[d][×]_[1] _}._

## **4.2. Trade-off between test error and model complexity**

In the previous section, we characterized the class of models that can be expressed by each architecture. Next, we study the trade-off between the optimal test error achievable by each model and the model complexity, defined as the number of its parameters.

Note that all the classes of models characterized in Theorem 4.1 are linear functions. For a linear model _**x** �→_ _**Ax**_[�] ,

its test error (model risk) is given by

**==> picture [213 x 84] intentionally omitted <==**

where we assumes that E[ _**xx**_[T] ] = _**I**_ (isotropic features). Since the term _σ_[2] is constant (independent of model _**A**_[�] ) we will drop it in sequel without effecting our discussion and focus on the excess risk. For a class of models _C_ we use the notation ER _[∗]_ ( _C_ ) to indicate the minimum excess risk achievable over the class _C_ :

**==> picture [119 x 24] intentionally omitted <==**

Note that ER _[∗]_ ( _C_ base( _T_ )) is obtained by the best _r_ -rank approximation to _**A**_ and ER _[∗]_ ( _C_ res) is obtained by the best _rT_ -rank approximation to _**A** −_ _**I**_ , both of which have simple characterization in terms of the singular values of _**A**_ and _**A** −_ _**I**_ , by using the celebrated Eckart–Young–Mirsky theorem. Deriving ER _[∗]_ ( _C_ GRN _−_ v1( _T_ )) and ER _[∗]_ ( _C_ GenB(T)) are more complicated. In the next theorem, we establish upper bounds on them.

**Theorem 4.2.** _Consider the singular value decomposition_ _**A** −_ _**I**_ = _**U**_ **Σ** _**V**_[T] _. For a given m ∈_ [ _d_ ] _, let_ _**U** m,_ **Σ** _m,_ _**V** m be the top m singular vectors and singular values and define_ **∆** := _**A** −_ _**I** −_ _**U** r∗_ **Σ** _r∗_ _**V** r_[T] _∗[, where][ r][∗]_[:=][ �] _[T] ℓ_ =1 _[r][ℓ][.][We then] have_

**==> picture [236 x 92] intentionally omitted <==**

_where {_ ∆ _ii}[d] i_ =1 _[are the diagonal entries of]_ **[ ∆]** _[.][In addition,]_

**==> picture [173 x 76] intentionally omitted <==**

_Here ν_ max _denotes the maximum eigenvalue of_ **[∆]**[+] 2 **[∆]**[T] _−_ 1 _d−r∗_[trace(] **[∆]**[)] _**[U]**[r][∗][,][⊥]_ _**[U]**_[ T] _r∗,⊥[and]_[ ˜] _[ν]_[max] _[ denotes the maximum] eigenvalue of_ **[∆]**[+] 2 **[∆]**[T] _− diag_ ( **∆** ) _._

**==> picture [418 x 125] intentionally omitted <==**

8000<br>25000<br>6000<br>20000<br>4000 G1, d = 100<br>15000 G2, d = 100<br>2000<br>G1, d = 200<br>0 10000 G2, d = 200<br>G1, d = 300<br>2000 5000 G2, d = 300<br>4000 0<br>6000<br>5000<br>0 20 40 60 80 100 2 4 6 8 10<br>r * min<br>Reduction in test error Reduction in test error<br>**----- End of picture text -----**<br>

_Figure 5._ Gain in the model performance achieved by GRN-v1 and GRN-v2 over ResNet. The plots represents the lower bounds for _G_ 1 and _G_ 2 given in Theorem 4.3(ii). Observe that the gain at larger dimension _d_ is higher. Left panel shows that the gain decreases as the collective rank _r∗_ of ResNet increases ( _λ_ min = 5, _λ_ max = 10). Right panel shows that the gain increases as the complexity of the target task ( _κ_ = _λ_ min _/λ_ max) increases ( _λ_ max = 10 and _r∗_ = 50 ).

We proceed by discussing the model complexity for each of the architectures, in terms of model size. The number of parameters for ResNet is given by 2 _dr∗_ , for GRN-v1 is given by 2 _dr∗_ + _T_ ( _T −_ 1) _/_ 2, and for GRN-v2 is given by 2 _dr∗_ + _dT_ ( _T −_ 1) _/_ 2. Note that by Theorem 4.2, if GRN-v1 and GRN-v2 achieve better Excess risk-model size trade-off compared to ResNet, then we can make this improvement arbitrarily strong by scaling _**A** −_ _**I**_ (and so **∆** ).

In the next theorem, we focus on GRN-v1 and GRN-v2 and provide sufficient conditions under which they achieve a better excess risk-model size trade-off. In the second part of the theorem, we also lower bound the improvement that GRN-v1 and GRN-v2 achieve in excess risk compared to ResNet, with using the same number of parameters.

**Theorem 4.3.** _Assume that_ _**A** −_ _**I** ⪰_ **0** _and let λ_ max _and λ_ min _>_ 0 _respectively denote the maximum and the minimum eigenvalues of_ _**A** −_ _**I** . Define κ_ := _λ_ min _/λ_ max _≤_ 1 _. Consider a ResNet model with collective rank r∗_ := � _Tt_ =1 _[r][t][.]_

( _i_ ) _If_

**==> picture [189 x 19] intentionally omitted <==**

_then GRN-v1 achieves a better excess risk-model size tradeoff compared to ResNet. In addition, if_

**==> picture [189 x 14] intentionally omitted <==**

_then GRN-v2 achieves a better trade-off compared to ResNet._

_Also, GRN-v3 achieves a better trade-off compared to ResNet, if_

**==> picture [236 x 42] intentionally omitted <==**

( _ii_ ) _Consider C_ GRN _−_ v1 _and C_ GRN _−_ v2 _, the class of models that can be expressed by the GRN-v1 and GRN-v2 architectures with the same number of parameters as a ResNet model with T layers and collective rank r∗. Define G_ 1 := ER _[∗]_ ( _C_ res) _−_ ER _[∗]_ ( _C_ GRN _−_ v1) _and G_ 2 := ER _[∗]_ ( _C_ res) _−_ ER _[∗]_ ( _C_ GRN _−_ v2) _as the reduction in the optimal excess risk achievable by these classes compared to the optimal excess risk of ResNet. We have_

**==> picture [231 x 74] intentionally omitted <==**

Our next result quantitatively shows the reduction in the collective rank one can achieve by GRNs, while maintaining the same test error as ResNet.

**Proposition 4.4.** _Consider a ResNet with collective rank r∗_ =[�] _[T] t_ =1 _[r][t] < d. A GRN-v1 or GRN-v2 model can achieve a smaller test error with collective rank r∗[′][,] where r∗[′]_[:=] _r∗_ 1 _−−dκκ_[2][2] _< r∗. Likewise, a GRN-v3 model achieve a smaller test error with collective rank_ ˜ _r∗, where r_ ˜ _∗[′]_[:=] _r∗_ 1 _−−dηη_[2][2] _< r∗, with η_ = � ( _κ_ (1+ _ξ_ 1+0) _−ξξ_ 00)[2] + _ξ_ 0 _and ξ_ 0 = _π_ ( _d_[2] 1 _−_ 1) _[.]_

## **4.3. Insights from the analysis**

Theorem 4.3 allows us to elucidate the role of different factors on the gain achieved by GRNs.

**Role of target task complexity.** Note that _κ_ = _λ_ min _/λ_ max _∈_ [0 _,_ 1] is a measure of complexity of the target task. Specifically, as _κ_ decreases, the matrix _**A**_ becomes closer to a low rank matrix, and hence learning it with low rank models becomes easier. Observe that the thresholds

given by the right hand side of (4.1)-(4.3) are increasing in _κ_ , i.e., for more complex tasks we see a wider range of collective rank where GRNs outperforms the trade-off achieved by ResNet. Another way to interpret Theorem 4.3(i) is that for a fixed target task (and so fixed _κ_ ), if the collective rank _r∗_ is above this threshold, the ResNet is already rich enough that it is hard to improve upon its trade-off.

**Role of collective rank.** Observe that the lower bound on the gains _G_ 1, _G_ 2, _G_ 3 given by Theorem 4.3(ii) are decreasing in _r∗_ . In other words, when the collective rank _r∗_ of ResNet becomes smaller, the level of information dilution occurring in ResNet increases, giving GRNs a better leverage to improve model perplexity with the same number of parameters.

**Role of input dimension.** Note that the upper bounds on _r∗_ given by (4.1) to (4.3) increase with the input dimension _d_ . Furthermore, the lower bounds on the gains _G_ 1, _G_ 2, _G_ 3 given in Theorem 4.3(ii) also increase with _d_ . Therefore, for larger input dimensions, we have both a wider range for _r∗_ where GRNs outperforms the trade-off achieved by ResNet, and moreover, we obtain a larger gain in reducing model error.

We refer to Figure 5 for an illustration of these trends.

## **4.4. Extension to nonlinear models**

We recall the definition of Bottleneck rank from (Jacot, 2023). For a function _f_ : Ω _�→_ R _[d]_ , its Bottleneck rank, denoted by rank _BN_ ( _f,_ Ω) is the smallest integer _k_ such that _f_ can be factorized as _f_ = _h ◦ g_ with inner dimension _k_ (i,e, _g_ : Ω _�→_ R _[k]_ and _h_ : R _[k] �→_ R _[d]_ ) It is also closely related to the Jacobian rank of a function defined as rank _**J**_ ( _f_ ) = max _**x** ∈_ Ω rank[ _**J** f_ ( _**x**_ )]. In general, rank _**J**_ ( _f_ ) _≤_ rank _BN_ ( _f_ ), but for functions of the form _f_ = _ψ ◦_ _**A** ◦ ϕ_ (for a linear map _**A**_ and two bijections _ψ_ and _ϕ_ ), we have rank _**J**_ ( _f_ ) = rank _BN_ ( _f_ ) = rank( _**A**_ ). These two notions of rank satisfy the following properties (Jacot, 2023):

- rank( _f ◦ g_ ) _≤_ min _{_ rank( _f_ ) _,_ rank( _g_ ) _}_

- rank( _f_ + _g_ ) _≤_ rank( _f_ ) + rank( _g_ )

**Proposition 4.5.** _Consider an MLP with ft_ ( _**z**_ ) = _**V** tφ_ ( _**U** t_ _**z**_ ) _with_ _**U** t ∈_ R _[r][t][×][d] ,_ _**V** t ∈_ R _[d][×][r][t] . Denote by r∗_ :=[�] _[T] t_ =1 _[r][t][ the collective rank of the network.][We have]_

- _C_ base _⊆_ � _f_ : rank _BN_ ( _f_ ) _≤_ min( _rt_ ) _[T] t_ =1� _._

- _C_ res _⊆{id_ + _f_ : rank _BN_ ( _f_ ) _≤ r∗}._

- _C_ GRN _−_ v1 _⊆{α · id_ + _f_ : rank _BN_ ( _f_ ) _≤ r∗}._

- _C_ GRN _−_ v2 _⊆{g_ : _g_ ( _**x**_ ) = _**Dx**_ + _f_ ( _**x**_ ) : rank _BN_ ( _f_ ) _≤_

- _r∗,_ _**D** is diagonal}._

## **5. Experiments**

We conduct experiments on language modeling and image classification tasks to evaluate the effectiveness of DCA and to validate our theoretical insights. For the language modeling tasks, the performance of DCA is compared against the standard transformer (Vaswani, 2017) on the LM1B (Chelba et al., 2013) and C4 (Raffel et al., 2020a) datasets. Unless stated otherwise, each model has an embedding dimension of 512 and an MLP dimension of four times the embedding dimension. By default, DCA uses a stack of all the previous layer outputs as input to the GRNs. When DCA includes only the first and last- _k_ layer outputs explicitly in the input stack (see Section 3.1), then this is denoted as _k_ -DCA.

Each model is trained with a sequence length of 128 and a batch size of 2048 over 64 TPUs for 500k steps, totaling 131B tokens. We use the AdamW optimizer (Loshchilov & Hutter, 2017) with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 98, a weight decay of 0 _._ 1, and a learning rate of 0 _._ 0016 with 1000 warmup steps and an inverse square root schedule (Raffel et al., 2020b).

**Model depth scaling.** For the first experiment, we pre-train a transformer and DCA on LM1B. We increase the model depth from 6 to 42 layers and show the relation between perplexity (Jelinek et al., 1977) and model size in Figure 6. The figure shows that DCA obtains a lower perplexity for a given parameter budget. Notably, the 30-layer DCA model obtains a better perplexity than the 42-layer transformer, making DCA more parameter-efficient than adding layers.

**==> picture [186 x 135] intentionally omitted <==**

19<br>Transformer<br>DCA<br>18<br>17<br>16<br>15<br>14<br>0.6 0.8 1.0 1.2 1.4 1.6<br>Number of parameters 1e8<br>Perplexity<br>**----- End of picture text -----**<br>

_Figure 6._ Perplexity on LM1B with 6, 12, 18, 24, 30, 36, and 42 layer transformer and DCA models.

**First and last-** _k_ **.** DCA can be made more efficient by including only the first and last- _k_ layer outputs explicitly in the input stack to the GRNs (see Section 3.1). In this experiment, we study the effect of _k_ on a 24-layer model’s efficiency and quality. Table 1 shows that reducing _k_ speeds up training while only slightly increasing the perplexity. Either small or large _k_ obtain good training efficiency, as DCA then obtains the final perplexity of the transformer in a third of the time. Setting _k_ = 2 results in a model with 48%

lower inference latency compared to _k_ = 24, thus setting _k_ to be small results in efficient training and fast inference.

_Table 1._ Training speed in batches per second, normalized time for a method to reach the perplexity of the transformer, and the final perplexity (PPL) of the transformer and DCA with varying _k_ .

|METHOD|SPEED<br>TIME<br>PPL|
|TRANSFORMER<br>1-DCA<br>2-DCA<br>4-DCA<br>8-DCA<br>16-DCA<br>24-DCA|**8.14**_±_**0.18**<br>1.00<br>15.14_±_0.06<br>5.62_±_0.04<br>**0.33**<br>14.48_±_0.05<br>5.39_±_0.06<br>**0.33**<br>14.41_±_0.04<br>5.01_±_0.12<br>0.37<br>14.50_±_0.03<br>4.35_±_0.14<br>0.47<br>14.49_±_0.02<br>3.86_±_0.08<br>0.40<br>14.35_±_0.07<br>3.72_±_0.08<br>0.39<br>**14.35**_±_**0.00**|

**Training time.** The effectiveness of a model architecture heavily depends on its training efficiency. Figure 7 shows the training time-perplexity trade-off for 24, 36, and 42 layer transformer and 2-DCA models. The figure shows that 2-DCA achieves better perplexity for a given training time, highlighting the training efficiency of DCA. The training time versus perplexity results when DCA uses all previous layer outputs in the GRNs are provided in Appendix F.

**==> picture [189 x 135] intentionally omitted <==**

Layers<br>17<br>24<br>36<br>42<br>16<br>Model<br>Transformer<br>2-DCA<br>15<br>14<br>0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75<br>Training time (s) 1e5<br>Perplexity<br>**----- End of picture text -----**<br>

_Figure 7._ Perplexity on LM1B versus the training time with transformer and 2-DCA models of various depths.

**Model width scaling.** Our theoretical results indicate that the benefit of GRN is inversely related to the rank of the model. With this experiment, we validate whether the theoretical results carry over to the transformer architecture by varying the model width. Table 2 shows the final perplexity of a 12-layer model with an embedding dimension ranging from 64 till 1024, pre-trained on LM1B. The delta column, with the difference between the transformer and DCA, shows that the benefit of DCA is reduced as the width of the model increases, which is consistent with our theoretical results. These results are in contrast with the depth scaling results, where the improvement of DCA is maintained for deeper models.

**Model scaling.** For this experiment, we train transformer

_Table 2._ Perplexity on LM1B for models of varying widths.

|WIDTH|TRANSFORMER<br>DCA<br>DELTA|
|64<br>192<br>384<br>768<br>1024|45.75_±_0.06<br>42.94_±_0.07<br>-2.82<br>25.49_±_0.15<br>23.92_±_0.04<br>-1.57<br>18.86_±_0.04<br>17.83_±_0.04<br>-1.03<br>14.70_±_0.04<br>14.11_±_0.07<br>-0.59<br>13.61_±_0.01<br>13.22_±_0.06<br>-0.39|

and 8-DCA models of increasing size on the C4 dataset. The results in Table 3 show that DCA consistently outperforms the standard transformer model. The absolute improvement in perplexity decreases for large models, which is consistent with the width scaling results. The perplexity throughout training is provided in Appendix G.

_Table 3._ Perplexity on C4 for models of varying depths and widths.

|D<br>W|PARAMS<br>TRANSF.<br>8-DCA<br>DELTA|
|9<br>771<br>18<br>771<br>13<br>1111<br>18<br>1111<br>18<br>1600|75M<br>27.876<br>26.443<br>-1.443<br>124M<br>23.013<br>21.810<br>-1.203<br>179M<br>21.570<br>20.461<br>-1.109<br>234M<br>19.756<br>18.824<br>-0.932<br>449M<br>17.166<br>16.764<br>-0.402|

**Retrofitting pre-trained models.** Since our method is identical to a standard residual network at initialization, adding DCA to a pre-trained model does not alter its function. In Table 4, we compare continuing training the pre-trained model with adding DCA to the pre-trained model. Incorporating DCA results in a perplexity improvement of 0.19 after 60k extra training steps, compared to just 0.02 for the transformer. Thus, pre-trained models with a residual architecture can also benefit from incorporating DCA.

_Table 4._ Perplexity on LM1B for extended training of 6-layer models. DCA is added to a 500k steps pre-trained transformer.

|STEPS|TRANSFORMER<br>DCA<br>DELTA|
|500K<br>500K+ 20K<br>500K+ 40K<br>500K+ 60K|18.98_±_0.01<br>18.98_±_0.01<br>0.00<br>18.96_±_0.02<br>18.81_±_0.01<br>-0.15<br>18.96_±_0.01<br>18.79_±_0.03<br>-0.17<br>18.96_±_0.01<br>18.79_±_0.04<br>-0.17|

**Training stability.** The occurrence of loss spikes is a problem when training large models as they can disrupt an expensive training run (Chowdhery et al., 2023). In Figures 7 and 8, we indeed observe clear loss spikes with the transformer model. Interestingly, training DCA is more stable, showing no significant loss spikes even for large models. This constitutes an important benefit of DCA.

**Comparison with related work.** We compare the perplexity of DCA with those obtained by the recent related works LAuReL (Menghani et al., 2024), DenseFormer (Pagliardini

et al., 2024), and Hyper-Connections (dynamic) (Zhu et al., 2024) in Table 5. DCA improves upon the prior best method, hyper-connections, with a difference in perplexity of 0.59, which is the biggest improvement among the methods.

_Table 5._ Perplexity (PPL) and parameter count on LM1B using a 6-layer model, comparing DCA with related work.

|METHOD|PARAMS<br>PPL|
|TRANSFORMER<br>LAUREL-PA<br>1X1-DENSEFORMER<br>HYPER-CONNECTIONS|49.65M<br>18.98_±_0.01<br>49.75M<br>18.99_±_0.05<br>49.65M<br>18.80_±_0.11<br>49.68M<br>18.65_±_0.03|
|DCA (OURS)|49.73M<br>**18.06**_±_**0.01**|

_Table 6._ Perplexity (PPL) on C4 using a 13-layer model and embedding dimension 1111, comparing DCA with related work. The baseline model has roughly 179M parameters.

|METHOD|PPL|
|TRANSFORMER<br>LAUREL-PA<br>1X1-DENSEFORMER<br>HYPER-CONNECTIONS(STACK SIZE=4)<br>HYPER-CONNECTIONS(STACK SIZE=10)|21.534<br>20.951<br>21.168<br>21.077<br>20.718|
|8-DCA (OURS)|**20.392**|

**Ablation study.** To determine the relative gain of each of the proposed generalizations, in Table 7 we show the perplexity obtained by each method described in Section 3. The GRN versions use one GRN instance per decoder block. DCA, in contrast, uses three independent instances of GRN-v3 per decoder block. The biggest improvement in perplexity comes from GRN-v1, followed by DCA and GRN-v2.

_Table 7._ Ablation study of DCA, showing the parameter count and the perplexity (PPL) on LM1B with a 6-layer model.

|ABLATION|PARAMS<br>PPL|
|TRANSFORMER<br>GRN-V1<br>GRN-V2<br>GRN-V3<br>DCA|49.65M<br>18.98_±_0.02<br>49.65M<br>18.80_±_0.11<br>49.66M<br>18.43_±_0.04<br>49.68M<br>18.41_±_0.10<br>49.73M<br>**18.06**_±_**0.01**|

**ImageNet classification.** In addition to the language modelling experiments, we also experiment with image classification using the ImageNet dataset and the vision transformer (ViT) model (Dosovitskiy et al., 2021). Since the ViT model is transformer-based, DCA can be incorporate in the same way as for the language models presented earlier. In Table 8, we present the results on the ViT-S/16 model (22M parameters) and follow the experimental setup by Beyer et al. (2022). The results show a 0.7% improvement in

classification accuracy, demonstrating that DCA effectively generalizes to the vision domain.

_Table 8._ Loss and Accuracy on ImageNet classification.

|METHOD|LOSS<br>ACCURACY|
|VIT<br>VIT + DCA (OURS)|0.5698<br>76.4<br>**0.5284**<br>**77.1**|

## **6. Conclusion**

This paper introduces DeepCrossAttention (DCA), a novel transformer architecture that enhances the flow of information across layers. It achieves lower perplexity for a given parameter budget and training time for a minimal increase in model parameters. DCA enables dynamic interactions between layer outputs by building on three generalizations of the standard residual network (GRN). We showed theoretically that GRN obtains a better test error-model complexity trade-off. In our DCA experiments we observe significant improvements in model stability, convergence, and quality.

## **Acknowledgment**

Adel Javanmard is supported in part by the NSF Award DMS-2311024, the Sloan fellowship in Mathematics, an Adobe Faculty Research Award, an Amazon Faculty Research Award, and an iORB grant from USC Marshall School of Business. The authors are grateful to anonymous reviewers for their feedback on improving this paper.

## **Impact Statement**

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## **References**

- Beyer, L., Zhai, X., and Kolesnikov, A. Better plain vit baselines for imagenet-1k. _arXiv preprint arXiv:2205.01580_ , 2022.

- Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., and Robinson, T. One billion word benchmark for measuring progress in statistical language modeling. _arXiv preprint arXiv:1312.3005_ , 2013.

- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. _Journal of Machine Learning Research_ , 24(240):1–113, 2023.

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., and Zhai, X. An image is worth 16x16 words: Transformers for image recognition at scale. _International Conference on Learning Representations_ , 2021.

- Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., and Zhou, X. Hyper-connections. _arXiv preprint arXiv:2409.19606_ , 2024.

- Gong, Y., Chung, Y.-A., and Glass, J. Ast: Audio spectrogram transformer. In _Interspeech 2021_ , pp. 571–575, 2021.

- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016.

- Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 4700–4708, 2017.

- Jacot, A. Implicit bias of large depth networks: a notion of rank for nonlinear functions. _The Eleventh International Conference on Learning Representations_ , 2023.

- Jelinek, F., Mercer, R. L., Bahl, L. R., and Baker, J. K. Perplexity—a measure of the difficulty of speech recognition tasks. _The Journal of the Acoustical Society of America_ , 62(S1):S63–S63, 1977.

- Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017.

- Menghani, G., Kumar, R., and Kumar, S. Laurel: Learned augmented residual layer. In _Workshop on Efficient Systems for Foundation Models II_ , 2024.

- Pagliardini, M., Mohtashami, A., Fleuret, F., and Jaggi, M. Denseformer: Enhancing information flow in transformers via depth weighted averaging. _arXiv preprint arXiv:2402.02622_ , 2024.

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research_ , 21 (140):1–67, 2020a.

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ , 21 (140):1–67, 2020b.

- Srivastava, R. K., Greff, K., and Schmidhuber, J. Training very deep networks. _Advances in neural information processing systems_ , 28, 2015.

- Vaswani, A. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017.

## **A. Proof of Theorem 4.1**

We restate each of the claims in the theorem statement, followed by its proof.

- _C_ base = _{_ _**x** �→_ _**Mx**_ : rank( _**M**_ ) _≤_ min( _rt_ ) _[T] t_ =1 _[}]_[.]

Note that by the inequality rank( _**AB**_ ) _≤_ min _{_ rank( _**A**_ ) _,_ rank( _**B**_ ) _}_ , if _**M**_ is of the form[�] _[T] t_ =1 _**[V]**[t]_[then][rank(] _**[M]**_[)] _[≤]_ min( _rt_ ) _[T] t_ =1[.][For the other direction consider any matrix] _**[ M]**_[with][ rank(] _**[M]**_[)][=] _[r]_[0] _[≤]_[min(] _[r][t]_[)] _[T] t_ =1[, and its SVD as] _**[ M]**_[=] _**P SQ**_[T] with _**P** ,_ _**Q** ∈_ R _[d][×][r]_[0] with full column ranks. By setting, _**V**_ 1 = _**P SQ**_[T] and _**V**_ 2 = _. . ._ = _**V** T_ = _**QQ**_[T] we have _**M**_ =[�] _[T] t_ =1 _**[V]**[t]_[, because] _**[ Q]**_[T] _**[Q]**_[ =] _**[ I]**_[and also][ rank(] _**[V]**[t]_[) =] _[ r]_[0] _[≤]_[min(] _[r][t]_[)] _[T] t_ =1 _[≤][r][t]_[.]

_• C_ res = _{_ _**x** �→_ ( _**I**_ + _**M**_ ) _**x**_ : rank( _**M**_ ) _≤ r∗}_ We have

**==> picture [261 x 116] intentionally omitted <==**

Note that each of the summand is of rank at most _rt_ , so it can be written as _**I**_ + _**M**_ with rank( _**M**_ ) _≤_[�] _[T] t_ =1 _[r][t]_[.][Hence] _T_ � _t_ =1[(] _**[I]**_[+] _**[ V]**[t]_[)] _**[x]**[ ∈C]_[res][.]

We next show that any _**I**_ + _**M**_ with rank( _**M**_ ) := _r ≤_[�] _[T] t_ =1 _[r][t]_[can be written as][ �] _[T] t_ =1[(] _**[I]**_[+] _**[ V]**[t]_[)][ with][ rank(] _**[V]**[t]_[)] _[ ≤][r][t]_[for] _t ∈_ [ _T_ ]. We show this claim by induction. For the basis ( _T_ = 1), we can take _**V**_ 1 = _**M**_ . To complete the induction step, we need to find _**V** ∈_ R _[d][×][d]_ such that rank( _**V**_ ) = _rT_ and ( _**I**_ + _**V**_ ) _[−]_[1] ( _**I**_ + _**M**_ ) _−_ _**I**_ is of rank at most[�] _[T] t_ =1 _[ −]_[1] _[r][t]_[.][Then by the] induction hypothesis, we can write

**==> picture [155 x 30] intentionally omitted <==**

with rank( _**V** t_ ) _≤ rt_ , which completes the proof. Without loss of generality, we assume _rT ≤ r_ ; otherwise we can take _**V** T_ = _**M**_ and _**V** t_ = **0** for _t ≤ T −_ 1.

To find such _**V**_ we write _**M**_ = _**P Q**_[T] with _**P** ,_ _**Q** ∈_ R _[d][×][r]_ having full column rank. Define _**P**_ 1 _,_ _**Q**_ 1 _∈_ R _[d][×][r][T]_ obtaining by considering the first _rT_ columns of _**P**_ and _**Q**_ . Additionally, define

**==> picture [348 x 13] intentionally omitted <==**

We next construct _**V**_ by setting _**V**_ := _**BC**_[T] . Clearly, rank( _**V**_ ) = _rT_ . We also have

**==> picture [359 x 46] intentionally omitted <==**

Here we consider the notation _**P**_ = [ _**P**_ 1 _|_ _**P** ∼_ 1] and _**Q**_ = [ _**Q**_ 1 _|_ _**Q** ∼_ 1]. The second step above follows from the Woodbury matrix identity. Rearranging the terms we have

**==> picture [488 x 45] intentionally omitted <==**

**==> picture [333 x 12] intentionally omitted <==**

To do this, we next show that

**==> picture [347 x 12] intentionally omitted <==**

Recalling (A.1) we have _**P**_ 1 = _**B**_ ( _**I**_ + _**Q**_[T] 1 _**[P]**_[1][)][.][Also]

**==> picture [355 x 78] intentionally omitted <==**

Therefore, _**P**_ 1 = _**B**_ ( _**I**_ + _**C**_[T] _**B**_ ) = ( _**I**_ + _**BC**_[T] ) _**B**_ . Likewise, recalling (A.1) we have _**Q**_ 1 = _**C**_ ( _**I**_ + _**P**_ 1[T] _**[Q]**_[1][)] _[−]_[1][.][Hence,]

**==> picture [195 x 13] intentionally omitted <==**

using (A.6). This completes the proof of (A.4) and so (A.3).

Invoking (A.2) we get

**==> picture [218 x 13] intentionally omitted <==**

which is of rank at most _r − rT ≤_[�] _[T] t_ =1 _[ −]_[1] _[r][t]_[, which completes the proof of the induction step.]

_• C_ GRN _−_ v1 = _{_ _**x** �→_ ( _α_ _**I**_ + _**M**_ ) _**x**_ : rank( _**M**_ ) _≤ r∗}_ . We prove this claim by induction. The induction basis ( _T_ = 0) follows readily since _**G**_ 1 _**b**_ 1 = _b_ 1 _**x**_ . Assume the induction _b_ 1 hypothesis for _t_ . We have _ft_ ( _gt_ ( _**x**_ )) = _**V** t_ _**G** t_ _**b** t_ and so _**G** t_ +1 = [ _**V** t_ _**G** t_ _**b** t |_ _**G** t_ ]. Writing _**b** t_ +1 = we obtain � _**b** ∼_ 1�

**==> picture [141 x 10] intentionally omitted <==**

By induction hypothesis, _**G** t_ _**b** ∼_ 1 is the set of functions of the form ( _α_ _**I**_ + _**M**_ ) _**x**_ with rank( _**M**_ ) _≤_[�] _[t] ℓ[−]_ =1[1] _[r][ℓ]_[.]

Since rank( _**V** t_ ) _≤ rt_ the set of functions that can be represented as _**G** t_ +1 _**b** t_ +1 is a subset of ( _α_ _**I**_ + _**M**_ ) _**x**_ with rank( _**M**_ ) _≤_ � _tℓ_ =1 _[r][ℓ]_[.][Conversely, any given] _[ M]_[of rank][ �] _[t] ℓ_ =1 _[r][ℓ]_[can be written as] _**[ M]**_[=] _**[M]**_[1][ +] _**[ V]**_[with][ rank(] _**[M]**_[1][)][=][�] _[t] ℓ[−]_ =1[1] _[r][ℓ]_[and] rank( _**V**_ ) = _rt_ . By induction hypothesis, ( _α_ _**I**_ + _**M**_ 1) _**x**_ can be expressed by the term _**G** t_ _**b** ∼_ 1. In addition, _**V x**_ can also be expressed by the term _**V** t_ _**G** t_ _**b** tb_ 1, by taking _**V** t_ = _**V**_ , _**b** t_ = (0 _,_ 0 _, . . . ,_ 1)[T] , _b_ 1 = 1, which is possible since they are free from the choice of _**b** ∼_ 1.

Hence, _**G** t_ +1 _**b** t_ +1 the set of functions of the form ( _α_ _**I**_ + _**M**_ ) _**x**_ with rank( _**M**_ ) _≤_[�] _[t] ℓ_ =1 _[r][ℓ]_[, completing the induction step.]

_• C_ GRN _−_ v2 = _{_ _**x** �→_ ( _**D**_ + _**M**_ ) _**x**_ : rank( _**M**_ ) _≤ r∗,_ _**D**_ is diagonal _}_ .

The proof follows similar to that of GRN-v1. For the induction basis ( _T_ = 0), we have ( _**G**_ 1 _⊙_ _**b**_ 1) **1** = ( _**x** ⊙_ _**b**_ 1) **1** = diag( _**b**_ 1) _**x**_ . Assume the induction hypothesis for _t_ . We have _ft_ ( _gt_ ( _**x**_ )) = _**V** t_ ( _**G** t ⊙_ _**b** t_ ) **1** and so _**G** t_ +1 = [ _**V** t_ ( _**G** t ⊙_ _**b** t_ ) **1** _|_ _**G** t_ ]. Writing _**b** t_ +1 = � _**b**_[(1)] _t_ +1 _[|]_ _**[b]**_[(] _t_ +1 _[∼]_[1)] � we obtain

**==> picture [231 x 15] intentionally omitted <==**

and hence

**==> picture [259 x 15] intentionally omitted <==**

By induction hypothesis, ( _**G** t ⊙_ _**b**_[(] _t_ +1 _[∼]_[1)][)] **[1]**[ is the set of functions of the form][ (] _**[D]**_[ +] _**[ M]**_[)] _**[x]**_[ with][ rank(] _**[M]**_[)] _[≤]_[�] _[t] ℓ[−]_ =1[1] _[r][ℓ]_[.][By] varying _**V** t,_ _**b** t_ and _**b**_[(1)] _t_[, the term][ diag][(] _**[b]**_[(1)] _t_ +1[)] _**[V]**[t]_[(] _**[G]**[t][ ⊙]_ _**[b]**[t]_[)] **[1]**[ covers all functions of the form] _**[ V x]**_[ with] _**[ V]**_[a] _[ d][ ×][ d]_[ matrices] of rank _rt_ (for given _**V**_ of rank _rt_ , take _**V** t_ = _**V**_ , _**b** t_ = [ **0** _|_ **0** _| . . . |_ **1** ], _**b**_[(1)] _t_ +1[=] **[ 1]**[, which is possible since they are free from]

the choice of _**b** t_[(] +1 _[∼]_[1)][).][Hence,][ (] _**[G]**[t]_[+1] _[ ⊙]_ _**[b]**[t]_[+1][)] **[1]**[ is the set of functions of the form][ (] _**[D]**_[ +] _**[ M]**_[)] _**[x]**_[ with][ rank(] _**[M]**_[)] _[ ≤]_[�] _[t] ℓ_ =1 _[r][ℓ]_[,] completing the induction step.

_• C_ GRN _−_ v3 _⊃{_ _**x** �→_ ( _**D**_ + _**M**_ ) _**x**_ + _σ_ ( _**w**_[T] _**x**_ ) _**x**_ : rank( _**M**_ ) _≤ r∗,_ _**D**_ is diagonal _,_ _**w** ∈_ R _[d][×]_[1] _}_ . We prove that

**==> picture [340 x 31] intentionally omitted <==**

we show the claim by induction on the number of layers. For the basis _T_ = 0, we have

**==> picture [273 x 12] intentionally omitted <==**

Assume the induction hypothesis for _t_ . We have _ft_ ( _gt_ ( _**x**_ )) = _**V** t_ ( _**G** t ⊙_ _**c** t_ ) **1** with _**c** t_ := _**b** t_ + **1** _σ_ ( _**w** t_[T] _**[G]**[t]_[)][ and so] _**[ G]**[t]_[+1][=] [ _**V** t_ ( _**G** t ⊙_ _**c** t_ ) **1** _|_ _**G** t_ ]. Writing _**c** t_ +1 = � _**c**_[(1)] _t_ +1 _[|]_ _**[c]**_[(] _t_ +1 _[∼]_[1)] � we obtain

**==> picture [231 x 15] intentionally omitted <==**

and hence

**==> picture [259 x 15] intentionally omitted <==**

We take _**b** t_ = [ **0** _|_ **0** _| . . . |_ **1** ] and _**w** t_ = **0** , and so _**c** t_ = [ **0** _|_ **0** _| . . . |_ **1** ]. We also take _**c**_[(1)] _t_ +1[=] **[ 1]**[.][So,]

**==> picture [175 x 15] intentionally omitted <==**

By induction hypothesis, ( _**G** t ⊙_ _**c**_[(] _t_ +1 _[∼]_[1)][)] **[1]**[ covers all the functions of the form] _**[ x]**[ �→]_[�] _[t] ℓ[−]_ =1[1] _**[V]**[ℓ]_ _**[x]**_[ +] _[ σ]_[(] _**[w]**_[T] _**[x]**_[)] _**[x]**_[ +] _**[ Dx]**_[, which] along with the above equation completes the induction step. Finally, note that any matrix _**M** ∈_ R _[d][×][d]_ of rank _r∗_ =[�] _[T] t_ =1 _[r][t]_[can be written as][ �] _[T] t_ =1 _**[V]**[t]_[, for some choices of matrices] _**V** t ∈_ R _[d][×][d]_ of rank _rt_ .

## **B. Proof of Theorem 4.2**

By Eckart–Young–Mirsky theorem, _**U** r∗_ **Σ** _r∗_ _**V** r_[T] _∗_[is the best rank] _[ r][∗]_[approximation to] _**[ A]**[−]_ _**[I]**_[, by which we obtain][ ER] _[∗]_[(] _[C]_[res][) =] _∥_ **∆** _∥_[2] _F_[.]

We also have by definition,

**==> picture [348 x 24] intentionally omitted <==**

Recall the SVD of _**A** − I_ = _**U**_ **Σ** _**V**_[T] and consider the following decompositions:

**==> picture [273 x 25] intentionally omitted <==**

with _**U** r∗_ _**UU** r_[T] _∗r∗_[+] _,⊥_ _**[ U]** ,_ _**V**[r] ∗r[,] ∗[⊥] ,⊥_ _**[U]**_[ T] _r∈∗,⊥_ R[=] _[d][×]_ _**[ I]**_[(] _[d]_[.] _[−][r][∗]_[)] , and **Σ** _r∗,⊥_ a diagonal matrix of size _d − r∗_ . Since _**U**_ is unitary matrix, we have

We then note that for any choice of _α_ , _**A**_[˜] , we have

**==> picture [333 x 48] intentionally omitted <==**

Next, by taking _**A**_[˜] = _**U** r∗_ **Σ** _r∗_ _**V** r_[T] _∗_[+ (1] _[ −][α]_[)] _**[U]**[r] ∗_ _**[U]**_[ T] _r∗_[=] _**[ U]**[r][∗]_[[] **[Σ]** _[r][∗]_ _**[V]** r_[T] _∗_[+ (1] _[ −][α]_[)] _**[U]**_[ T] _r∗_[]][ as the rank-] _[r][∗]_[matrix, we obtain] _**A** − α_ _**I** −_ _**A**_[˜] = **∆** + (1 _− α_ ) _**U** r∗,⊥_ _**U** r_[T] _∗,⊥[.]_

Invoking the characterization (B.1), we arrive at

**==> picture [331 x 88] intentionally omitted <==**

2 where in the second equality, we used the fact that _**U** r∗,⊥_ is unitary and so ��� _**U** r∗,⊥_ _**U**_ T _r∗,⊥_ ��� _F_[=] _[ d][ −][r][∗]_[.][In addition, observed] that **∆** = _**A** −_ _**I** −_ _**U** r∗_ **Σ** _r∗_ _**V** r_[T] _∗_[=] _**[ U]**_ **[Σ]** _**[V]**_[T] _[ −]_ _**[U]**[r] ∗_ **[Σ]** _[r] ∗_ _**[V]** r_[T] _∗_[=] _**[ U]**[r] ∗[,][⊥]_ **[Σ]** _[r] ∗[,][⊥]_ _**[V]** r_[T] _∗,⊥[.]_

Therefore,

**==> picture [323 x 14] intentionally omitted <==**

and so trace( **∆**[T] _**U** r∗,⊥_ _**U** r_[T] _∗,⊥_[) = trace(] **[∆]**[T][) = trace(] **[∆]**[)][.][This completes the proof of the upper bound on][ ER] _[∗]_[(] _[C]_[GRN] _[−]_[v1][)][.] For ER _[∗]_ ( _C_ GRN _−_ v2) we have

**==> picture [358 x 104] intentionally omitted <==**

In addition, since GRN-v2 optimizes over a larger class of models (using diagonals instead of scale of identity), we have

**==> picture [376 x 22] intentionally omitted <==**

Combining (B.2) and (B.3) we obtain the claimed upper bound on ER _[∗]_ ( _C_ GRN _−_ v2).

We next proceed to bound ER _[∗]_ ( _C_ GRN _−_ v3). By Theorem 4.1, this quantity is at most the optimal objective value of the following optimization problem:

**==> picture [355 x 37] intentionally omitted <==**

We calculate the expectation in the objective over the gaussian vector _**x** ∼_ N(0 _,_ _**I**_ ). Define the shorthand _**B**_ := _**A** −_ _**D** −_ _**M**_ . We write

**==> picture [425 x 37] intentionally omitted <==**

These two expectations are characterized by our next lemma.

**Lemma B.1.** _Suppose that_ _**x** ∼_ N(0 _,_ _**I**_ ) _and let σ_ ( _z_ ) = _z_ **1** ( _z ≥_ 0) _be the ReLu function. Then,_

**==> picture [350 x 56] intentionally omitted <==**

Using the result of Lemma B.1 in (B.5) we obtain

**==> picture [433 x 31] intentionally omitted <==**

With this characterization, we next proceed to calculate the optimal objective value of (B.4). We start by minimizing over _**w**_ . To do this, we first fix _∥_ _**w** ∥ℓ_ 2 = _α_ , and optimize over the direction of _**w**_ , and then optimize over _α_ . Note that _**w**_[T] _**Bw**_ = _**w**_[T] ( _**B**_ + _**B**_[T] ) _/_ 2 _**w**_ . Since ( _**B**_ + _**B**_[T] ) _/_ 2 is symmetric, the maximum is achieved when _**w**_ s in the direction of its maximum eigenvalue. Define _λ_ _**[B]**_ max[as the maximum eigenvalue of][ (] _**[B]**_[ +] _**[ B]**_[T][)] _[/]_[2][.][We then have]

**==> picture [419 x 54] intentionally omitted <==**

We next continue with minimization over _**M** ,_ _**D**_ . Since we want to derive upper bound on the minimum objective value, we consider two choices of ( _**M** ,_ _**D**_ ) motivated by the analysis of GRN-v1 and GRN-v2.

- Choice 1: Similar to the analysis of GRN-v1, we set _**M**_ = _**U** r∗_ [ **Σ** _r∗_ _**V** r_[T] _∗_[+][(1] _[−][α]_[)] _**[U]**_[ T] _r∗_[]][and] _**[D]**_[=] _[α]_ _**[I]**_[with] _α_ = 1 + _d−_ 1 _r∗_[trace(] **[∆]**[)][.][With these choices we have]

**==> picture [167 x 85] intentionally omitted <==**

2 In addition, ��� _**U** r∗,⊥_ _**U**_ T _r∗,⊥_ ��� _F_[=] _[ d][ −][r][∗]_[and][ trace(] **[∆]**[T] _**[U]**[r][∗][,][⊥]_ _**[U]**_[ T] _r∗,⊥_[) = trace(] **[∆]**[)][.][Hence,]

**==> picture [155 x 22] intentionally omitted <==**

Furthermore, trace( _**B**_ ) = 0. Using these identities in (B.9) we obtain that the optimum objective value of (B.4) satisfies the following:

**==> picture [336 x 26] intentionally omitted <==**

with _ν_ max denoting the maximum eigenvalue of **[∆]**[+] 2 **[∆]**[T] _− d−_ 1 _r∗_[trace(] **[∆]**[)] _**[U]**[r][∗][,][⊥]_ _**[U]**_[ T] _r∗,⊥_[.]

• Choice 2: Similar to the analysis of GRN-v2, we set _**U** r∗_ **Σ** _r∗_ _**V** r_[T] _∗_[and] _**[ D]**_[=][ diag][(] _**[I]**_[+] **[ ∆]**[)][.][This way we have]

**==> picture [114 x 55] intentionally omitted <==**

Hence, _∥_ _**B** ∥_[2] _F_[=] _[ ∥]_ **[∆]** _[∥]_[2] _F[−]_[�] _[d] i_ =1[∆] _ii_[2][and][ trace(] _**[B]**_[) = 0][.][Using these identities in][ (][B.11][)][ we obtain that the optimum] objective value of (B.4) satisfies the following:

**==> picture [161 x 30] intentionally omitted <==**

˜ with _ν_ denoting the maximum eigenvalue of **[∆]**[+] 2 **[∆]**[T] _−_ diag( **∆** ).

Combining the bound from the two cases, we get

**==> picture [403 x 30] intentionally omitted <==**

which completes the proof of theorem.

_Proof._ (Lemma B.1) Since the distribution of _**x**_ is rotation invariant, without loss of generality we assume _**w**_ = _∥_ _**w** ∥ℓ_ 2 _**e**_ 1. We then have

**==> picture [358 x 15] intentionally omitted <==**

where _**x** ∼_ 1 = ( _x_ 2 _, . . . , xd_ ). We have

**==> picture [199 x 21] intentionally omitted <==**

Also, since _**x** ∼_ 1 is independent of _x_ 1 we have

**==> picture [223 x 21] intentionally omitted <==**

Combining the last three equations, we get

**==> picture [149 x 22] intentionally omitted <==**

To show the other claim, we note that

**==> picture [368 x 49] intentionally omitted <==**

Therefore in matrix form we have E[ _σ_ ( _x_ 1) _**xx**_[T] ] = ~~_√_~~ 12 _π_[(] _**[I]**_[+] _**[ e]**_[1] _**[e]**_[T] 1[)][.][This][completes][the][proof][of][(][B.7][)][as][by][rotation] invariance of distribution of _**x**_ we can assume _**w**_ = _∥_ _**w** ∥ℓ_ 2 _**e**_ 1.

## **C. Proof of Theorem 4.3**

Let _p_ := 2 _dr∗_ where we recall that _r∗_ =[�] _[T] ℓ_ =1 _[r][ℓ]_[.][Note][that] _[p]_[is][the][number][of][parameters][for][ResNet][with] _[T]_[layers] and ranks _rt_ for each layer _t_ . We will compare the test error of GRN-v1 and ResNet with _p_ number of parameters. This corresponds to a model in GRN-v1 with _T[′]_ layers such that 2 _d_[�] _[T] ℓ_ =1 _[ ′][r][ℓ]_[+] _[ T][ ′]_[(] _[T][ ′][−]_[1)] _[/]_[2][=] _[p]_[.][We][set][the][shorthand] _r∗[′]_[:=][ �] _[T] ℓ_ =1 _[ ′][r][ℓ]_[and let] _[ σ]_[1] _[≥][. . .][ ≥][σ][d]_[ be the singular values of] _**[ A]**[ −]_ _**[I]**_[.][By Theorem][ 4.2][ we have]

**==> picture [321 x 54] intentionally omitted <==**

Therefore, ER _[∗]_ ( _C_ GRN _−_ v1) _<_ ER _[∗]_ ( _C_ res) if the following holds:

**==> picture [313 x 32] intentionally omitted <==**

(Note that _r∗[′][< r][∗]_[since] _[ T][ ′][< T]_[).][However note that the left hand side of this condition is upper bounded by]

**==> picture [112 x 31] intentionally omitted <==**

Additionally, the right-hand side of the condition is lower bounded by

**==> picture [156 x 32] intentionally omitted <==**

So a sufficient condition for (C.1) is that

**==> picture [131 x 12] intentionally omitted <==**

Writing it in terms of _κ_ , we need

**==> picture [294 x 12] intentionally omitted <==**

Our next lemma gives alower bound on _r∗[′]_[.]

**Lemma C.1.** _Consider a standard Resnet model with collective rank r∗, and also a GRN-v1 model with collective rank r[′] , a GRN-v2 model with collective rank r[′′] , and a GRN-v3 model with collective rank r[′′′] , which have the same number of parameters as in the standard Resent model. We then have_

**==> picture [319 x 45] intentionally omitted <==**

Using Lemma C.1, condition C.2 is satisfied provided that

**==> picture [195 x 18] intentionally omitted <==**

Solving the above inequality for _r∗/d_ and after some algebraic calculation, we simplify the above inequality as follows:

**==> picture [144 x 19] intentionally omitted <==**

For GRN-v2, the argument goes along the same lines. Fixing number of parameters to _p_ , this corresponds to a model in GRN-v2 with _T[′′]_ layers such that 2 _d_[�] _[T] ℓ_ =1 _[ ′′][r][ℓ]_[+] _[ dT][ ′′]_[(] _[T][ ′′][−]_[1)] _[/]_[2][=] _[p]_[.][We][use][the][shorthand] _[r] ∗[′′]_[:=][�] _ℓ[T]_ =1 _[ ′′][r][ℓ]_[.][By] Theorem 4.2,

**==> picture [219 x 60] intentionally omitted <==**

Following the same argument as the one for GRN-v1 (replacing _r∗[′]_[with] _[r] ∗[′′]_[)][we][derive][that][GRN-v2][achieves][a][better] trade-off than standard ResNet, if

**==> picture [294 x 12] intentionally omitted <==**

(Note that this is analogous to (C.2) where _r∗[′]_[is replaced by] _[ r] ∗[′′]_[.)] Using Lemma C.1, condition C.6 is satisfied provided that

**==> picture [185 x 19] intentionally omitted <==**

By some algebraic calculation, this inequality can be simplified to

**==> picture [143 x 14] intentionally omitted <==**

We next proceed with the case of GRN-v3. By Theorem 4.2 we have

**==> picture [238 x 25] intentionally omitted <==**

with _ν_ max denoting the maximum eigenvalue of **[∆]**[+] 2 **[∆]**[T] _− d−_ 1 _r∗_[trace(] **[∆]**[)] _**[U]**[r] ∗[′′′][,][⊥]_ _**[U]**_[ T] _r∗[′′′][,][⊥]_[.][Rewriting this bound in terms of] eigenvalues we get

**==> picture [421 x 39] intentionally omitted <==**

Here we used the fact that the _**U** r∗[′′′][,][⊥]_[is the eigenspace of] **[ ∆]**[and its eigenvalues are] _[ σ][r] ∗[′′′]_[+1] _[≥][. . .][≥][σ][d]_[.][To lighten the] notation, we use the shorthand _σ_ := _σr∗′′′_[+1][and] _[ A]_[ :=][ �] _[d] i_ = _r∗[′′′]_[+1] _[ σ][i]_[.][Then in order to have][ ER] _[∗]_[(] _[C]_[GRN] _[−]_[v3][)] _[ <]_[ ER] _[∗]_[(] _[C]_[res][)][, it] suffices to have

**==> picture [225 x 30] intentionally omitted <==**

Note that the left-hand side is upper bounded by ( _r∗ − r∗[′′′]_[)] _[σ]_[.][In addition, this is quadratic inequality in] _[ A]_[.][Solving this] inequality for _A_ this corresponds to the following:

**==> picture [274 x 36] intentionally omitted <==**

Define the shorthand _ξ_ :=

_π_ ( _d_ +1)(1 _d−r∗[′′′]_[)][.][Then the above can be written as]

**==> picture [310 x 35] intentionally omitted <==**

We also have _A ≥_ ( _d − r∗[′′′]_[)] _[λ]_[min][and] _[ σ][≤][λ]_[max][, so] _[ A/σ][≥]_[(] _[d][ −][r] ∗[′′′]_[)] _[κ]_[.][Hence, the above condition holds if]

**==> picture [309 x 32] intentionally omitted <==**

It is easy to verify that the right-hand side is decreasing in _ξ_ . In addition, by definition of _ξ_ and since _r∗[′′′][< r][∗][≤][d]_[, we have] 1 _ξ ≥ ξ_ 0 := _π_ ( _d_[2] _−_ 1)[.][So a sufficient condition for (][C.8][) is]

**==> picture [312 x 32] intentionally omitted <==**

or equivalently,

**==> picture [150 x 46] intentionally omitted <==**

( _κ_ (1+ _ξ_ 0) _−ξ_ 0)[2] + _ξ_ 0 Define _η_ := � 1+ _ξ_ 0 . Rewriting the above inequality we need

**==> picture [295 x 12] intentionally omitted <==**

Next, by Lemma C.1, we have _r∗ − r∗[′′′][≤]_[(] _[√]_ 1 _._ 6 + _r∗ −_ 1)[2] , by which a sufficient condition for (C.10) is as follows:

**==> picture [193 x 13] intentionally omitted <==**

By some algebraic calculation, this inequality can be simplified to

**==> picture [145 x 13] intentionally omitted <==**

This completes the proof of the first item in the theorem statement. To prove the second item in the theorem statement, we write

**==> picture [401 x 100] intentionally omitted <==**

where in the last inequality we used Lemma C.1. A similar bound can be derived for GRN-v2, replacing _r∗[′]_[with] _[ r] ∗[′′]_[in the argument.][Specifically, we have]

**==> picture [391 x 45] intentionally omitted <==**

where in the last inequality we used the lower bound given for _r[′′]_ in Lemma C.1. For GRN-v3, we have

**==> picture [445 x 125] intentionally omitted <==**

where we recall the shorthand _A_ :=[�] _[d] i_ = _r∗[′′′]_[+1] _[ σ][i]_[ and] _[ σ]_[:=] _[ σ][r] ∗[′′′]_[+1][.][In the second inequality above, we used the fact that] _σr∗′′′_[+1] _[, . . . , σ][r][∗][≤][σ][r] ∗[′′′]_[+1][=] _[ σ]_[, along with the inequality][ (] _[a][ −][b]_[)][2] _[≥][a]_[2] _[/]_[2] _[ −][b]_[2][.]

Noting that _A ≥_ ( _d − r∗[′′′]_[)] _[λ]_[min][and] _[ σ][≤][λ]_[max][, we continue from the above chain of inequalities, as follows:]

**==> picture [409 x 110] intentionally omitted <==**

where we used Lemma C.1 in step ( _a_ ).

## **C.1. Proof of Lemma C.1**

A standard Resnet model with collective rank _r∗_ has 2 _dr∗_ number of parameters. A model in GRN-v1 with collective rank _r∗[′]_[has][ 2] _[dr] ∗[′]_[+] _[ T][ ′]_[(] _[T][ ′][ −]_[1)] _[/]_[2][ parameters.][Therefore, by assumption]

**==> picture [307 x 11] intentionally omitted <==**

Since each layer has rank at least one, we also have _r∗[′][≥][T][ ′]_[.][We define the shorthand] _[ξ]_[=] � _T[′]_ ( _T[′] −_ 1) (so _r[′] ≥ ξ_ ). Combining these two inequalities and writing them in terms of _ξ_ , we get

**==> picture [85 x 12] intentionally omitted <==**

Solving this inequality for _ξ_ we get _ξ ≤_ 2 _[√] d_[2] + _dr∗ −_ 2 _d_ . Using this bound in (C.11) we get

**==> picture [150 x 13] intentionally omitted <==**

Simplifying this inequality, we arrive at

**==> picture [123 x 13] intentionally omitted <==**

The upper bound _r∗[′][≤][r][∗]_[also follows simply from (][C.11][).] For GRN-v2 model we follow the same argument. A model in GRN-v2 with collective rank _r[′′]_ has 2 _dr∗[′′]_[+] _[ dT][ ′′]_[(] _[T][ ′′][ −]_[1)] _[/]_[2] parameters and so

**==> picture [304 x 12] intentionally omitted <==**

Since each layer has rank at least one, we also have _r∗[′′][≥][T][ ′′]_[.][Define the shorthand] _[ ξ][′]_[:=] � _T[′′]_ ( _T[′′] −_ 1). Combining the previous two equation, we get

**==> picture [79 x 13] intentionally omitted <==**

Solving this inequality for _ξ[′]_ we get _ξ[′] ≤_ 2 _[√]_ 1 + _r∗ −_ 2. Using this bound back in (C.12) we obtain

**==> picture [129 x 12] intentionally omitted <==**

This simplifies to _r∗ −_ ( _[√]_ 1 + _r∗ −_ 1)[2] _≤ r∗[′′]_[.]

We follow the same argument for GRN-v3. A model in GRN-v3 with collective rank _r[′′′]_ has 2 _dr∗[′′′]_[+] _[dT][ ′′′]_[(] _[T][ ′′′][−]_[1)] _[/]_[2+] _[dT][ ′′′]_ parameters and so

**==> picture [308 x 12] intentionally omitted <==**

Since each layer has rank at least one, we also have _r∗[′′′][≥][T][ ′′′]_[.][Hence,]

**==> picture [101 x 11] intentionally omitted <==**

Solving for _T[′′′]_ we get _T[′′′] ≤_ 1 _/_ 2( _[√]_ 25 + 16 _r∗ −_ 5). Using this bound in (C.13), we have

**==> picture [209 x 87] intentionally omitted <==**

The upper bound _r∗[′′′][< r][∗]_[follows easily from (][C.13][).]

## **D. Proof of Proposition 4.4**

The result follows from conditions (C.2), (C.6) and (C.10) which respectively provide sufficient conditions for GRN-v1, GRN-v2 and GRN-v3 to achieve smaller test error than a ResNet model, with the same number of parameters.

## **E. Proof of Proposition 4.5**

The proof is similar to the linear case by induction on _T_ . Note that for showing this direction ( _C_ base _, C_ res _, C_ GRN _−_ v1 _, C_ GRN _−_ v2 being a subset of the rank constrained functions) we only used the following two properties of the rank function which holds also for the Bottleneck rank: rank( _f ◦ g_ ) _≤_ min _{_ rank( _f_ ) _,_ rank( _g_ ) _}_ and rank( _f_ + _g_ ) _≤_ rank( _f_ ) + rank( _g_ ).

## **F. Training time versus perplexity on the LM1B dataset**

This appendix provides additional results on training time versus perplexity for DCA models. Figure 8 shows the training time-perplexity trade-off for 12, 24, and 36 layer transformer and DCA models trained on the LM1B dataset. The figure shows that DCA achieves a better perplexity for a given training time (except for the first few training steps of the 36-layer model). Thus, highlighting the training efficiency of DCA.

**==> picture [181 x 132] intentionally omitted <==**

Layers<br>17<br>12<br>24<br>36<br>16<br>Model<br>Transformer<br>DCA<br>15<br>14<br>0.0 0.5 1.0 1.5 2.0 2.5<br>Training time (s) 1e5<br>Perplexity<br>**----- End of picture text -----**<br>

_Figure 8._ Perplexity on LM1B pre-training versus the training time with transformer and DCA models of various depths.

## **G. Steps versus perplexity on the C4 dataset**

This appendix provides additional results on training steps versus perplexity for 8-DCA models. Figure 9 shows the training steps-perplexity trade-off for 75M, 179M, and 449M parameter transformer and 8-DCA models trained on the C4 dataset. The results show the improved model convergence and quality of DCA.

**==> picture [182 x 131] intentionally omitted <==**

35.0<br>Params Model<br>32.5<br>75M Transformer<br>30.0 179M DCA<br>449M<br>27.5<br>25.0<br>22.5<br>20.0<br>17.5<br>15.0<br>0 1 2 3 4 5<br>Training step 1e5<br>Perplexity<br>**----- End of picture text -----**<br>

_Figure 9._ Perplexity on C4 pre-training versus the number of steps with transformer and 8-DCA models of various sizes.

_Table 9._ Perplexity (PPL) on LM1B with and without the model inputs for 6-layer GRN-v3.

|METHOD|PPL|
|TRANSFORMER<br>GRN-V3 (LAST4 LAYERS)<br>GRN-V3 (MODEL INPUTS+ LAST3 LAYERS)|20.878<br>20.301<br>**20.227**|

## **H. Distribution of learned weights**

Figure 10 shows the distribution of the learned bias values for each GRN-v3 instance of a 30-layer model. The layers tend to weight the inputs and the last few layers the most and frequently assign a negative bias for the intermediate layers, indicating that the layers are filtered out as a result of the ReLU activation. In Table 9, we show that indeed the GRN-v3 model perplexity improves as the model inputs are included in addition to the last few layer outputs.

**==> picture [431 x 428] intentionally omitted <==**

Layer 2 Layer 3 Layer 4 Layer 5 Layer 6<br>0.3 0.3 0.3 0.3 0.3<br>0.2 0.2 0.2 0.2 0.2<br>0.1 0.1 0.1 0.1 0.1<br>0.0 0.0 0.0 0.0 0.0<br>0.1 0.1 0.1 0.1 0.1<br>0.2 0.2 0.2 0.2 0.2<br>0.00 0.25 0.50 0.75 1.00 0.0 0.5 1.0 1.5 2.0 0 1 2 3 0 1 2 3 4 0 2 4<br>Layer 7 Layer 8 Layer 9 Layer 10 Layer 11<br>0.3 0.3 0.3 0.3 0.3<br>0.2 0.2 0.2 0.2 0.2<br>0.1 0.1 0.1 0.1 0.1<br>0.0 0.0 0.0 0.0 0.0<br>0.1 0.1 0.1 0.1 0.1<br>0.2 0.2 0.2 0.2 0.2<br>0 2 4 6 0 2 4 6 0 2 4 6 8 0 2 4 6 8 0.0 2.5 5.0 7.5 10.0<br>Layer 12 Layer 13 Layer 14 Layer 15 Layer 16<br>0.3 0.3 0.3 0.3 0.3<br>0.2 0.2 0.2 0.2 0.2<br>0.1 0.1 0.1 0.1 0.1<br>0.0 0.0 0.0 0.0 0.0<br>0.1 0.1 0.1 0.1 0.1<br>0.2 0.2 0.2 0.2 0.2<br>0.0 2.5 5.0 7.5 10.0 0 5 10 0 5 10 0 5 10 0 5 10 15<br>Layer 17 Layer 18 Layer 19 Layer 20 Layer 21<br>0.3 0.3 0.3 0.3 0.3<br>0.2 0.2 0.2 0.2 0.2<br>0.1 0.1 0.1 0.1 0.1<br>0.0 0.0 0.0 0.0 0.0<br>0.1 0.1 0.1 0.1 0.1<br>0.2 0.2 0.2 0.2 0.2<br>0 5 10 15 0 5 10 15 0 5 10 15 0 5 10 15 0 5 10 15 20<br>Layer 22 Layer 23 Layer 24 Layer 25 Layer 26<br>0.3 0.3 0.3 0.3 0.3<br>0.2 0.2 0.2 0.2 0.2<br>0.1 0.1 0.1 0.1 0.1<br>0.0 0.0 0.0 0.0 0.0<br>0.1 0.1 0.1 0.1 0.1<br>0.2 0.2 0.2 0.2 0.2<br>0 5 10 15 20 0 5 10 15 20 0 10 20 0 10 20 0 10 20<br>Layer 27 Layer 28 Layer 29 Layer 30 Outputs<br>0.3 0.3 0.3 0.3 0.3<br>0.2 0.2 0.2 0.2 0.2<br>0.1 0.1 0.1 0.1 0.1<br>0.0 0.0 0.0 0.0 0.0<br>0.1 0.1 0.1 0.1 0.1<br>0.2 0.2 0.2 0.2 0.2<br>0 10 20 0 10 20 0 10 20 0 10 20 30 0 10 20 30<br>Layer Layer Layer Layer Layer<br>Bias value<br>Bias value<br>Bias value<br>Bias value<br>Bias value<br>Bias value<br>**----- End of picture text -----**<br>

_Figure 10._ Distribution of learned bias values on LM1B pre-training with a 30 layer GRN-v3 transformer model. The solid line indicates the median value and the shaded area represents the 90th percentile.
