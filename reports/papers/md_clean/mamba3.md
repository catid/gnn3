# Mamba-3: Improved Sequence Modeling using State Space Principles

Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and Albert Gu

Carnegie Mellon University, Princeton University, Together AI, and Cartesia AI

## **Abstract**

Scaling inference-time compute has emerged as an important driver of LLM performance, making inference efficiency a central focus of model design alongside model quality. While the current Transformer-based models deliver strong model quality, their quadratic compute and linear memory make inference expensive. This has spurred the development of sub-quadratic models with reduced linear compute and constant memory requirements. However, many recent linear models trade off model quality and capability for algorithmic efficiency, failing on tasks such as state tracking. Moreover, their theoretically linear inference remains hardware-inefficient in practice. Guided by an inference-first perspective, we introduce three core methodological improvements inspired by the state space model (SSM) viewpoint of linear models. We combine: (1) a more expressive recurrence derived from SSM discretization, (2) a complex-valued state update rule that enables richer state tracking, and (3) a multi-input, multi-output (MIMO) formulation for better model performance without increasing decode latency. Together with architectural refinements, our **Mamba-3** model achieves significant gains across retrieval, state-tracking, and downstream language modeling tasks. At the 1.5B scale, Mamba-3 improves average downstream accuracy by 0.6 percentage points compared to the next best model (Gated DeltaNet), with Mamba-3’s MIMO variant further improving accuracy by another 1.2 points for a total 1.8 point gain. Across state-size experiments, Mamba-3 achieves comparable perplexity to Mamba-2 despite using half of its predecessor’s state size. Our evaluations demonstrate Mamba-3’s ability to advance the performance-efficiency Pareto frontier.

## **1 Introduction**

Test-time compute has emerged as a key driver of progress in LLMs, with techniques like chain-of-thought reasoning and iterative refinement demonstrating that inference-time scaling can unlock new capabilities (Snell et al. 2024; Wu et al. 2025). The rapid rise of parallel, agentic workflows has only intensified the need for efficient inference and deployment of such models (Anthropic 2026; OpenAI 2026). This paradigm shift makes inference efficiency (Kwon et al. 2023; Li et al. 2024) paramount, as the practical impact of AI systems now depends critically on their ability to perform large-scale inference during deployment. Model architecture design plays a fundamental role in determining inference efficiency, as architectural choices directly dictate the computational and memory requirements during generation. While Transformerbased models (Vaswani et al. 2017) are the current industry standard, they are fundamentally bottlenecked by linearly increasing memory demands through the KV cache and quadratically increasing compute requirements through the selfattention mechanism. These drawbacks have motivated recent lines of work on sub-quadratic models, e.g., state space models (SSMs) and linear attention, which retain constant memory and linear compute while attaining comparable or better performance than their Transformer counterparts. These models have made it into the mainstream, with layers such as Mamba-2 (Dao and Gu 2024) and Gated DeltaNet (GDN) (Schlag, Irie, and Schmidhuber 2021; S. Yang, B. Wang, Y. Zhang, et al. 2025) recently incorporated into large-scale hybrid models that match the performance of pure Transformer alternatives with much higher efficiency (Kimi Team et al. 2025; NVIDIA et al. 2025; Tencent Hunyuan Team et al. 2025; A. Yang et al. 2025).

Despite the success of linear models, significant progress remains in improving their performance, in particular on advancing the Pareto frontier between model quality and inference efficiency. For example, Mamba-2 was developed to improve

> ∗Equal contribution.

> †Equal advising.

training speed and simplicity over Mamba-1 (Gu and Dao 2024), by sacrificing some expressivity and thus performing worse for inference-matched models. In addition, they have been shown to lack certain capabilities, such as poor statetracking abilities, e.g., simply determining parity of bit sequences (Grazzi, Siems, Zela, et al. 2025; Sarrof, Veitsman, and Hahn 2024). Finally, despite these sub-quadratic models being prized for theoretically efficient inference and thus their widespread adoption, their inference algorithms are not hardware efficient. In particular, because these algorithms were developed from a training perspective, their decoding phase has low arithmetic intensity (the ratio of FLOPs to memory traffic), resulting in large portions of hardware remaining idle.

To develop more performant models from an inference-first paradigm, we introduce three core methodological changes on top of Mamba-2, influenced by an SSM-centric viewpoint of sub-quadratic models.

**Exponential-Trapezoidal Discretization.** We provide a simple technique for discretizing time-varying, selective SSMs. Through our framework, we can derive several new discretization methods. One of our instantiations, referred to as “exponential-Euler,” formalizes Mamba-1 and Mamba-2’s heuristic discretization that previously lacked theoretical justification. Our new “exponential-trapezoidal” instantiation is a more expressive generalization of “exponential-Euler,” where the recurrence can be expanded to reveal an implicit convolution applied on the SSM input. Combined with explicit _𝐵,𝐶_ bias terms, Mamba-3 can empirically replace the short causal convolution in language model architectures, which was previously hypothesized to be essential for recurrent models.

**Complex-valued State Space Model.** By viewing the underlying SSM of Mamba-3 as complex-valued, we enable a more expressive state update than Mamba-2’s. This change in update rule, designed to be lightweight for training and inference, overcomes the lack of state-tracking ability common in many current linear models. We show that our complexvalued update rule is equivalent to a data-dependent rotary embedding and can be efficiently computed (Su et al. 2023), and empirically demonstrate its ability to solve synthetic tasks outside the capabilities of prior linear models.

**Multi-Input, Multi-Output (MIMO) SSM.** To improve FLOP efficiency during decoding, we switch from an outerproduct–based state update to a matrix-multiplication–based state update. From the view of the signal processing foundations of SSMs, such a transition exactly coincides with the generalization from a single-input single-output (SISO) sequence dynamics to a multiple-input multiple-output (MIMO) one. Here, we find that MIMO is particularly suitable for inference, as the extra expressivity enables more computation during the memory-bound state update during decoding, without increasing the state size and compromising speed.

Put together, these improvements form the core of our **Mamba-3** layer. Methodologically, we note that these all arise naturally from an SSM-centric perspective but are not immediate from other popular viewpoints of modern linear layers such as linear attention or test-time regression; we discuss these connections further in Section 5. Empirically, we validate our new model’s abilities and capabilities on a suite of synthetic state-tracking and language-modeling tasks.

- **Better Quality.** At 1.5B scale, Mamba-3 (MIMO) improves downstream language modeling accuracy by **+2.2** over Transformers, **+1.9 points** over Mamba-2, and **+1.8** over GDN, while Mamba-3 (SISO) improves over the next best model, GDN, by **+0.6** points. Furthermore, across state size experiments, Mamba-3 (MIMO) with state size 64 matches the perplexity of Mamba-2 with state size 128, effectively achieving the **same language modeling performance with half the latency** .

- **New Capabilities.** Mamba-3’s complexification of the SSM state enables it to **solve synthetic state-tracking tasks that Mamba-2 cannot** . We empirically demonstrate that the efficient RoPE-like calculation is able to near perfectly solve arithmetic tasks, while Mamba-3 without RoPE and Mamba-2 perform no better than random guessing.

- **Inference Efficiency.** Mamba-3 (MIMO) improves hardware utilization. It increases decoding FLOPs by up to **4** × relative to Mamba-2 at fixed state size, while maintaining **similar wall-clock decode latency** , and simultaneously improving perplexity and downstream performance. We release fast training and inference kernels for Mamba-3.[1]

Mamba-3 (SISO) improves quality and capability over prior linear models, and Mamba-3 (MIMO) further improves performance over Mamba-3 (SISO) and other strong baselines while matching inference speed with Mamba-2. Both of our Mamba-3 variants advance the performance-latency Pareto frontier through their strong modeling capabilities and hardware-efficient design.

> 1https://github.com/state-spaces/mamba.

## **2 Preliminaries**

## **2.1 Notation**

Scalars are denoted by plain-text letters (e.g., _𝑥,𝑦_ ). Tensors, including vectors and matrices, are denoted by bold letters (e.g., _**h** ,_ _**C**_ ). The shape of the tensor can be inferred from the context. We denote the input sequence length as _𝑇_ , the model dimension as _𝐷_ , and the SSM state size as _𝑁_ . For time indices, we use subscripts (e.g., _𝑥𝑡_ for the input at time _𝑡_ ). The Hadamard product between two tensors is denoted by ⊙. For a vector _**v**_ ∈ R _[𝑑]_ , we denote Diag( _**v**_ ) ∈ R _[𝑑]_[×] _[𝑑]_ as the diagonal matrix with the vector _**v**_ as the diagonal, and for products of scalars across time steps, we use the notation _𝛼𝑡_ ··· _𝑠_ = _𝛼𝑡_[×] : _𝑠_[=][ �] _𝑖[𝑡]_ = _𝑠[𝛼][𝑖]_[.]

## **2.2 SSM Preliminaries**

State Space Models (SSMs) describe continuous-time linear dynamics via

**==> picture [295 x 12] intentionally omitted <==**

where _**h**_ ( _𝑡_ ) ∈ R _[𝑁]_ is the hidden state, _𝑥_ ( _𝑡_ ) ∈ R the input, and _**A**_ ( _𝑡_ ) ∈ R _[𝑁]_[×] _[𝑁]_ , _**B**_ ( _𝑡_ ) _,_ _**C**_ ( _𝑡_ ) ∈ R _[𝑁]_ . We will occasionally refer to _**A**_ ( _𝑡_ ) as the _state-transition_ and _**B**_ ( _𝑡_ ) _𝑥_ ( _𝑡_ ) as the _state-input_ ; this also extends to their discretized counterparts. For discrete sequences with step size Δ _𝑡_ , Mamba-1 and Mamba-2 _discretized_ the system to the following recurrence

**==> picture [269 x 13] intentionally omitted <==**

**Mamba-2’s Parameterization.** The core of the Mamba-2 layer (Dao and Gu 2024) is a _data-dependent_ and hardwareefficient SSM. Both the state-transition and state-input are made data-dependent through the projection of Δ _𝑡_ ∈ R _>_ 0 and _**B** ,_ _**C**_ ∈ R _[𝑁]_ from the current token. By parameterizing the state-transition _**A** 𝑡_ as a scalar times identity ( _**A** 𝑡_ = _𝐴𝑡_ _**I** 𝑁_ × _𝑁_ , where _𝐴𝑡_ ∈ R _<_ 0), the SSM recurrence can be efficiently computed with the matrix multiplication tensor cores of GPUs. Defining _𝛼𝑡_ � _𝑒_[Δ] _[𝑡][𝐴][𝑡]_ ∈(0 _,_ 1) and _𝛾𝑡_ � Δ _𝑡_ , the update becomes

**==> picture [327 x 12] intentionally omitted <==**

The data-dependent state-transition _𝛼𝑡_ controls the memory horizon of each SSM within the layer. Δ _𝑡_ in particular modulates both the state-transition and state-input: a larger Δ _𝑡_ forgets faster and up-weights the current token more strongly, while a smaller Δ _𝑡_ retains the hidden state with minimal contributions from the current token.

_Remark_ 1 _._ In Mamba-2, _𝐴𝑡_ is data-independent, since the overall discrete transition _𝛼𝑡_ � _𝑒_[Δ] _[𝑡][𝐴][𝑡]_ is data-dependent through Δ _𝑡_ . In Mamba-3, we empirically found that data-dependent _𝐴𝑡_ has similar performance to data-independent _𝐴𝑡_ , and chose the former as a default for consistency so that all SSM parameters are data-dependent.

## **2.3 Structured Masked Representation and State Space Duality**

Mamba-2 showed that a large class of SSMs admit a _matrix_ form that vectorizes the time-step recurrence. Through the state space duality (SSD) framework, recurrent SSMs can be represented within a parallel form that incorporates an element-wise mask to model the state-transition decay.

SSD provides a general framework for a duality between linear recurrence and parallelizable (matrix-multiplication-based) computational forms

**==> picture [286 x 12] intentionally omitted <==**

where _**L**_ ∈ R _[𝑇]_[×] _[𝑇]_ is a structured mask, _**B** ,_ _**C**_ ∈ R _[𝑇]_[×] _[𝑁]_ , _**X**_ ∈ R _[𝑇]_[×] _[𝐷]_ are the inputs to the SSM and _**Y**_ ∈ R _[𝑇]_[×] _[𝐷]_ is its output. Different structures on _**L**_ give rise to various instantiations of SSD.

Equation (2) also draws a general connection between recurrence and attention, by setting _**Q**_ � _**C**_ , _**K**_ � _**B**_ , _**V**_ � _**X**_ and viewing _**L**_ as a data-dependent mask. In fact, the simplest case of SSD is (causal) linear attention (Katharopoulos et al. 2020), where _**L**_ is the causal triangular mask.

**==> picture [491 x 114] intentionally omitted <==**

𝑡!<br>≈ 𝐵 𝜏𝑥 𝜏𝑑𝜏<br>1 𝛾' !𝑒 [!][!][(#][!][$%) ]<br> 𝑡!"#<br>ℳ [=] 𝛼!:! × 1 𝛽! 𝛾!<br>𝛼%:! × 𝛼%:% × 1 𝛽% 𝛾%<br>𝛼&:! × 𝛼&:% × 𝛼&:& × 1 𝛽& 𝛾&<br>𝑡!"# 𝑡! 𝑡!"# 𝑡!<br>**----- End of picture text -----**<br>

Figure 1: **Left:** The structured mask induced by the exponential-trapezoidal rule (Section 3.1) is a product of the decay and two-band convolutional mask. **Right:** Euler (hold endpoint) versus Trapezoidal (average endpoints) integral approximation.

Mamba-2 is a generalization where

**==> picture [318 x 55] intentionally omitted <==**

composed of terms _𝛼𝑡,𝛾𝑡_ from equation (1).[2]

In Section 3.1.3, we show that Mamba-3 is a generalization of Mamba-2 with a more expressive _**L**_ , and hence also an instance of SSD.

## **3 Methodology**

We introduce Mamba-3, a state space model with three new innovations: “exponential-trapezoidal” discretization for more expressive dynamics (Section 3.1), complex-valued state spaces for state tracking (Section 3.2), and multi-input multi-output (MIMO) to improve modeling power and inference-time hardware utilization (Section 3.3). These advances address the quality, capability, and efficiency limitations of current sub-quadratic architectures. We combine these together into an updated Mamba architecture block in Section 3.4.

## **3.1 Exponential-Trapezoidal Discretization**

Structured SSMs are naturally defined as continuous-time dynamical systems that map input functions, _𝑥_ ( _𝑡_ ) ∈ R, to output functions, _𝑦_ ( _𝑡_ ) ∈ R, for time _𝑡 >_ 0. The underlying continuous state space system is defined by a first-order ordinary differential equation (ODE) for the state _**h**_[�] ( _𝑡_ ) and an algebraic equation for the output _𝑦_ ( _𝑡_ ). In sequence modeling, however, the data is only observed at discrete time steps, which then requires applying a _discretization step_ to the SSM to transform its continuous-time dynamics into a discrete recurrence.

Discretization methods are well-studied in classical control theory with several canonical formulas used in earlier SSM works in deep learning (Gu, Goel, and Ré 2022; Gu, Gupta, et al. 2022; Smith, Warrington, and Linderman 2023). These mechanisms were traditionally stated and applied to linear-time invariant (LTI) systems, and their derivations do not directly apply to linear-time varying (LTV) systems. Additionally, while Mamba-1 adapted the zero-order hold (ZOH) method to LTV systems without proof, the complexity associated with selective SSMs prompted the use of an additional heuristic approximation that lacked theoretical justification and did not correspond to any established discretization technique. In the following subsection, we formalize the previous heuristics used in current LTV SSMs through our discretization framework and utilize it to propose a more expressive discretization scheme.

> 2In the original Mamba-2 paper, _𝛾_ does not appear because it is viewed as folded into the _**B**_ term. In this paper, _**B** 𝑡_ represents the continuous parameter, whereas in Mamba-2, _**B** 𝑡_ represents the discretized parameter which is equivalent to _𝛾𝑡_ _**B** 𝑡_ .

> 3While the Mamba-1 paper reports ZOH discretization, the implementation follows https://github.com/state-spaces/mamba/issues/129.

Table 1: Table of canonical linear-time invariant discretizations (top) and custom linear-time varying discretizations derived from our exponential-adjusted framework (bottom), along with their appearance in structured SSMs used in deep learning. Our theory formalizes the prior Mamba discretization as exponential-Euler and extends it with the more expressive exponential-trapezoidal method.� The generalized discretization framework converts a continuous SSM _**h**_ ( _𝑡_ ) = _**A**_ ( _𝑡_ ) _**h**_ ( _𝑡_ ) + _**B**_ ( _𝑡_ ) _𝑥_ ( _𝑡_ ) into the discrete recurrence _**h** 𝑡_ = _𝛼𝑡_ _**h** 𝑡_ −1 + _𝛽𝑡_ _**B** 𝑡_ −1 _𝑥𝑡_ −1 + _𝛾𝑡_ _**B** 𝑡𝑥𝑡_ , where various discretization methods yield different formulas for _𝛼𝑡, 𝛽𝑡,𝛾𝑡_ .

|**Discretization Method**|_𝛼𝑡_||_𝛽𝑡_|_𝛾𝑡_||**Appearance**|
|---|---|---|---|---|---|---|
|**Forward Euler**|_𝐼_+Δ_𝐴_||—|Δ||—|
|**Backward Euler**|(_𝐼_−Δ_𝐴_)−1||—|(_𝐼_−Δ_𝐴_)−1 Δ||—|
|**Trapezoidal**|(_𝐼_−Δ<br>2 _𝐴_)−1(_𝐼_+ Δ<br>2 _𝐴_)||—|(_𝐼_−Δ<br>2 _𝐴_)−1Δ||S4|
|**Zero-Order Hold**|exp(Δ_𝐴_)||—|_𝐴_−1|(exp(Δ_𝐴_) −_𝐼_)|S4D, S5|
|**Zero-Order Hold**|exp(Δ_𝑡𝐴𝑡_)||—|_𝐴_−1<br>_𝑡_|(exp(Δ_𝑡𝐴𝑡_) −_𝐼_)||
|**Exponential-Euler**|exp(Δ_𝑡𝐴𝑡_)||—|Δ_𝑡_||Mamba-1, -23|
|**Exponential-Trapezoidal**|exp(Δ_𝑡𝐴𝑡_)||(1−_𝜆𝑡_)Δ_𝑡_exp(Δ_𝑡𝐴𝑡_)|_𝜆𝑡_Δ_𝑡_||Mamba-3|

## **3.1.1 Overview of Exponential-Adjusted Discretization**

We introduce a simple derivation that leads to a class of new discretization methods for LTV state space models. The method can be instantiated in various ways; we show that one instantiation results in the heuristic used in Mamba-1/2, thereby theoretically justifying it (exponential-Euler). We also introduce a more powerful discretization (exponentialtrapezoidal) used in Mamba-3.

The high-level intuition of our derivation originates from the closed form solution _𝑥_ ( _𝑡_ ) = _𝑒[𝑡𝐴] 𝑥_ (0) of a simple linear ODE _𝑥_[′] ( _𝑡_ ) = _𝐴𝑥_ ( _𝑡_ ), which discretizes to _𝑥𝑡_ +1 = _𝑒_[Δ] _[𝐴] 𝑥𝑡_ . In this example, the exponential dominates the dynamics of the underlying first-order ODE, resulting in imprecise approximations when using low-order methods without significantly constraining Δ. Thus, we analyze the dynamics of the _exponential-adjusted_ system _𝑒_[−] _[𝐴𝑡] 𝑥_ ( _𝑡_ ). The adjusted system yields a discrete recurrent form where the state-transition and the state-input integrals are approximated separately—the statetransition integral is approximated by a right-hand approximation, i.e. _𝐴_ ( _𝑠_ ) � _𝐴_ ( _𝜏𝑡_ ) for all _𝑠_ ∈[ _𝜏𝑡_ −1 _,𝜏𝑡_ ], yielding,

**==> picture [295 x 72] intentionally omitted <==**

which serves as the foundation for further discretization techniques for the state-input integral. The full derivation is detailed in Proposition 5.

**ZOH.** The classical zero-order hold discretization method can be derived from the foundation above with a specific approximation of the right-hand side integral. By treating _𝐴𝑡,_ _**B**_ ( _𝜏_ ) _,𝑥_ ( _𝜏_ ) as constants over the interval [ _𝜏𝑡_ −1 _,𝜏𝑡_ ] where the values are fixed to the right endpoint _𝜏𝑡_ , the integral results in _𝐴𝑡_[−][1] (exp(Δ _𝑡𝐴𝑡_ ) − _𝐼_ ) _**B** 𝑡𝑥𝑡_ .

We note that this formally proves that the classical ZOH formula for LTI systems applies to LTV by naively replacing the parameters _𝐴, 𝐵,_ Δ with their time-varying ones.

**Exponential-Euler (Mamba-1/-2).** While Mamba-1 stated to use the time-varying ZOH formula above, Mamba-1 and Mamba-2 actually used an additional approximation in the released implementation. This discretization can be recovered by approximating the state-input integral with _Euler’s rule_ (Süli and Mayers 2003) and holding the (right) endpoint constant

throughout the interval (Fig. 1)

**==> picture [340 x 29] intentionally omitted <==**

We call equation (4) the _exponential-Euler_ discretization method, stemming from the exponential integration followed by Euler approximation. This derivation justifies the formulas used in Mamba-1/-2’s implementation.

**Exponential-Trapezoidal (Mamba-3).** However, Euler’s rule provides only a first-order approximation of the stateinput integral and its local truncation error scales as _𝑂_ (Δ _𝑡_[2][)][. In contrast, we introduce a] _[ generalized trapezoidal rule]_[, which] provides a second-order accurate approximation of the integral, offering improved accuracy over Euler’s rule. Specifically, it approximates the integral with a _data-dependent, convex combination of both interval endpoints_ . This generalization extends the classical trapezoidal rule (Süli and Mayers 2003), which simply averages the interval endpoints (Figure 1).

**Proposition 1** (Exponential-Trapezoidal Discretization) **.** _Approximating the state-input integral in equation (16) by the general trapezoidal rule yields the recurrence,_

**==> picture [365 x 27] intentionally omitted <==**

_where 𝜆𝑡_ ∈[0 _,_ 1] _is a data-dependent scalar, 𝛼𝑡_ � _𝑒_[Δ] _[𝑡][𝐴][𝑡] , 𝛽𝑡_ � (1 − _𝜆𝑡_ )Δ _𝑡𝑒_[Δ] _[𝑡][𝐴][𝑡] , 𝛾𝑡_ � _𝜆𝑡_ Δ _𝑡 ._

_Remark_ 2 (Expressivity) _._ The exponential-trapezoidal rule is a generalization of (a) the classical trapezoid rule, which is recovered when _𝜆𝑡_ =[1] 2[, and (b) Mamba-2’s Euler’s rule, which is recovered when] _[ 𝜆][𝑡]_[=][ 1.]

_Remark_ 3 (Error Rate) _._ This is a second-order discretization of the state-input integral and its error scales as _𝑂_ (Δ _𝑡_[3][)][ under] standard stability assumptions, provided that the trapezoidal parameter satisfies _𝜆𝑡_ =[1] 2[+] _[ 𝑂]_[(][Δ] _[𝑡]_[)][.][However, our ablations] indicate that not enforcing this constraint is better for empirical performance. See Appendix A.2 and A.3 for details.

Our new discretization framework and the two instantiations, exponential-Euler and exponential-trapezoidal, are, to the best of our knowledge, novel for structured SSMs used in deep learning. Table 1 compares and summarizes canonical and commonly used discretization schemes for state space models.

## **3.1.2 Exponential-Trapezoidal Recurrence as an Implicit Convolution**

Our generalized exponential-trapezoidal discretization is equivalent to applying a _data-dependent_ convolution of size two on the state-input to the SSM. In particular, a normal SSM in recurrent form materializes the state-input _**v** 𝑡_ = _**B** 𝑡𝑥𝑡_ , then computes a linear recurrence _**h** 𝑡_ = _𝛼𝑡_ _**h** 𝑡_ −1 + _𝛾𝑡_ _**v** 𝑡_ . In equation (6) we instead first apply a width-2 convolution on _**v** 𝑡_ (weighted by _𝛽,𝛾_ ) before passing it into the linear recurrence.

_Remark_ 4 (Convolution Differences) _._ There is a distinct difference between the “convolution” induced by exponentialtrapezoidal discretization and the standard short convolutions used by sequence models such as Mamba and GDN. Standard short convolutions are independent operations applied on _𝑥𝑡_ (and often _**B** 𝑡,_ _**C** 𝑡_ ) _outside_ the core recurrence, while our new discretization can be interpreted as a convolution on the _state-input_ _**B** 𝑡𝑥𝑡 within_ the core recurrence.

## **3.1.3 Parallel Representation of Exponential-Trapezoidal Recurrence**

Our new recurrence can be instantiated as a case of SSD and has a corresponding parallel form to equation (2). Expanding the state recurrence from _**h**_ 0 = _𝛾_ 0 _**B**_ 0 _𝑥_ 0 results in _**h** 𝑇_ = _𝛼𝑇_ ···2( _𝛾_ 0 _𝛼_ 1 + _𝛽_ 1) _**B**_ 0 _𝑥_ 0 +· · ·+ _𝛾𝑇_ _**B** 𝑇 𝑥𝑇_ , where the SSM output is _**y** 𝑇_ = _𝛼𝑇_ ···2( _𝛾_ 0 _𝛼_ 1 + _𝛽_ 1) _**C** 𝑇_[⊤] _**[B]**_[0] _[𝑥]_[0][ + · · · +] _[𝛾][𝑇]_ _**[C]** 𝑇_[⊤] _**[B]**[𝑇][𝑥][𝑇]_[. Unrolling these rows shows that the mask induced by the trapezoidal update] is no longer a fixed averaging of endpoints (as in the classical trapezoid rule), but a _data-dependent convex combination_ of the two interval endpoints.

Under the SSD framework (2) with parallel form _**Y**_ = ( _**L**_ ⊙ _**CB**_[⊤] ) _**X**_ , Mamba-3 corresponds to a mask _**L**_ whose structure

is a 1-semiseparable matrix composed with a 2-band matrix:[4]

**==> picture [456 x 67] intentionally omitted <==**

This parallel formulation enables the hardware-efficient matmul-focused calculation of the SSM output for training.

We note that the convolutional connection of Mamba-3 can also be seen through this parallel dual form, where multiplication by the 2-band matrix in equation (7) represents convolution with weights _𝛽,𝛾_ . In Appendix A.1, we use the SSD tensor contraction machinery to prove that the parallel form is equivalent to a vanilla SSM with a state-input convolution.

_Remark_ 5 _._ The structured mask of Mamba-3 can be viewed as generalizing Mamba-2, which instead of the 2-band matrix has a diagonal matrix with _𝛾𝑡_ only (3).

## **3.2 Complex-Valued SSMs**

Modern SSMs are designed with efficiency as the central goal, motivated by the need to scale to larger models and longer sequences. For instance, successive architectures have progressively simplified the state-transition matrix: S4 (Gu, Goel, and Ré 2022) used complex-valued Normal Plus Low Rank (NPLR) matrices, Mamba (Gu and Dao 2024) reduced this to a diagonal of reals, and Mamba-2 (Dao and Gu 2024) further simplified it to a single scaled identity matrix. Although these simplifications largely maintain language modeling performance, recent works (Grazzi, Siems, Zela, et al. 2025; Merrill, Petty, and Sabharwal 2025; Sarrof, Veitsman, and Hahn 2024) have shown that the restriction to real, non-negative eigenvalue transitions degrades the capabilities of the model on simple state-tracking tasks—here referring primarily to the solvable-group regime (TC[0] ) such as parity—which can be solved by a one-layer LSTM. This limitation, formalized in Theorem 1 of (Grazzi, Siems, Schrodi, et al. 2024), arises from restricting the eigenvalues of the transition matrix to real numbers, which cannot represent “rotational” hidden state dynamics. For instance, consider the parity function on binary inputs {0 _,_ 1}, defined as[�] _𝑡[𝑥] 𝑡_[mod 2.][This task can be performed using update:] _**[h]** 𝑡_[=] _**[R]**_[(] _[𝜋𝑥] 𝑡_[)] _**[h]** 𝑡_ −1[, where] _**[ R]**_[(·)][is a 2-D] rotation matrix. Such rotational dynamics cannot be expressed with real eigenvalues.

## **3.2.1 Complex SSM with Exponential-Euler Discretization**

To recover this capability, we begin with complex SSMs (8), which _are_ capable of representing state-tracking dynamics. We show that, under discretization (Proposition 5), complex SSMs can be formulated as real SSMs with a _block-diagonal transition matrix composed of_ 2 × 2 _rotation matrices_ (Proposition 2). We then show that this is equivalent to applying _data-dependent rotary embeddings_ on both the input and output projections _**B** ,_ _**C**_ respectively. This result establishes a theoretical connection between complex SSMs and data-dependent RoPE embeddings (Proposition 3). Finally, the “RoPE trick” used in Su et al. (2023) allows for an efficient implementation of complex-valued state-transition matrices with minimal computational overhead compared to real-valued SSMs.

**Proposition 2** (Complex-to-Real SSM Equivalence) **.** _Consider a complex-valued SSM_

**==> picture [355 x 35] intentionally omitted <==**

_where_ _**h**_ ( _𝑡_ ) ∈ C _[𝑁]_[/][2] _,_ _**θ**_ ( _𝑡_ ) _,_ _**B**_ ( _𝑡_ ) _,_ _**B**_[ˆ] ( _𝑡_ ) _,_ _**C**_ ( _𝑡_ ) _,_ _**C**_[ˆ] ( _𝑡_ ) ∈ R _[𝑁]_[/][2] _, and 𝑥_ ( _𝑡_ ) _,𝐴_ ( _𝑡_ ) ∈ R _. Under exponential-Euler discretization, this system is equivalent to a real-valued SSM_

**==> picture [305 x 27] intentionally omitted <==**

> 4Incidentally, this is a special case of a 2-semiseparable matrix.

_with state_ _**h** 𝑡_ ∈ R _[𝑁] , projections_

**==> picture [178 x 25] intentionally omitted <==**

_and a transition matrix_

**==> picture [307 x 25] intentionally omitted <==**

## The proof is given in Appendix B.1.

Proposition 2 shows that the discretized complex SSM of state dimension _𝑁_ /2 has an equivalent real SSM with doubled state dimension ( _𝑁_ ), and its transition matrix is a scalar decayed block-diagonal matrix of 2 × 2 data-dependent rotation matrices ( _𝑒_[Δ] _[𝑡][𝐴][𝑡]_ _**R** 𝑡_ ).

**Proposition 3** (Complex SSM, Data-Dependent RoPE Equivalence) **.** _Under the notation established in Proposition 2, consider the real SSM defined in equation (9) unrolled for 𝑇 time-steps. The output of the above SSM is equivalent to that of a vanilla scalar transition matrix-based SSM (4) with a data-dependent rotary embedding applied on the_ _**B** ,_ _**C** components of the SSM, as defined by:_

**==> picture [398 x 32] intentionally omitted <==**

_where the matrix product represents right matrix multiplication, e.g.,_[�] _𝑖_[1] =0 _**[R]**[𝑖]_[=] _**[ R]**_[0] _**[R]**_[1] _[. We refer to the usage of a transformed] real-valued SSM to compute the complex SSM as the “RoPE trick.”_

## The proof is given in Appendix B.2.

To observe the connection of complex SSMs to RoPE embeddings, note that in the above proposition, the data-dependent rotations _**R** 𝑖_ are aggregated across time-steps and applied to _**C** ,_ _**B**_ , which, by the state space duality framework, correspond to the query ( _**Q**_ ) and key ( _**K**_ ) components of attention (Section 2.3). Analogously, vanilla RoPE (Su et al. 2023) applies _data-independent_ rotation matrices, where the rotation angles follow a fixed frequency schedule _**θ**_ [ _𝑖_ ] = 10000[−][2] _[𝑖]_[/] _[𝑁]_ .

## **3.2.2 Complex SSM with Exponential-Trapezoidal Discretization**

After deriving the recurrence for complex SSMs with exponential-Euler discretization, the generalization to exponentialtrapezoidal discretization is similar. Proposition 4 provides the full recurrence with the RoPE trick for Mamba-3.

**Proposition 4** (Rotary Embedding Equivalence with Exponential-Trapezoidal Discretization) **.** _Discretizing a complex SSM with the exponential-trapezoidal rule (Proposition 1) yields the recurrence_

**==> picture [361 x 66] intentionally omitted <==**

_Here,_ _**R** 𝑡 is the block-diagonal rotation matrix defined in Proposition 2._

The proof is in Appendix B.3.

We empirically validate that our complex SSM, implemented via data-dependent RoPE, is capable of solving state-tracking tasks that real-valued SSMs with and without standard RoPE cannot (Table 5b), supporting theoretical claims.

## **3.3 Multi-Input, Multi-Output**

Scaling test-time compute has opened new frontiers in model capability, such as agentic workflows, where inference takes up an increasing share of the overall compute budget. This has placed a renewed focus on inference efficiency of

Table 2: Arithmetic Intensity for (a) SISO, (b) MIMO. The batch and head dimensions cancel out. The arithmetic intensity of MIMO increases linearly with rank _𝑅_ , enabling better hardware utilization during memory-bound phases like decode. Here _𝑁_ is the state size (expansion factor) and _𝑃_ is the head dimension. For Mamba-3, typically _𝑅_ ≪ _𝑁, 𝑃_ .

|**Input**|**Output**|**FLOPs**|**Arithmetic**|**Input**|**Output**|**FLOPs**|**Arithmetic**|
|---|---|---|---|---|---|---|---|
||||**Intensity**||||**Intensity**|
|_𝐻𝑡_: (_𝑁, 𝑃_)<br>_𝑥𝑡_: (_𝑃_)|_𝑦𝑡_: (_𝑃_)|5_𝑁𝑃_−_𝑃_|5_𝑁𝑃_−_𝑃_<br>2(1+2_𝑁_+_𝑃_+_𝑁𝑃_)<br>≈2_._5=Θ(1)|_𝐻𝑡_: (_𝑁, 𝑃_)<br>_𝑥𝑡_: (_𝑃, 𝑅_)|_𝑦𝑡_: (_𝑃, 𝑅_)|4_𝑁𝑃𝑅_+<br>_𝑁𝑃_−_𝑃𝑅_|4_𝑁𝑃𝑅_+_𝑁𝑃_−_𝑃𝑅_<br>2(1+2_𝑁𝑅_+_𝑃𝑅_+_𝑁𝑃_)<br>=Θ(min(_𝑁, 𝑃, 𝑅_))|
|_𝑎𝑡_: (1)||||_𝑎𝑡_: (1)|||=Θ(_𝑅_),_𝑅_≪_𝑁, 𝑃_|
|_𝑏𝑡_: (_𝑁_)||||_𝑏𝑡_: (_𝑁, 𝑅_)||||
|_𝑐𝑡_: (_𝑁_)||||_𝑐𝑡_: (_𝑁, 𝑅_)||||
||(a) SISO (2-byte||data).||(b) MIMO (2-byte||data).|

language models and spurred the adoption of SSMs and sub-quadratic layers which feature fixed-sized hidden states and thus offer lower compute and memory requirements. Although these new layers have a lower wall-clock time compared to Transformers, their decoding is heavily memory-bound, resulting in low hardware utilization. In this section, we use the SSM perspective to introduce a methodological refinement to the Mamba-3 recurrence that allows for _increased model FLOPs without increasing decoding wall-clock time, resulting in a better model with the same decoding speed._

**Decoding Arithmetic Intensity.** To improve hardware efficiency, we need to consider the arithmetic intensity of token generation, defined as FLOPs divided by the number of input-output bytes for a given op. Since SSM decoding saturates the memory bandwidth with idle compute (i.e., being _memory-bound_ ), we would like to increase its arithmetic intensity to effectively overlay compute with memory I/O. More concretely, the arithmetic intensity for a single generation in Mamba is around 2 _._ 5 ops per byte (Table 2a), while the arithmetic intensity for bfloat16 matmul is about 295 ops per byte for NVIDIA H100-SXM5 (NVIDIA 2022). Consequently, SSM decoding falls far short of a compute-bound regime, and moreover it is not clear how one can adjust the existing parameters in Mamba to mitigate the lack of hardware efficiency. We note that this observation applies generally to other sub-quadratic models, such as causal linear attention.

**From SISO to MIMO.** Consider a single head of a typical SSM with _head dimension 𝑃_ , which involves stacking the SISO recurrence _**h** 𝑡_ ← _𝛼𝑡_ _**h** 𝑡_ −1 + Δ _𝑡_ _**B** 𝑡𝑥𝑡_ with _𝑃_ copies sharing the same _𝛼𝑡,_ Δ _𝑡_ and _**B** 𝑡_ . The resulting broadcasted recurrence _**h** 𝑡_ ← _𝛼𝑡_ _**h** 𝑡_ −1 + Δ _𝑡_ _**B** 𝑡_ _**x** 𝑡_[⊤][takes vector inputs] _**[ x]**[𝑡]_[∈][R] _[𝑃]_[and has matrix-valued states] _**[ h]**[𝑡]_[∈][R] _[𝑁]_[×] _[𝑃]_[.]

Note that the memory traffic (input and output size) is dominated by the state _**h** 𝑡_ , while the computation mainly comprises the outer product _**B** 𝑡_ _**x** 𝑡_[⊤][which][has][FLOPs][proportional][to] _[𝑁𝑃]_[.][By][increasing][the][dimension][of][the][latter][terms,] transforming _**B** 𝑡_ ∈ R _[𝑁]_ → _**B** 𝑡_ ∈ R _[𝑁]_[×] _[𝑅]_ and _**x** 𝑡_ ∈ R _[𝑃]_ → _**x** 𝑡_ ∈ R _[𝑃]_[×] _[𝑅]_ , the memory traffic does not significantly increase (for small _𝑅_ ) while the FLOPs consumed increase by a factor of _𝑅_ (Table 2a). Thus, this transformation increases the arithmetic intensity of the recurrence. Furthermore, the increase in arithmetic intensity is translated into practical gains, since the outer product _**B** 𝑡_ _**x** 𝑡_[⊤][becomes a hardware-efficient matrix-matrix product (matmul),][which is computed using] fast tensor-cores, incurring only a marginal latency cost. As a result, the MIMO recurrence is more expressive than the original SISO recurrence, computing _𝑅_ × more FLOPs while practically preserving the decoding speed.

For similar reasons, the computation of the output from the state, _**y** 𝑡_ ← _**C** 𝑡_[⊤] _**[h]**[𝑡]_[acquires][an][extra][rank] _[ 𝑅]_[by][modifying] the output projection as _**C** 𝑡_ ∈ R _[𝑁]_ → _**C** 𝑡_ ∈ R _[𝑁]_[×] _[𝑅]_ . Overall, this transformation is equivalent to expanding the original single-input, single-output (SISO) recurrence to multi-input, multi-output (MIMO).

**Training MIMO SSMs.** While the MIMO formulation is motivated by _inference_ efficiency, the _training_ algorithms for SSMs (including our developments in Section 3.1, Section 3.2) have been typically developed for SISO models. We begin with the observation that MIMO SSMs can be expressed in terms of _𝑅_[2] SISO SSMs, where _𝑅_ SISO SSMs sharing the same recurrence are summed for each of the _𝑅_ MIMO outputs. In particular, define _**C** 𝑡_[(] _[𝑖]_[)] ∈ R _[𝑁] ,_ _**B** 𝑡_[(] _[𝑗]_[)] ∈ R _[𝑁] ,_ _**x** 𝑡_[(] _[𝑗]_[)] ∈ R _,_ Δ _𝑡_ ∈ R,

where _𝑖, 𝑗_ ∈{0 _, ..., 𝑅_ − 1}, then we have,

**==> picture [305 x 74] intentionally omitted <==**

⊤ Thus, _𝑦𝑡_[(] _[𝑖]_[)] =[�] _𝑗_[SSM][�] _[𝛼,]_[ Δ] _[,]_ _**[ B]**_[(] _[𝑗]_[)] _[,]_ _**[ C]**_[ (] _[𝑖]_[)] _[,]_ _**[ x]**_[(] _[𝑗]_[)][�] _𝑡_[, where][ SSM][�] _[𝛼,]_[ Δ] _[,]_ _**[ B]**_[(] _[𝑗]_[)] _[,]_ _**[ C]**_[ (] _[𝑖]_[)] _[,]_ _**[ x]**_[(] _[𝑗]_[)][�] _𝑡_[:][=] � _**C** 𝑡_[(] _[𝑖]_[)] � _**h** 𝑡_[(] _[𝑗]_[)] with _**h** 𝑡_[(] _[𝑗]_[)] from (12).

Furthermore, improvements to standard SISO-based SSM models can be directly applied to MIMO models as the underlying SISO training algorithms can be utilized as a black-box. This observation allows a MIMO model to be trained by invoking the SISO algorithm _𝑅_[2] times as a black box in parallel. In contrast, when computed in the recurrent form, equation (12), (13), and (14) can be performed sequentially, incurring only an _𝑅_ -times overhead relative to SISO SSMs (recall the discussion on MIMO decoding FLOPs).

**Chunked Algorithm for MIMO SSMs.** Many modern SISO recurrent models, including Mamba-2, are computed using a _chunked_ algorithm, where the sequence is divided into chunks of length _𝐶_ . Within each chunk, a parallel (but asymptotically slower) algorithm is applied, while a recurrence is computed across chunks. Chunked algorithms interpolate between two extremes: a fully parallel and a fully sequential algorithm. By exploiting this structure, we can reduce the training cost of MIMO SSMs to _𝑅_ times that of SISO SSMs. This idea also appears in the SSD framework—SSD applies a hardware-friendly quadratic algorithm within each chunk, while using the recurrent form across chunks, and shows that when the state and head dimensions are comparable, setting the chunk size to this dimension yields an overall lineartime algorithm. Specifically, SSD’s intra-chunk computation incurs[�] 2 _𝐶_[2] _𝑁_ + 2 _𝐶_[2] _𝑃_[�] FLOPs per chunk, giving a total of _𝑇𝐶_ �2 _𝐶_ 2 _𝑁_ + 2 _𝐶_ 2 _𝑃_ � = 2 _𝑇𝐶_ ( _𝑁_ + _𝑃_ ). The inter-chunk computation incurs 4 _𝑁𝑃𝐶_ + 2 _𝑁𝑃_ FLOPs per chunk, for a total of _𝑇_[=][4] _[𝑇𝑁𝑃]_[+] _[𝑇]_[Setting] _[ 𝐶]_[=] _[𝑃]_[=] _[𝑁]_[, the total FLOP count is 8] _[𝑇𝑁]_[2][, which] _𝐶_[(][4] _[𝑁𝑃𝐶]_[+][ 2] _[𝑁𝑃]_[)] _𝐶_[2] _[𝑁𝑃]_[(ignoring negligible terms).] is linear in _𝑇_ . The chunked algorithm for SSD can be naturally generalized into MIMO SSMs. In such a case, the FLOP counts of state projection _**Bx**_[⊤] and state emission _**C**_[⊤] _**h**_ increase by _𝑅_ ×, while the FLOP count of the intrachunk component _**C**_[⊤] _**B**_ increases by _𝑅_[2] ×. As a result, the intra-chunk computation incurs 2 ·[�] _[𝑇] 𝐶_[(] _[𝐶𝑅]_[)][2] _[𝑁]_[+] _[𝑇] 𝐶_[(] _[𝐶𝑅]_[)][2] _[𝑃]_[�][FLOPs][and][inter-chunk] computation incurs 4 · _[𝑇] 𝐶[𝑁𝑃]_[(] _[𝐶𝑅]_[)][+][ 2][ ·] _[𝑇] 𝐶[𝑁𝑃]_[FLOPs.][Thus,][setting] _[ 𝐶𝑅]_[=] _[𝑁]_[=] _[𝑃]_[yields a total FLOP count of 8] _[𝑇𝑅𝑁]_[2][,][an] _𝑅_ -fold increase in FLOP count. Intuitively, setting MIMO chunk size as _𝑅_[1][times the SISO chunk size, i.e.,] _[ 𝐶]_[MIMO][←] _𝑅_[1] _[𝐶]_[SISO][,] maintains the SISO intra-chunk FLOP count while increasing the number of chunks by a factor of _𝑅_ , resulting in an overall _𝑅_ -times increase in FLOP count instead of an _𝑅_[2] -times increase while keeping the algorithm hardware-friendly.

The training speed of algorithms in practice depends on details of the kernel implementation strategy, architectural choices such as how the MIMO parameters are instantiated, and problem dimensions, but should be no more than _𝑅_ times slower. Our released Triton Mamba-3 SISO kernels are roughly on par with the Triton Mamba-2 kernels, and MIMO kernels only incur a slowdown of 2× when _𝑅_ = 4, as compute latency can be parallelized with memory movement. Table 6 benchmarks the prefill speed of various kernels which is equivalent to the forward pass of the training kernel.

**MIMO Instantiation.** Among various choices for MIMO parameterizations, Mamba-3’s approach achieves a balance that preserves the state size and number of SSMs of its SISO counterpart, while avoiding excessive growth in parameter count. The naive conversion of a SISO SSM to a rank _𝑅_ MIMO SSM would incur an _𝑅_ × increase in parameters as all projections that model the inputs to the SSM, _**B** ,_ _**C** ,_ _**x**_ , would increase. Block-level components, such as the gate _**z**_ (which so far has been ignored for simplicity) and output _**y**_ projection would also be impacted. This influx in parameter count would be intractable at larger model scales. To counteract this, we make the following change. Mamba’s multi-value attention (MVA) head structure results in shared _**B** ,_ _**C**_ across heads, so these components’ projections can be directly converted to incorporate the new MIMO rank _𝑅_ with only a slight increase in parameter count from _𝐷𝑁_ to _𝐷𝑁𝑅_ for the entire layer (recall _𝐷_ as the model dimension). However, the SSM input _**x** 𝑡_ , output _**y** 𝑡_ , and gate _**z** 𝑡_ are unique per head and therefore dominate the parameter count. Here, directly adjusting the projections would increase the parameter count from _𝐷𝑃_ to _𝐷𝑃𝑅_ for _each head_ . Instead, we keep the original SISO projection and element-wise scale each dimension of the projected output to size _𝑅_ with a learnable, data-independent vector, resulting in _𝐷𝑃_ + _𝑃𝑅_ parameters for each head.

**==> picture [491 x 244] intentionally omitted <==**

N<br>Linear projection<br>X X<br>Y Y<br>SSM SSM Sequence transformation<br>A X B C A X B C<br>MIMO projection (optional)<br>! ! RoPE !<br>& Nonlinearity (activation,<br>N N<br>Conv normalization, multiplication, etc.)<br>Mamba-2 Block Mamba-3 Block<br>**----- End of picture text -----**<br>

Figure 2: Contrasting Mamba-2 and Mamba-3 Architectures: Key updates include exponential-trapezoidal discretization, data-dependent RoPE embeddings, MIMO projections, QK normalization, and learnable biases.

This mitigates the multiplicative increase to a more reasonable additive parameter count increase. Appendix C details the parameterization, and all MIMO-variants in our paper are parameter-matched to their SISO counterparts by reducing the MLP width.

_Remark_ 6 _._ For simplicity, all discussion in this section was for simpler 2-term recurrences such as that arising from exponential-Euler discretization; the generalization to the 3-term exponential-trapezoidal recurrence is similar.

## **3.4 Mamba-3 Architecture**

The overall architecture follows Llama (Grattafiori et al. 2024), alternating Mamba-3 and SwiGLU blocks with pre-norm. The Mamba-3 block retains the overall layout of its predecessor, while introducing several key modifications.

**Updated SSM Recurrence.** The SSD layer is replaced with the more expressive complex-valued exponential-trapezoidal SSM defined in Proposition 4. Mamba-3 employs the SISO SSM by default to enable fair comparisons with other SISO-like models, but its MIMO variant can be trained and deployed as a stronger alternative to baseline Mamba-3 (Table 3). Our SSM _**A**_ is complex with both real and imaginary components produced by data-dependent projections. With Figure 2, this is partitioned into the real-valued _𝐴_ and imaginary-valued Θ; the former is passed into the SSD black box as in Mamba-2, while the latter is computed through the RoPE trick.

**BC / QK Normalization.** RMS normalizations are added following the _**B** ,_ _**C**_ projection, mirroring the QKNorm commonly used in modern Transformers (Henry et al. 2020; Wortsman et al. 2023) and other recent linear models (Hu et al. 2025; S. Yang, Kautz, and Hatamizadeh 2025). We call this either BC normalization (BCNorm) or QK normalization (QKNorm) interchangeably. We find that BCNorm is also able to stabilize large-scale runs, resulting in the removal of the post-gate RMSNorm layer (introduced in Mamba-2 for stability) in our pure Mamba-3 models. However, in hybrid models, the removed RMSNorm layer is crucial for long-context extrapolation (Table 4).

_**B** ,_ _**C**_ **Biases.** Similarly to Yu and Erichson (2025), which proved that adding channel-specific biases to _**B**_ in a blockwise variant of Mamba-1 grants universal approximation capabilities, Mamba-3 incorporates learnable, head-specific, channelwise biases into the _**B**_ and _**C**_ components after the BCNorm.

Table 3: Downstream language modeling evaluations on models trained with 100B FineWeb-Edu tokens. Best results for each size are **bolded** , and second best are underlined, excluding Mamba-3 MIMO variants. All models are trained with the same procedure. Mamba-3 SISO outperforms Mamba-2 and others at every model scale, and the MIMO variant with rank _𝑅_ = 4 further improves modeling capabilities.

|**Model**|FW-Edu|LAMB.|LAMB.|HellaS.|PIQA|Arc-E|Arc-C|WinoGr.|OBQA|Average|
|---|---|---|---|---|---|---|---|---|---|---|
||ppl↓|ppl↓|acc↑|acc_n↑|acc↑|acc↑|acc_n↑|acc↑|acc↑|acc↑|
|Transformer-180M|16_._89|45_._0|32_._5|39_._0|67_._1|59_._8|27_._9|51_._2|21_._8|42_._8|
|GDN-180M|16_._52|40_._8|31_._3|40_._2|66_._3|62_._3|28_._2|51_._7|22_._0|43_._2|
|Mamba-2-180M|16_._76|41_._8|30_._9|40_._1|66_._8|60_._1|27_._3|52_._0|23_._2|42_._9|
|**Mamba-3-SISO-180M**|16_._59|37_._7|32_._5|40_._8|66_._1|61_._5|27_._9|52_._0|22_._8|43_._4|
|**Mamba-3-MIMO-180M**|16_._46|32_._1|34_._0|41_._0|66_._7|60_._6|27_._7|52_._9|22_._0|43_._5|
|Transformer-440M|13_._03|21_._2|41_._7|50_._5|69_._9|67_._6|34_._6|56_._7|26_._0|49_._6|
|GDN-440M|13_._01|18_._0|41_._9|50_._9|70_._0|67_._0|34_._6|56_._1|27_._6|49_._7|
|Mamba-2-440M|13_._00|19_._6|40_._8|51_._7|70_._6|68_._8|35_._0|54_._1|26_._0|49_._6|
|**Mamba-3-SISO-440M**|12_._87|19_._6|40_._2|51_._7|71_._9|68_._9|34_._4|55_._8|26_._0|49_._8|
|**Mamba-3-MIMO-440M**|12_._72|17_._1|43_._4|52_._8|70_._8|69_._6|35_._6|56_._3|28_._4|51_._0|
|Transformer-880M|11_._42|15_._0|44_._7|57_._2|72_._6|71_._6|39_._2|57_._7|26_._8|52_._8|
|GDN-880M|11_._37|12_._9|47_._6|57_._3|73_._3|71_._4|38_._7|58_._8|28_._6|53_._7|
|Mamba-2-880M|11_._35|13_._8|45_._0|58_._1|72_._5|72_._3|38_._7|56_._8|30_._2|53_._4|
|**Mamba-3-SISO-880M**|11_._23|12_._9|47_._2|58_._8|73_._6|72_._7|40_._2|58_._4|30_._0|54_._4|
|**Mamba-3-MIMO-880M**|11_._11|11_._8|49_._5|59_._2|73_._7|74_._7|41_._2|59_._9|28_._6|55_._3|
|Transformer-1.5B|10_._51|11_._1|50_._3|60_._6|73_._8|74_._0|40_._4|58_._7|29_._6|55_._4|
|GDN-1.5B|10_._45|10_._9|49_._2|61_._3|74_._3|75_._3|41_._2|58_._0|31_._6|55_._8|
|Mamba-2-1.5B|10_._47|12_._0|47_._8|61_._4|73_._6|75_._3|41_._8|57_._5|32_._6|55_._7|
|**Mamba-3-SISO-1.5B**|10_._35|10_._9|49_._4|61_._9|73_._6|75_._9|42_._7|59_._4|32_._0|56_._4|
|**Mamba-3-MIMO-1.5B**|10_._24|10_._2|51_._7|62_._3|75_._3|76_._5|44_._5|60_._6|32_._6|57_._6|

We hypothesize that these biases also induce a convolution-like behavior in the model. Specifically, adding biases to _**B**_ and _**C**_ introduces data-independent components into SSMs that function more similarly to convolutions. Ablations on the bias parameterization are located in Appendix F.

The combination of data-independent bias parameters, together with exponential-trapezoidal discretization (which itself induces a convolution on the state-input), is empirically able to obviate the short causal convolution and its accompanying activation function present in Mamba-2 and most modern recurrent models (Section 4.2).

## **4 Empirical Validation**

We empirically validate our SSM-centric methodological changes through the Mamba-3 model on a host of synthetic and real-world tasks. Section 4.1 evaluates Mamba-3 on language modeling and retrieval-based tasks. Section 4.2 ablates the effect of our new SSM components such as discretization and complex transitions. Section 4.3 explores the inference efficiency of the Mamba-3 family and MIMO Mamba-3’s benefits over the SISO variant under fixed inference compute, and Section 4.4 benchmarks the performance of our Mamba-3 training and inference kernels.

## **4.1 Language Modeling**

All models are pretrained with 100B tokens of the FineWeb-Edu dataset (Penedo et al. 2024) with the Llama-3.1 tokenizer (Grattafiori et al. 2024) at a 2K context length with the same standard training protocol. Training and evaluation details can be found in Appendix D.

Across all four model scales, Mamba-3 outperforms popular baselines at various downstream tasks (Table 3). We highlight that Mamba-3 does not utilize the external short convolution that has been empirically identified as an important compo-

nent in many performant linear models (Allen-Zhu 2025; Gu and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2025).

## **4.1.1 MIMO**

We aim to further verify the gain from MIMO by investigating its language-modeling capabilities by training MIMO models with rank _𝑅_ = 4 under the same settings. To ensure that the total parameter count is comparable to SISO-based models, we decrease the inner dimension of the MLP layers in MIMO models to compensate for the increase due to the MIMO projections. In the 1.5B-parameter models, for instance, the MLP inner dimension is reduced by only 6 _._ 6%, from 4096 to 3824. See Appendix C for more details.

On both validation perplexity and our suite of language evaluation tasks (Table 3), we see significant gains when moving from SISO to MIMO for our Mamba-3 models. Namely, we achieve a significant perplexity gain of 0 _._ 11 on the 1.5B models, and Figure 3 illustrates the downward shift in our validation loss. On the language evaluation front, we see gains on most tasks when compared to SISO, resulting in an average gain of 1.2 percentage points over SISO.

## **4.1.2 Retrieval Capabilities**

Beyond standard language modeling, an important measure for linear models is their retrieval ability—how well they can recall information from earlier in the sequence (A. Arora et al. 2025; S. Arora, Eyuboglu, et al. 2025). Unlike attention models, which can freely revisit past context with the growing KV cache, linear models must compress context into a fixed-size state. This trade-off is reflected in the Transformer baseline’s substantially stronger retrieval scores. To evaluate Mamba-3 under this lens, Table 4 compares it against baselines on both real-world and synthetic needle-in-ahaystack (NIAH) tasks (Hsieh et al. 2024), using our pretrained 1.5B models from Section 4.1. We restrict the task sequence length to 2K tokens to match the training setup and adopt the cloze-style format for our real-world tasks to mirror the next-token-prediction objective, following S. Arora, Eyuboglu, et al. (2025) and S. Arora, Timalsina, et al. (2024).

Mamba-3 is competitive on real-world associative recall and question-answering (TQA, SQuAD) but struggles when extracting information from semi-structured or unstructured data (SWDE, FDA). On synthetic NIAH tasks, however, Mamba3 surpasses or matches baselines on most cases and notably demonstrates markedly better out-of-distribution retrieval abilities than its Mamba-2 predecessor.

**Improving Retrieval with Hybrid Models.** Because of the natural retrieval-based weaknesses of fixed state-size, we predict that linear layers will be predominantly used _in hybrid architectures_ that mitigate this downside with quadratic selfattention layers. To evaluate how Mamba-3 performs within this architectural paradigm, we train our hybrid models at the same scale in an interleaving fashion with a 5:1 ratio of linear layer to NoPE self-attention (B. Yang et al. 2025). As seen in prior work (Waleffe et al. 2024), hybrid models outperform the Transformer baseline. We find that the reintroduction of the pre-output projection RMSNorm (pre-gate, grouped RMSNorm in Table 4) to the Mamba-3 layer improves the length generalization retrieval abilities at the slight cost of in-context, real-world retrieval tasks and is highly competitive as a linear sequence mixing backbone when mixed with self-attention. However, the ideal norm type (grouped vs default) and its placement (pre- vs post-gate) is still unclear due to competing tradeoffs (Appendix E, Table 9), as we find that hybrid models and their exact characteristics and dynamics are complex and oftentimes unintuitive, a point echoed in recent works such as Cabannes et al. (2025).

## **4.2 SSM Methodology Ablations**

Table 5a ablates the changes that Mamba-3 introduces to core SSM components, mainly the introduction of BC bias and exponential-trapezoidal discretization. We report the pretraining test perplexity on models at the 440M scale, trained for Chinchilla optimal tokens. We find that the bias and exponential-trapezoidal SSM synergize well and make the short convolution utilized by many current linear models redundant.

We empirically demonstrate that data-dependent RoPE in Mamba-3 enables state tracking. Following Grazzi, Siems, Zela, et al. (2025), we evaluate on tasks from the Chomsky hierarchy—Parity, Modular Arithmetic (without brackets), and Modular Arithmetic (with brackets)—and report scaled accuracies in Table 5b. Mamba-3 solves Parity and Modular Arithmetic (without brackets), and nearly closes the accuracy gap on Modular Arithmetic (with brackets). In contrast, Mamba-3 without RoPE, Mamba-3 with standard RoPE (Su et al. 2023), and Mamba-2 fail to learn these tasks. We use the state-trackingenabled variant of GDN and observe that Mamba-3 is competitive—matching parity and approaching its performance on

Table 4: Retrieval capabilities measured by a mixture of real-world and synthetic retrieval tasks. Real-world retrieval tasks utilize cloze variants of the original datasets and are truncated to 2K length. Mamba-3 demonstrates strong associative recall, question-answering, and length generalization on needle-in-a-haystack (NIAH), but suffers with information extraction of semi-structured and unstructured data. The Transformer baseline uses RoPE which may explain its length generalization issues, and hybrid models utilize NoPE (no positional embeddings). We find a pre-gate, grouped RMSNorm can be added to Mamba-3 SISO hybrid models to improve the length generalization of the NIAH tasks at a slight decrease in real-world retrieval performance.

|**Model (1.5B)**<br>Context Length|**Model (1.5B)**<br>Context Length|SWDE<br>SQD.<br>FDA<br>TQA<br>NQ<br>Drop<br>2048|NIAH-Single-1<br>1024<br>2048<br>4096|NIAH-Single-2<br>1024<br>2048<br>4096|NIAH-Single-3<br>1024<br>2048<br>4096|
|---|---|---|---|---|---|
|Pure|Transformer<br>GDN<br>Mamba-2<br>**Mamba-3 SISO**<br>**Mamba-3 MIMO**|48_._9<br>46_._6<br>58_._4<br>67_._5<br>31_._7<br>26_._4|100_._0<br>100_._0<br>0_._0|92_._2<br>100_._0<br>0_._0|98_._6<br>99_._4<br>0_._0<br>83_._8<br>68_._4<br>34_._2<br>95_._8<br>87_._4<br>13_._4<br>92_._4<br>81_._4<br>34_._2<br>95_._8<br>84_._4<br>25_._6|
|||32_._7<br>40_._0<br>28_._3<br>63_._5<br>25_._7<br>24_._5<br><br>30_._7<br>39_._1<br>23_._7<br>64_._3<br>25_._1<br>28_._5<br><br>28_._5<br>40_._1<br>23_._4<br>64_._5<br>26_._5<br>27_._4<br><br>36_._3<br>41_._7<br>29_._3<br>64_._5<br>26_._2<br>26_._3<br>|100_._0<br>100_._0<br>99_._8<br><br>100_._0<br>99_._6<br>62_._0<br><br>100_._0<br>100_._0<br>88_._2<br><br>100_._0<br>100_._0<br>93_._0<br>|100_._0<br>93_._8<br>49_._8<br>100_._0<br>53_._8<br>11_._8<br>100_._0<br>95_._4<br>50_._6<br>100_._0<br>86_._0<br>40_._4||
|Hybrid|GDN<br>Mamba-2<br>Mamba-3 SISO<br>Mamba-3 SISO Norm∗|54_._6<br>48_._4<br>58_._8<br>64_._9<br>32_._7<br>30_._0<br><br>58_._2<br>45_._6<br>71_._0<br>66_._1<br>33_._4<br>28_._1<br><br>58_._5<br>47_._0<br>65_._9<br>64_._8<br>33_._4<br>27_._0<br><br>58_._6<br>47_._3<br>52_._4<br>65_._7<br>33_._3<br>28_._5<br>|100_._0<br>100_._0<br>71_._4<br>100_._0<br>100_._0<br>3_._2<br>100_._0<br>100_._0<br>36_._2<br><br>100_._0<br>100_._0<br>100_._0<br>|99_._6<br>100_._0<br>60_._2<br>99_._6<br>98_._8<br>0_._0<br>100_._0<br>100_._0<br>9_._4<br>100_._0<br>100_._0<br>96_._0|70_._0<br>96_._2<br>24_._0<br>98_._2<br>98_._0<br>0_._0<br>99_._8<br>100_._0<br>8_._8<br>99_._8<br>97_._2<br>56_._8|

Table 5: **Left** : Ablations on core modeling components of Mamba-3 SISO, results on test split of dataset. **Right** : Formal language evaluation (scaled accuracy, %). Higher is better. SISO models are trained on short sequences and evaluated on longer lengths to test length generalization. For GDN we report the variant with eigenvalue range [−1 _,_ 1].

|**Model Variant**<br>ppl↓<br>Mamba-3−bias−trap<br>16_._68<br>Mamba-3−bias<br>16_._49<br>Mamba-3<br>15_._72<br>Mamba-3+conv<br>15_._85<br>bli  440M l A bii|**Model**<br>Parity↑<br>Arith. w/o<br>brackets↑<br>Arith. w/<br>brackets↑|
|---|---|
||Mamba-3<br>100_._00<br>98_._51<br>87_._75<br>Mamba-3 (w/ Std. RoPE)<br>1_._56<br>20_._70<br>2_._62<br>Mamba-3 (w/o RoPE)<br>2_._27<br>1_._49<br>0_._72<br>Mamba-2<br>0_._90<br>47_._81<br>0_._88<br>GDN [-1,1]<br>100_._00<br>99_._25<br>93_._50|

(a) Component ablation at 440M scale. A combination of our BC bias and exponential-trapezoidal discretization makes the ubiquitous short convolution optional.

(b) Performance comparison on formal language tasks. Results show that unlike Mamba-2, Mamba-3 features state-tracking ability stemming from data-dependent RoPE embeddings.

both modular-arithmetic tasks. Experimental settings are covered in Appendix D.

## **4.3 Inference Efficiency to Performance Tradeoff**

As _𝑑_ state governs the decode runtime for the sub-quadratic models considered in this paper (Section 3.3), we use it as a proxy for inference speed. By plotting the validation perplexity (a proxy for model performance) as a function of _𝑑_ state, we aim to formulate a holistic picture about how sub-quadratic models can trade off performance with inference speed.

Figure 3 shows such a Pareto frontier for the Mamba models considered in this paper. For each data point, we train a 440M parameter model to 2× Chinchilla optimal tokens on the Fineweb-Edu dataset, where the model is configured with a _𝑑_ state of {16 _,_ 32 _,_ 64 _,_ 128}. As expected, we observe an inverse correlation between validation loss and _𝑑_ state. Moreover, there is a general downward shift on the Pareto frontier moving from Mamba-2 to Mamba-3, indicating a stronger model: in this setting, Mamba-3 with 2× smaller state size achieves better pretraining perplexity than its Mamba-2 counterpart, resulting in a faster model with the same quality or a better model for the same speed.

A further downward shift is observed when moving from the SISO variant of Mamba-3 to the MIMO variant of Mamba-3 (where we set the MIMO rank _𝑅_ = 4 and decrease the MLP inner dimension to parameter match the SISO variants).

We expand the comparison to include the GDN baseline in Appendix E, Figure 6, which also shows Mamba-3 comparing favorably to GDN.

|14.4<br>14.6<br>14.8<br>15.0<br>15.2<br>Pretraining Perplexity|105<br>Relative Total State Size<br>Relative Total State Size vs Pretraining Perplexity<br>Mamba-2<br>Mamba-3<br>Mamba-3 MIMO||||
|---|---|---|---|---|
|||**Model**|**FP32**<br>_𝑑_state =64<br>_𝑑_state =128|**BF16**<br>_𝑑_state =64<br>_𝑑_state =128|
|||Mamba-2<br>GDN<br>Mamba-3 (SISO)<br>Mamba-3 (MIMO)|0_._295<br>0_._409<br>0_._344<br>0_._423<br>0_._310<br>0_._399<br>0_._333<br>0_._431|0_._127<br>0_._203<br>0_._176<br>0_._257<br>0_._110<br>0_._156<br>0_._137<br>0_._179|
|||Table 6:<br>Kernel latency (in milliseconds) comparison<br>|||

Table 6: Kernel latency (in milliseconds) comparison across models, precision, and _𝑑_ state values. Mamba-3 introduces minimal overhead compared to Mamba-2 and features highly efficient practical implementations. Our Mamba-3 SISO kernels are faster than reference Mamba-2 and GDN kernels at the commonly used bf16, _𝑑_ state = 128 setting. Mamba-3 MIMO ( _𝑅_ = 4) incurs little additional cost compared to SISO.

Figure 3: Exploration of state size (inference speed proxy) versus pretraining perplexity (performance proxy) across different Mamba variants. Mamba-3 improves the Pareto frontier compared to previous recurrent SISO models, while incorporating MIMO further shifts the frontier through better modeling performance without increasing state size.

Table 7: Prefill and Prefill+Decode latency across sequence lengths. Mamba-3 adds minimal overhead to its forward-pass and retains competitive decode latencies. Details in Appendix G.

|**Model**|**512 tokens**<br>**1024 tokens**<br>**2048 tokens**<br>**4096 tokens**<br>**16384 tokens**<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec|**512 tokens**<br>**1024 tokens**<br>**2048 tokens**<br>**4096 tokens**<br>**16384 tokens**<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec<br>Prefll<br>Prefll+Dec|
|vLLM (Llama-3.2-1B)<br>Gated DeltaNet<br>Mamba-2<br>Mamba-3 (SISO)<br>Mamba-3 (MIMO_𝑅_=4)|**0.26**<br>4.45<br>**0.52**<br>9.60<br>**1.08**<br>20.37<br>**2.08**<br>58.64<br>**12.17**<br>0.51<br>4.56<br>1.01<br>9.11<br>2.01<br>18.22<br>4.00<br>36.41<br>16.21<br>0.51<br>4.66<br>1.02<br>9.32<br>2.02<br>18.62<br>4.02<br>37.22<br>16.22<br>0.51<br>**4.39**<br>1.01<br>**8.78**<br>2.02<br>**17.57**<br>4.01<br>**35.11**<br>16.22<br>0.60<br>4.74<br>1.21<br>9.48<br>2.42<br>18.96<br>4.76<br>37.85<br>19.44|976.50<br>145.87<br>149.02<br>**140.61**<br>151.81|

## **4.4 Fast Mamba-3 Kernels**

We complement Mamba-3’s methodological advances with optimized kernels that deliver fast inference in practical settings. We implement a new series of inference kernels for Mamba-3—using Triton for the forward (prefill) path and CuTe DSL for decode—and compare their per-token decode latency against the released Triton kernels for Mamba-2 and GDN in Table 6.[5] The evaluation measures a single decode step at batch size 128 on a single H100 for both FP32 and BF16 datatypes; models are 1.5B parameters with model dimension 2048 and state dimension ∈{64 _,_ 128}. Across all configurations, SISO achieves the lowest latency amongst baselines. MIMO, with its higher arithmetic intensity, increases the decoding FLOPs without significantly increasing decode runtime. Our benchmarks indicate that our CuTe DSL decode implementation is competitive and that the additional components of Mamba-3 (exponential-trapezoidal update, complex-valued state, and MIMO projections) are lightweight. This supports our overall inference-first perspective: Mamba-3 admits a **simple, low-latency implementation** while providing strong empirical performance.

Table 7 benchmarks both end-to-end latency across different decoding sequence length and prefill time for the same sequence length. The decode time is consistent with Table 6, where Mamba-3 (SISO) is fastest; Mamba-3 (MIMO) is on par with Mamba-2; and all linear methods are faster than optimized attention as sequence length grows. We also see that MIMO incurs a moderate overhead for prefill, as discussed in Section 3.3. Details of the benchmark are in Appendix G.

> 5Details on each kernel DSL and the exact kernel fusion structure is provided in Appendix G.

## **5 Related Work**

## **5.1 Linear-Time Sequence Mixers**

A growing body of work seeks to replace the quadratic softmax-based attention mechanism (Bahdanau, Cho, and Bengio 2014; Vaswani et al. 2017) with linear runtime alternatives. Prominent approaches can be categorized under three broad frameworks: linear attention, test-time training, and state space models.

Many nascent linear attention (LA) models aimed to approximate softmax attention through kernel feature maps (Choromanski et al. 2022; Katharopoulos et al. 2020), while recent models have discarded the feature maps for raw dot-products between queries and keys, modulated by decays or masks (Yutao Sun et al. 2023; S. Yang, B. Wang, Shen, et al. 2024). More recently, fast-weight programmers Schlag, Irie, and Schmidhuber (2021) that modulate the state memory with key-value pairs have also fallen under the umbrella term “linear attention.” S. Yang, Kautz, and Hatamizadeh (2025) and S. Yang, B. Wang, Y. Zhang, et al. (2025) originated from this line of work and enhanced traditional linear attention by replacing the additive memory update with a delta-rule recurrence. This has further spurred on a host of work improving the efficiency and capabilities of linear models built on the delta rule (Hu et al. 2025; Kimi Team et al. 2025).

A parallel line of test-time training (TTT) or test-time regression (TTR) work views sequence modeling as an online learning task during inference. Here, the recurrent state represents a compressed summary of past inputs, and recurrent steps update the state to memorize new information (Yu Sun et al. 2025; Tandon et al. 2025; T. Zhang et al. 2025). Equivalently, these methods can be viewed as optimization of a global regression objective, and recurrent state updates represent iterative optimization procedures such as variants of gradient descent (K. A. Wang, Shi, and Fox 2025).

Structured state space models (SSMs) are another view of modern recurrent models inspired by classical signal processing and dynamical systems. Early versions of SSMs such as S4 (Gu, Goel, and Ré 2022; Gupta, Gu, and Berant 2022; Smith, Warrington, and Linderman 2023) used linear time invariant (LTI) layers with structured state transition matrices, for example diagonal or low-rank plus diagonal, to facilitate efficient computation and stable learning of long-context tasks (Gu, Goel, and Ré 2022; Gupta, Gu, and Berant 2022; Smith, Warrington, and Linderman 2023). The introduction of time-varying, input-dependent selectivity to SSMs in Mamba-1 (Gu and Dao 2024) reduced the disparity between self-attention and linear models on information-dense modalities, notably language modeling. Subsequently, Mamba-2 (Dao and Gu 2024) formalized the connection between SSMs and (linear) attention through the structured state space duality (SSD) that we build on in this work.

## **5.2 State Tracking and Complex State Space Models**

**Expressivity and State Tracking.** Recent work characterizes the types of state that recurrent, constant-memory mixers can maintain, revealing algorithmic deficiencies in previous SSM-based models. Merrill, Petty, and Sabharwal (2025) show that under finite precision, practical SSMs collapse to TC[0] , leading to failures on tasks like permutation composition over _𝑆_ 5 unless the primitive is extended. Similarly, Yu and Erichson (2025) prove that a single-layer Mamba is not a universal approximator. Several modifications have been proposed to improve expressivity. For instance, the same work shows that a block-biased variant regains the universal approximation property with only minor changes, either through block decomposition or a channel-specific bias. Allowing negative eigenvalues or non-triangular transitions enables linear RNNs—including diagonal and Householder/DeltaNet forms—to capture parity and, under mild assumptions, regular languages (Grazzi, Siems, Zela, et al. 2025). Complex-valued parameterizations provide another avenue for enhanced expressivity.

**Complex State Space Models.** Structured SSMs prior to Mamba were frequently complex-valued, rooted in traditional SSM theory. They also generally excelled in domains such as vision and audio, which have explicit frequency-based information content, rather than language. While some models such as H3 (Fu et al. 2023), RetNet (Yutao Sun et al. 2023), and Megalodon (Ma et al. 2024) kept complex-valued SSMs while targeting language modeling, they still noticeably underperformed Transformers.

Additionally, because these models were LTI and were computed using very different algorithms (in particular, convolutions or explicit recurrence) than modern selective SSMs such as Mamba, they generally did not use the RoPE trick to handle the complex part. An exception is RetNet, which introduced a model in between linear attention and Mamba-2 that used constant scalar decays (as opposed to no decay in LA and data-dependent decay in Mamba-2) with an additional

constant complex phase that was implemented through RoPE.

In general, complex numbers have been empirically found to be unhelpful for language modeling, and hence were phased out in Mamba-1 and successors, including parallel lines of work on linear attention and test-time training. Mamba-3 represents the first modern recurrent model with complex-valued state transitions, which were introduced for specific purposes of increasing expressivity and state-tracking ability. By incorporating the RoPE trick, this represents, to the best of our knowledge, the first usage of data-dependent RoPE grounded in theoretical motivations.

## **5.3 Multi-Input, Multi-Output**

S4 (Gu, Goel, and Ré 2022) is a single-input, single-output LTI system where each dimension of the input was assigned its own independent SSM. Such SISO models have a significantly larger recurrent state than classical RNNs, and necessitated more complicated mathematical machinery to compute them efficiently. Aiming to simplify the model, S5 (Smith, Warrington, and Linderman 2023) and LRU (Orvieto et al. 2023) replaced the set of SISO SSMs with a multi-input, multi-output SSM applied directly on the entire vectorized input. This change reduced the effective state capacity but enabled an alternate computation path by directly computing the recurrence with a parallel scan. While this trade-off between state capacity and modeling performance was less pronounced in LTI models, Mamba-1 (S6) (Gu and Dao 2024) and Mamba2 (Dao and Gu 2024) returned to the SISO system due to the importance of a large state size in the time-varying setting. The computational bottleneck associated with the increased state size was addressed with a hardware-aware parallel scan algorithm for Mamba-1 and a matrix multiplication-based algorithm for Mamba-2.

The introduction of MIMO to Mamba-3 significantly diverges from prior work. Unlike previous MIMO models, which aimed to simplify training algorithms at the cost of slightly reduced expressivity, Mamba-3’s MIMO structure is motivated to _increase_ modeling power while preserving _inference_ efficiency. Accordingly, its state expansion is kept at Mamba-1/-2 levels to maintain modeling capabilities while trading off additional training compute.

## **5.4 The State Space Model Viewpoint**

Although modern recurrent models have several different viewpoints that largely converge (Section 5.1), each framework has slightly different interpretations and motivations that can lead to different design spaces and extensions. In particular, linear attention and test-time training are more closely related and can perhaps be lumped together under a framework of _associative memory_ that explicitly aims to memorize input data through “key-value” stores; either through approximations to the canonical KV method (i.e., quadratic attention) in LA, or by minimizing soft optimization objectives in TTT. On the other hand, state space models have a different lineage, as reflected both in terminology (e.g., _𝐴, 𝐵,𝐶,𝑋_ instead of _𝑄, 𝐾,𝑉_ ) and in their natural extensions. Notably, the methodological improvements in Mamba-3 are all associated with the SSM viewpoint specifically and are less motivated from associative memory frameworks.

1. **Exponential-Trapezoidal Discretization.** The SSM viewpoint entails the discretization of a continuous ODE governing the system; our exponential-trapezoidal discretization falls out of an improved discretization method. As associative memory methods do not use discretization, it is not obvious how to interpret a 3-term recurrence such as exponential-trapezoidal under alternate viewpoints.

2. **Complex-Valued State Transitions.** Complex SSMs have long been a staple of dynamical systems, and it is natural to consider complex values as an extension of selective SSMs. On the other hand, the associative memory framework interprets the _𝐴_ state transition as a coefficient of an objective function, for example corresponding to the weight of an L2 regularization (or weight-decay) term in the optimization objective (K. A. Wang, Shi, and Fox 2025). However, complex values are meaningless as the coefficient of a regression objective; hence, Mamba-3 is not obviously interpretable within these frameworks.

3. **Multi-Input, Multi-Output.** MIMO is a classical concept from the state space model literature and does not naturally appear in associative memory (linear attention or test-time training) frameworks. However, we do note that the MIMO formulation introduced in this paper is not directly tied to SSM theory—and instead is motivated from a computational perspective—and our techniques can be adapted to other modern recurrent models as well.

There continues to be vigorous progress in the development of linear-time sequence models, and the discussion here only captures a portion of them. We anticipate a growing space of unified frameworks, improved understanding, and new generalizations as the development of these models continually evolves.

## **6 Conclusion And Future Work**

We introduce Mamba-3, a state space model with several methodological improvements over prior SSMs: a more powerful recurrence via exponential-trapezoidal discretization; improved expressivity through complex-valued state transitions; and higher inference efficiency and modeling abilities with a MIMO formulation. The base SISO version of Mamba-3 delivers strong language modeling results, both standalone and in interleaved hybrid architectures, and advances the Pareto frontier on the performance-efficiency tradeoff over prior linear sequence models. The MIMO version trades off slower training for even stronger modeling power, while maintaining competitive inference efficiency compared to Mamba-2. Put together, the techniques in Mamba-3 show simple and theoretically motivated improvements from the state space model viewpoint, and open up new directions and design principles for efficient sequence models.

## **Acknowledgments.**

We gratefully acknowledge the support of the Schmidt Sciences AI2050 fellowship, the Google ML and Systems Junior Faculty Awards, the Google Research Scholar program, Princeton Language and Intelligence (PLI), Together AI, and Cartesia AI. KL is supported by the NSF GRFP under Grant DGE2140739. We also thank Sukjun Hwang and Gaurav Ghosal for helpful feedback and discussions.

## **References**

- [1] Zeyuan Allen-Zhu. “Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers”. In: _SSRN Electronic Journal_ (May 2025). https://ssrn.com/abstract=5240330.

- [2] Anthropic. _Introducing Claude Opus 4.6_ . Feb. 2026. url: https://www.anthropic.com/news/claude-opus-4-6 (visited on 02/17/2026).

- [3] Aryaman Arora, Neil Rathi, Nikil Roashan Selvam, Róbert Csordás, Dan Jurafsky, and Christopher Potts. _Mechanistic evaluation of Transformers and state space models_ . 2025. arXiv: 2505.15105 [cs.CL]. url: https://arxiv.org/ abs/2505.15105.

- [4] Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, and Christopher Ré. _Simple linear attention language models balance the recall-throughput tradeoff_ . 2025. arXiv: 2402.18668 [cs.CL]. url: https://arxiv.org/abs/2402.18668.

- [5] Simran Arora, Aman Timalsina, Aaryan Singhal, Benjamin Spector, Sabri Eyuboglu, Xinyi Zhao, Ashish Rao, Atri Rudra, and Christopher Ré. _Just read twice: closing the recall gap for recurrent language models_ . 2024. arXiv: 2407. 05483 [cs.CL]. url: https://arxiv.org/abs/2407.05483.

- [6] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. _Neural Machine Translation by Jointly Learning to Align and Translate_ . 2014. arXiv: 1409.0473 [cs.CL]. url: https://arxiv.org/abs/1409.0473.

- [7] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. _PIQA: Reasoning about Physical Commonsense in Natural Language_ . 2019. arXiv: 1911.11641 [cs.CL]. url: https://arxiv.org/abs/1911.11641.

- [8] Loïc Cabannes, Maximilian Beck, Gergely Szilvasy, Matthijs Douze, Maria Lomeli, Jade Copet, Pierre-Emmanuel Mazaré, Gabriel Synnaeve, and Hervé Jégou. _Short window attention enables long-term memorization_ . 2025. arXiv: 2509.24552 [cs.LG]. url: https://arxiv.org/abs/2509.24552.

- [9] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. _Rethinking Attention with Performers_ . 2022. arXiv: 2009.14794 [cs.LG]. url: https://arxiv.org/abs/2009.14794.

- [10] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. _Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge_ . 2018. arXiv: 1803.05457 [cs.AI]. url: https://arxiv.org/abs/1803.05457.

- [11] Tri Dao and Albert Gu. _Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality_ . 2024. arXiv: 2405.21060 [cs.LG]. url: https://arxiv.org/abs/2405.21060.

- [12] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. _DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs_ . 2019. arXiv: 1903.00161 [cs.CL]. url: https://arxiv.org/abs/1903.00161.

- [13] Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Ré. _Hungry Hungry Hippos: Towards Language Modeling with State Space Models_ . 2023. arXiv: 2212.14052 [cs.LG]. url: https://arxiv.org/ abs/2212.14052.

- [14] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. _The Language Model Evaluation Harness_ . Version v0.4.3. July 2024. doi: 10.5281/zenodo.12608602. url: https://zenodo.org/records/12608602.

- [15] Aaron Grattafiori et al. _The Llama 3 Herd of Models_ . 2024. arXiv: 2407.21783 [cs.AI]. url: https://arxiv.org/ abs/2407.21783.

- [16] Riccardo Grazzi, Julien Siems, Simon Schrodi, Thomas Brox, and Frank Hutter. _Is Mamba Capable of In-Context Learning?_ 2024. arXiv: 2402.03170 [cs.LG]. url: https://arxiv.org/abs/2402.03170.

- [17] Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, and Massimiliano Pontil. _Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues_ . 2025. arXiv: 2411.12537 [cs.LG]. url: https:// arxiv.org/abs/2411.12537.

- [18] Albert Gu and Tri Dao. _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_ . 2024. arXiv: 2312.00752 [cs.LG]. url: https://arxiv.org/abs/2312.00752.

- [19] Albert Gu, Karan Goel, and Christopher Ré. _Efficiently Modeling Long Sequences with Structured State Spaces_ . 2022. arXiv: 2111.00396 [cs.LG]. url: https://arxiv.org/abs/2111.00396.

- [20] Albert Gu, Ankit Gupta, Karan Goel, and Christopher Ré. “On the Parameterization and Initialization of Diagonal State Space Models”. In: _arXiv preprint arXiv:2206.11893_ (2022). url: https://arxiv.org/abs/2206.11893.

- [21] Ankit Gupta, Albert Gu, and Jonathan Berant. _Diagonal State Spaces are as Effective as Structured State Spaces_ . 2022. arXiv: 2203.14343 [cs.LG]. url: https://arxiv.org/abs/2203.14343.

- [22] Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. _Query-Key Normalization for Transformers_ . 2020. arXiv: 2010.04245 [cs.CL]. url: https://arxiv.org/abs/2010.04245.

- [23] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. _RULER: What’s the Real Context Size of Your Long-Context Language Models?_ 2024. arXiv: 2404.06654 [cs.CL]. url: https://arxiv.org/abs/2404.06654.

- [24] Jiaxi Hu, Yongqi Pan, Jusen Du, Disen Lan, Xiaqiang Tang, Qingsong Wen, Yuxuan Liang, and Weigao Sun. _Comba: Improving Bilinear RNNs with Closed-loop Control_ . 2025. arXiv: 2506.02475 [cs.LG]. url: https://arxiv.org/ abs/2506.02475.

- [25] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. _TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension_ . 2017. arXiv: 1705.03551 [cs.CL]. url: https://arxiv.org/abs/ 1705.03551.

- [26] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. _Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention_ . 2020. arXiv: 2006.16236 [cs.LG]. url: https://arxiv.org/abs/ 2006.16236.

- [27] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, Weizhou Liu, Yanru Chen, Weixin Xu, Longhui Yu, Yejie Wang, Yu Fan, Longguang Zhong, Enming Yuan, Dehao Zhang, Yizhi Zhang, T. Y. Liu, Haiming Wang, Shengjun Fang, Weiran He, Shaowei Liu, Yiwei Li, Jianlin Su, Jiezhong Qiu, Bo Pang, Junjie Yan, Zhejun Jiang, Weixiao Huang, Bohong Yin, Jiacheng You, Chu Wei, Zhengtao Wang, Chao Hong, Yutian Chen, Guanduo Chen, Yucheng Wang, Huabin Zheng, Feng Wang, Yibo Liu, Mengnan Dong, Zheng Zhang, Siyuan Pan, Wenhao Wu, Yuhao Wu, Longyu Guan, Jiawen Tao, Guohong Fu, Xinran Xu, Yuzhi Wang, Guokun Lai, Yuxin Wu, Xinyu Zhou, Zhilin Yang, and Yulun Du. _Kimi Linear: An Expressive, Efficient Attention Architecture_ . 2025. arXiv: 2510.26692 [cs.CL]. url: https://arxiv.org/abs/2510.26692.

- [28] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. “Natural Questions: A Benchmark for Question Answering Research”. In: _Transactions of the Association for Computational Linguistics_ 7 (2019). Ed. by Lillian Lee, Mark Johnson, Brian Roark, and Ani Nenkova, pp. 452–466. doi: 10.1162/tacl_a_00276. url: https: //aclanthology.org/Q19-1026/.

- [29] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. _Efficient Memory Management for Large Language Model Serving with PagedAttention_ . 2023. arXiv: 2309.06180 [cs.LG]. url: https://arxiv.org/abs/2309.06180.

- [30] Baolin Li, Yankai Jiang, Vijay Gadepally, and Devesh Tiwari. _LLM Inference Serving: Survey of Recent Advances and Opportunities_ . 2024. arXiv: 2407.12391 [cs.DC]. url: https://arxiv.org/abs/2407.12391.

- [31] Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, and Chunting Zhou. _Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length_ . 2024. arXiv: 2404.08801 [cs.LG]. url: https://arxiv.org/abs/2404.08801.

- [32] William Merrill, Jackson Petty, and Ashish Sabharwal. _The Illusion of State in State-Space Models_ . 2025. arXiv: 2404. 08819 [cs.LG]. url: https://arxiv.org/abs/2404.08819.

- [33] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. _Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering_ . 2018. arXiv: 1809.02789 [cs.CL]. url: https://arxiv.org/abs/ 1809.02789.

- [34] NVIDIA. _NVIDIA H100 Tensor Core GPU White Paper_ . Tech. rep. NVIDIA, 2022. url: https://resources.nvidia. com/en-us-hopper-architecture/nvidia-h100-tensor-c.

- [35] NVIDIA et al. _Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models_ . 2025. arXiv: 2504. 03624 [cs.CL]. url: https://arxiv.org/abs/2504.03624.

- [36] OpenAI. _Introducing GPT-5.3-Codex_ . Feb. 2026. url: https://openai.com/index/introducing-gpt-5-3-codex/ (visited on 02/17/2026).

- [37] Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. _Resurrecting Recurrent Neural Networks for Long Sequences_ . 2023. arXiv: 2303.06349 [cs.LG]. url: https: //arxiv.org/abs/2303.06349.

- [38] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. _The LAMBADA dataset: Word prediction requiring a broad discourse context_ . 2016. arXiv: 1606.06031 [cs.CL]. url: https://arxiv.org/abs/1606.06031.

- [39] Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. _The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale_ . 2024. arXiv: 2406.17557 [cs.CL]. url: https://arxiv.org/abs/2406.17557.

- [40] Pranav Rajpurkar, Jian Zhang, and Percy Liang. “Know What You Don’t Know: Unanswerable Questions for SQuAD”. In: _ACL 2018_ . 2018.

- [41] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. _WinoGrande: An Adversarial Winograd Schema Challenge at Scale_ . 2019. arXiv: 1907.10641 [cs.CL]. url: https://arxiv.org/abs/1907.10641.

- [42] Yash Sarrof, Yana Veitsman, and Michael Hahn. _The Expressive Capacity of State Space Models: A Formal Language Perspective_ . 2024. arXiv: 2405.17394 [cs.CL]. url: https://arxiv.org/abs/2405.17394.

- [43] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. _Linear Transformers Are Secretly Fast Weight Programmers_ . 2021. arXiv: 2102.11174 [cs.LG]. url: https://arxiv.org/abs/2102.11174.

- [44] Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. _Simplified State Space Layers for Sequence Modeling_ . 2023. arXiv: 2208.04933 [cs.LG]. url: https://arxiv.org/abs/2208.04933.

- [45] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. _Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters_ . 2024. arXiv: 2408.03314 [cs.LG]. url: https://arxiv.org/abs/2408. 03314.

- [46] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. _RoFormer: Enhanced Transformer with Rotary Position Embedding_ . 2023. arXiv: 2104.09864 [cs.CL]. url: https://arxiv.org/abs/2104.09864.

- [47] Endre Süli and David F. Mayers. _An Introduction to Numerical Analysis_ . Cambridge University Press, 2003. [48] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. _Learning to (Learn at Test Time): RNNs with Expressive Hidden States_ . 2025. arXiv: 2407.04620 [cs.LG]. url: https://arxiv.org/abs/2407.04620.

- [49] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. _Retentive Network: A Successor to Transformer for Large Language Models_ . 2023. arXiv: 2307.08621 [cs.CL]. url: https: //arxiv.org/abs/2307.08621.

- [50] Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, and Yu Sun. _End-to-End Test-Time Training for Long Context_ . 2025. arXiv: 2512.23675 [cs.LG]. url: https://arxiv.org/abs/2512.23675.

- [51] Tencent Hunyuan Team et al. _Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought_ . 2025. arXiv: 2505.15431 [cs.CL]. url: https://arxiv.org/abs/2505. 15431.

- [52] M. Tenenbaum and H. Pollard. _Ordinary Differential Equations: An Elementary Textbook for Students of Mathematics, Engineering, and the Sciences_ . Dover Books on Mathematics. Dover Publications, 1985. isbn: 9780486649405. url: https://books.google.com/books?id=iU4zDAAAQBAJ.

- [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. “Attention is all you need”. In: _Advances in neural information processing systems_ . 2017, pp. 5998–6008. url: http://arxiv.org/abs/1706.03762.

- [54] Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, and Bryan Catanzaro. _An Empirical Study of Mamba-based Language Models_ . 2024. arXiv: 2406.07887 [cs.LG]. url: https://arxiv.org/abs/2406.07887.

- [55] Ke Alexander Wang, Jiaxin Shi, and Emily B. Fox. _Test-time regression: a unifying framework for designing sequence models with associative memory_ . 2025. arXiv: 2501.12352 [cs.LG]. url: https://arxiv.org/abs/2501.12352.

- [56] Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. _Small-scale proxies for large-scale Transformer training instabilities_ . 2023. arXiv: 2309.14322 [cs.LG]. url: https://arxiv.org/abs/2309.14322.

- [57] Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. _Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models_ . 2025. arXiv: 2408.00724 [cs.AI]. url: https://arxiv.org/abs/2408.00724.

- [58] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. _Qwen3 Technical Report_ . 2025. arXiv: 2505.09388 [cs.CL]. url: https://arxiv.org/abs/2505.09388.

- [59] Bowen Yang, Bharat Venkitesh, Dwarak Talupuru, Hangyu Lin, David Cairuz, Phil Blunsom, and Acyr Locatelli. _Rope to Nope and Back Again: A New Hybrid Attention Strategy_ . 2025. arXiv: 2501.18795 [cs.CL]. url: https: //arxiv.org/abs/2501.18795.

- [60] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. _Gated Delta Networks: Improving Mamba2 with Delta Rule_ . 2025. arXiv: 2412.06464 [cs.CL]. url: https://arxiv.org/abs/2412.06464.

- [61] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. _Gated Linear Attention Transformers with Hardware-Efficient Training_ . 2024. arXiv: 2312.06635 [cs.LG]. url: https://arxiv.org/abs/2312.06635.

- [62] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. _Parallelizing Linear Transformers with the Delta Rule over Sequence Length_ . 2025. arXiv: 2406.06484 [cs.LG]. url: https://arxiv.org/abs/2406.06484.

- [63] Annan Yu and N. Benjamin Erichson. _Block-Biased Mamba for Long-Range Sequence Processing_ . 2025. arXiv: 2505. 09022 [cs.LG]. url: https://arxiv.org/abs/2505.09022.

- [64] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. _HellaSwag: Can a Machine Really Finish Your Sentence?_ 2019. arXiv: 1905.07830 [cs.CL]. url: https://arxiv.org/abs/1905.07830.

- [65] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, and Hao Tan. _Test-Time Training Done Right_ . 2025. arXiv: 2505.23884 [cs.LG]. url: https://arxiv.org/abs/ 2505.23884.

## **A Exponential-Trapezoidal Discretization**

**Proposition 5** (Variation of Constants (Tenenbaum and Pollard 1985)) **.** _Consider the linear SSM_

**==> picture [120 x 12] intentionally omitted <==**

_where_ _**h**_ ( _𝑡_ ) ∈ R _[𝑁] , 𝐴_ ( _𝑡_ ) ∈ R _is a scalar decay, and_ _**B**_ ( _𝑡_ ) _𝑥_ ( _𝑡_ ) ∈ R _[𝑁] . For_ Δ _𝑡 discretized time grid 𝜏𝑡_ = _𝜏𝑡_ −1 + Δ _𝑡 , the hidden state satisfies equation (15), which can then be approximated to equation (16) with 𝑂_ (Δ _𝑡_[2][)] _[error.][The][approximation][of][the] remaining integral on the state-input can have varying error bounds depending on the method used: an example can be found in Appendix A.2._

**==> picture [394 x 53] intentionally omitted <==**

_Proof._ Starting from the initial linear SSM, an integrating factor _𝑧_ ( _𝑡_ ) � _𝑒_ ∫0 _𝑡_[−] _[𝐴]_[(] _[𝑠]_[)] _[𝑑𝑠]_ is applied to facilitate integration.

**==> picture [163 x 12] intentionally omitted <==**

Considering _𝑧_[′] ( _𝑡_ ) = − _𝐴_ ( _𝑡_ ) _𝑧_ ( _𝑡_ ); through rearranging the terms and integrating between the time grid [ _𝜏𝑡_ −1 _,𝜏𝑡_ ]

**==> picture [188 x 25] intentionally omitted <==**

results in

**==> picture [213 x 25] intentionally omitted <==**

which can be arranged in a more familiar form

**==> picture [251 x 25] intentionally omitted <==**

Substituting the integrating factor _𝑧_ ( _𝜏_ ) corresponds to

**==> picture [297 x 26] intentionally omitted <==**

We approximate the state-transition integral with a right-hand assumption where ∀ _𝑠_ ∈[ _𝜏𝑡_ −1 _,𝜏𝑡_ ] _,𝐴_ ( _𝑠_ ) � _𝐴_ ( _𝜏𝑡_ ) which we refer to as _𝐴𝑡_ ,

**==> picture [249 x 43] intentionally omitted <==**

incurring a local truncation error of order _𝑂_ (Δ _𝑡_[2][)][. Thus, we have approximated the exponential dynamics of the adjusted] underlying ODE and leave the state-input integral to be approximated with any host of methods. □

## **A.1 Exponential-Trapezoidal Discretization’s Mask Matrix**

_Proof._ When viewing the tensor contraction form, let us call _𝐶_ = ( _𝑇, 𝑁_ ) _, 𝐵_ = ( _𝑆, 𝑁_ ) _, 𝐿_ = ( _𝑇,𝑆_ ) _,𝑋_ = ( _𝑆, 𝑃_ ) based on the Mamba-2 paper. With this decomposition of our mask, we can view _𝐿_ = contract( _𝑇𝑍,𝑍𝑆_ → _𝑇𝑆_ )( _𝐿_ 1 _, 𝐿_ 2).

The original contraction can be seen as

**==> picture [173 x 10] intentionally omitted <==**

We can now view it as

contract( _𝑇𝑁,𝑆𝑁,𝑇𝐽, 𝐽𝑆,𝑆𝑃_ → _𝑇𝑃_ )( _𝐶, 𝐵, 𝐿_ 1 _, 𝐿_ 2 _,𝑋_ )

This can be broken into the following:

**==> picture [147 x 10] intentionally omitted <==**

**==> picture [163 x 40] intentionally omitted <==**

We can view this step: contract( _𝑍𝑆,𝑆𝑁𝑃_ → _𝑍𝑁𝑃_ )( _𝐿_ 2 _,𝑍_ ) as a convolution of size two applied on the state-input ( _𝐵,𝑋_ outer product) prior to the decay with the traditional SSD _𝐿_ = _𝐿_ 1 matrix. □

## **A.2 Exponential-Trapezoidal Discretization Error Rate**

**Standard assumptions.** We assume that: _𝐴_ ( _𝑡_ ) _,_ _**B**_ ( _𝑡_ ) _,𝑥_ ( _𝑡_ ) are bounded and _𝐶_[3] on each timestep, so that _𝑔_ ( _𝜏_ ) has three bounded derivatives; the map _**h**_ ↦→ _𝐴_ ( _𝑡_ ) _**h**_ + _**B**_ ( _𝑡_ ) _𝑥_ ( _𝑡_ ) is Lipschitz in _**h**_ which is true for linear systems; _𝜆𝑡_ lies in a bounded interval so that the update is zero-stable.

_Proof._ Let _**g**_ ( _𝜏_ ) � _𝑒_[(] _[𝑡][𝑘]_[−] _[𝜏]_[)] _[𝐴][𝑘]_ _**B**_ ( _𝜏_ ) _𝑥_ ( _𝜏_ ) denote the integrand in the second term of Proposition 5. Since _𝐴_ ( _𝑡_ ) _,_ _**B**_ ( _𝑡_ ) _,𝑥_ ( _𝑡_ ) are _𝐶_[3] on [ _𝑡𝑘_ −1 _,𝑡𝑘_ ], the function _𝑔_ has three bounded derivatives. A second-order Taylor expansion of _𝑔_ around _𝑡𝑘_ −1 gives us,

**==> picture [256 x 26] intentionally omitted <==**

Recall that the trapezoidal approximation to this integral is given by,

**==> picture [149 x 19] intentionally omitted <==**

**==> picture [481 x 69] intentionally omitted <==**

Hence, the error is given by:

**==> picture [288 x 26] intentionally omitted <==**

Under the assumption that _𝜆𝑡_ =[1] 2[+] _[ 𝑐][𝑡]_[Δ] _[𝑡]_[, where] _[ 𝑐][𝑡]_[=] _[ 𝑂]_[(][1][)][, then][1] 2[−] _[𝜆][𝑡]_[=][−] _[𝑐][𝑡]_[Δ] _[𝑡]_[=] _[ 𝑂]_[(][Δ] _[𝑡]_[)][and thus the][ Δ] _𝑡_[2][term is] _[ 𝑂]_[(][Δ] _𝑡_[3][)][.] Therefore,

**==> picture [111 x 25] intentionally omitted <==**

which yields an _𝑂_ (Δ _𝑡_[3][)][local truncation error.] □

## **A.3 Exponential-Trapezoidal Parameterization**

**Setting:** All runs use the Mamba-3 (SISO) 440M model trained at Chinchilla scale, with the other architectural and optimization hyperparameters being the same as in Table 3.

The default model uses a data-dependent gate _𝜆𝑡_ = _𝜎_ ( _𝑢𝑡_ ), where _𝑢𝑡_ is a learned projection of the current input token. In Table 8, we try different parameterizations for _𝜆𝑡_ and find that the default parameterization empirically performs the best. Hence, we choose the simpler default parameterization that does _not_ enforce _𝜆𝑡_ = 2[1][+] _[ 𝑂]_[(][Δ] _[𝑡]_[)][.]

Table 8: **Ablations on** _𝜆𝑡_ **parameterization in the exponential-trapezoidal update.**

|**Parameterization**|**Form of**_𝜆𝑡_|**ppl**↓|
|**Default**|_𝜎_(_𝑢𝑡_)|**15.72**|
|Fixed 1/2|1<br>2|15.76|
|No trapezoidal (Euler)|1|15.81|

## **B Complex SSM Proofs**

## **B.1 Proof of Proposition 2**

**Proposition 2** (Complex-to-Real SSM Equivalence) **.** _Consider a complex-valued SSM_

**==> picture [355 x 36] intentionally omitted <==**

_where_ _**h**_ ( _𝑡_ ) ∈ C _[𝑁]_[/][2] _,_ _**θ**_ ( _𝑡_ ) _,_ _**B**_ ( _𝑡_ ) _,_ _**B**_[ˆ] ( _𝑡_ ) _,_ _**C**_ ( _𝑡_ ) _,_ _**C**_[ˆ] ( _𝑡_ ) ∈ R _[𝑁]_[/][2] _, and 𝑥_ ( _𝑡_ ) _,𝐴_ ( _𝑡_ ) ∈ R _. Under exponential-Euler discretization, this system is equivalent to a real-valued SSM_

**==> picture [305 x 27] intentionally omitted <==**

_with state_ _**h** 𝑡_ ∈ R _[𝑁] , projections_

**==> picture [178 x 25] intentionally omitted <==**

_and a transition matrix_

**==> picture [307 x 25] intentionally omitted <==**

_Proof._ We first present the derivation for _𝑁_ = 2; the block-diagonal structure for general even _𝑁_ follows by grouping pairs of coordinates.

Let _ℎ𝑡_ + _𝑖ℎ_[ˆ] _𝑡_ denote the complexified hidden state, with parameters _𝐴_ ( _𝑡_ ) + _𝑖𝜃_ ( _𝑡_ ) and _𝐵_ ( _𝑡_ ) + _𝑖𝐵_[ˆ] ( _𝑡_ ) for the transition and input, respectively. By the variation of constants formula (Proposition 5), applying zero-order hold and Euler’s rule over a step [ _𝑡𝑘_ −1 _,𝑡𝑘_ ] gives

**==> picture [215 x 13] intentionally omitted <==**

Expanding the exponential,

**==> picture [179 x 19] intentionally omitted <==**

_ℎ𝑡_ so in real coordinates _**h** 𝑡_ = � _ℎ_ ˆ _𝑡_ � ∈ R[2] the recurrence becomes

**==> picture [223 x 44] intentionally omitted <==**

Stacking across _𝑁_ /2 such pairs yields the block-diagonal transition

**==> picture [216 x 25] intentionally omitted <==**

For the output,

**==> picture [187 x 26] intentionally omitted <==**

which defines the real projection _**C** 𝑡_ ∈ R _[𝑁]_ in the proposition. This proves the equivalence between complex SSM and the real block-diagonal system with rotations. □

## **B.2 Proof of Proposition 3**

**Proposition 3** (Complex SSM, Data-Dependent RoPE Equivalence) **.** _Under the notation established in Proposition 2, consider the real SSM defined in equation (9) unrolled for 𝑇 time-steps. The output of the above SSM is equivalent to that of a vanilla scalar transition matrix-based SSM (4) with a data-dependent rotary embedding applied on the_ _**B** ,_ _**C** components of the SSM, as defined by:_

**==> picture [398 x 32] intentionally omitted <==**

_where the matrix product represents right matrix multiplication, e.g.,_[�] _𝑖_[1] =0 _**[R]**[𝑖]_[=] _**[ R]**_[0] _**[R]**_[1] _[. We refer to the usage of a transformed] real-valued SSM to compute the complex SSM as the “RoPE trick.”_

_Proof._ Consider the SSM

**==> picture [348 x 13] intentionally omitted <==**

where (as in Proposition 3) _𝐴𝑡_ ∈ R is a scalar (so that _𝑒_[Δ] _[𝑡][𝐴][𝑡]_ is a scalar and commutes with rotations), and _**R** 𝑡_ is blockdiagonal orthogonal/unitary, hence _**R** 𝑡_[−][1] = _**R** 𝑡_[⊤][and the matrices] _**[ R]**[𝑖][,]_ _**[ R]**[𝑗]_[commute, i.e.] _**[ R]**[𝑖]_ _**[R]**[𝑗]_[=] _**[ R]**[𝑗]_ _**[R]**[𝑖]_[.] Unrolling the recurrence with the convention that an empty product is the identity,

**==> picture [315 x 28] intentionally omitted <==**

Thus

**==> picture [342 x 29] intentionally omitted <==**

Using its unitary property,

**==> picture [223 x 32] intentionally omitted <==**

Since _𝑒_[Δ] _[𝑠][𝐴][𝑠]_ are scalars, they commute with rotations; hence

**==> picture [360 x 66] intentionally omitted <==**

Define the rotated parameters _**C** 𝑡_ :=[��] _𝑠[𝑡]_ =0 _**[R]** 𝑠_[⊤] � _**C** 𝑡_ and _**B** 𝑖_ :=[��] _[𝑖] 𝑠_ =0 _**[R]** 𝑠_[⊤] � _**B** 𝑖 ._ Then,

**==> picture [316 x 31] intentionally omitted <==**

Equivalently, introducing the rotated state _**h**_[˜] _𝑡_ :=[��] _𝑠[𝑡]_ =0 _**[R]** 𝑠_[⊤] � _**h** 𝑡_ ,

**==> picture [341 x 30] intentionally omitted <==**

## **B.3 Proof of Proposition 4**

**Proposition 4** (Rotary Embedding Equivalence with Exponential-Trapezoidal Discretization) **.** _Discretizing a complex SSM with the exponential-trapezoidal rule (Proposition 1) yields the recurrence_

**==> picture [361 x 66] intentionally omitted <==**

_Here,_ _**R** 𝑡 is the block-diagonal rotation matrix defined in Proposition 2._

_Proof._ We begin from the complex SSM (as in Prop. 2)

**==> picture [233 x 40] intentionally omitted <==**

where _𝐴_ ( _𝑡_ ) ∈ R is a scalar and _**θ**_ ( _𝑡_ ) _,_ _**B**_ ( _𝑡_ ) _,_ _**B**_[ˆ] ( _𝑡_ ) _,_ _**C**_ ( _𝑡_ ) _,_ _**C**_[ˆ] ( _𝑡_ ) ∈ R _[𝑁]_[/][2] .

Recall from Prop. 5,

**==> picture [396 x 69] intentionally omitted <==**

Applying Prop. 1 to the above integral, we get

where

**==> picture [219 x 12] intentionally omitted <==**

Since _𝑒_[Δ] _[𝑡]_[(] _[𝐴][𝑡]_[+] _[𝑖]_ _**[θ]**[𝑡]_[)] = _𝛼𝑡 𝑒[𝑖]_[Δ] _[𝑡]_ _**[θ]**[𝑡]_ and as shown in Prop. 2, multiplication by _𝑒[𝑖]_[Δ] _[𝑡]_ _**[θ]**[𝑡]_ is a block-diagonal rotation in real coordinates, we get the real _𝑁_ -dimensional recurrence

**==> picture [490 x 87] intentionally omitted <==**

We define the following.

**==> picture [284 x 30] intentionally omitted <==**

Left-multiplying equation (25) by[�] _𝑠[𝑡]_ =0 _**[R]** 𝑠_[⊤][and using] _**[ R]** 𝑡_[⊤] _**[R]**[𝑡]_[=] _[ 𝐼]_[,]

**==> picture [167 x 31] intentionally omitted <==**

This is a vanilla scalar-transition SSM with data-dependent rotary embeddings absorbed into _**B** ,_ _**C**_ via cumulative products of _**R** 𝑠_[⊤][.] □

## **C MIMO for Mamba-3**

**Mamba with MIMO.** With a given batch, head, and sequence position _𝑡_ , consider the input _**U** 𝑡_ ∈ R _[𝐷]_ . Also denote _𝑃, 𝑅_ ∈ N as the head dimension and MIMO rank, respectively. We first obtain SSM parameters via a set of projections defined in terms of tensor contraction notation as follows:

**==> picture [388 x 26] intentionally omitted <==**

where _**W** 𝐵,_ _**W** 𝐶,_ _**W** 𝑋_[′] _,_ _**W** 𝑋_ are model parameters. Additionally, we obtain the residual gate term _**Z** 𝑡_ in the same manner as _**X** 𝑡_ with weights _**W** 𝑍_[′] and _**W** 𝑍_ . This parameterization is used to prevent the parameter count from increasing by a factor of _𝑅_ .

The state update and the SSM output are then computed via the following MIMO SSM:

**==> picture [255 x 12] intentionally omitted <==**

Intermediate output _**Y** 𝑡_[′][is obtained by the residual function] _[ 𝜙]_[,] _**[Y]** 𝑡_[′][←] _[𝜙]_[(] _**[Y]**[𝑡][,]_ _**[ Z]**[𝑡]_[)][,][where] _[ 𝜙]_[(] _**[Y]**[𝑡][,]_ _**[ Z]**[𝑡]_[)][:][=] _**[Y]**[𝑡]_[⊙][SiLU][(] _**[Z]**[𝑡]_[)][in] our case. Finally, the layer output _**O** 𝑡_ ∈ R _[𝐷]_ is computed via the following down projections:

**==> picture [371 x 11] intentionally omitted <==**

This formulation enhances the existing Mamba-3 architecture by providing a lightweight parameterization that transforms the set of independent SISO SSMs within each head into a set of MIMO SSMs.

**MIMO Parameter Matching.** The MIMO variant of Mamba-3 incurs additional parameters compared to its SISO counterpart. We therefore reduce the hidden dimension of the MLP layers to parameter match the SISO variants as follows:

|**Model**|180M|440M|880M|1_._5B|
|---|---|---|---|---|
|SISO MLP dim|1,500|2,048|3,072|4,096|
|MIMO MLP dim (_𝑅_=4)|1,264|1,792|2,800|3,824|

## **D Experimental Details**

**Language Modeling.** Our pretraining procedures follow those of Dao and Gu (2024)’s section D.2. All models at each scale follow the same procedure and were trained with bfloat16. The Mamba family of models were trained using the standard expand factor of 2 and a _𝑑_ state of 128 and head dimension of 64. The Transformer baselines follow Dao and Gu (2024), and the GDN baselines follow (S. Yang, Kautz, and Hatamizadeh 2025) where _𝑞,𝑘_ dim = 128 _, 𝑣_ dim = 256. We utilize the Llama-3.1 tokenizer (Grattafiori et al. 2024) for all models.

We utilize LM Evaluation Harness (Gao et al. 2024) to test the zero-shot language modeling capabilities of our pretrained model on LAMBADA (OpenAI version) (Paperno et al. 2016), HellaSwag (Zellers et al. 2019), PIQA (Bisk et al. 2019), Arc-Easy/Arc-Challenge (Clark et al. 2018), WinoGrande (Sakaguchi et al. 2019), and OpenBookQA (Mihaylov et al. 2018).

**Real-World and Synthetic Retrieval.** For our real-world retrieval tasks, we evaluate on the common suite consisting of SWDE (S. Arora, Eyuboglu, et al. 2025), SQuAD (Rajpurkar, J. Zhang, and Liang 2018), FDA (S. Arora, Eyuboglu, et al. 2025), TriviaQA (Joshi et al. 2017), NQ (Kwiatkowski et al. 2019), and DROP (Dua et al. 2019). We utilize the cloze-formatted version of the aforementioned tasks provided by S. Arora, Eyuboglu, et al. (2025) and S. Arora, Timalsina, et al. (2024), as the original datasets are in a question-answering format, making it challenging for solely pretrained models. All tasks

Table 9: Ablations of optional norm type (grouped vs default) and placement (pre- vs post-gate) on pretrained hybrid Mamba-3 SISO models at the 1.5B scale. All models have BCNorm. No additional norm demonstrates the strongest incontext retrieval performance on average, while pre-gate, grouped RMS results in the best performance on synthetic retrieval, especially on lengths longer than its training context.

|**Mamba-3 Norm Type**<br>Context Length|LM Avg.<br>—|SWDE<br>SQD.<br>FDA<br>TQA<br>NQ<br>Drop<br>2048|NIAH-Single-1<br>1024<br>2048<br>4096|NIAH-Single-2<br>1024<br>2048<br>4096|NIAH-Single-3<br>1024<br>2048<br>4096|
|---|---|---|---|---|---|
|No Norm<br>Post-Gate Default RMS<br>Pre-Gate Default RMS<br>Post-Gate Grouped RMS<br>Pre-Gate Grouped RMS|56_._4<br>56_._5<br>55_._9<br>56_._2<br>56_._1|58_._5<br>47_._0<br>65_._9<br>64_._8<br>33_._4<br>27_._0<br><br>54_._5<br>46_._6<br>61_._9<br>65_._4<br>31_._9<br>29_._2<br><br>55_._4<br>46_._9<br>67_._3<br>65_._4<br>33_._0<br>28_._1<br><br>51_._4<br>46_._7<br>56_._8<br>64_._2<br>30_._4<br>27_._6<br><br>58_._6<br>47_._3<br>52_._4<br>65_._7<br>33_._3<br>28_._5<br>|100_._0<br>100_._0<br>36_._2<br><br>100_._0<br>100_._0<br>100_._0<br><br>100_._0<br>100_._0<br>86_._2<br><br>100_._0<br>100_._0<br>79_._4<br><br>100_._0<br>100_._0<br>100_._0<br>|100_._0<br>100_._0<br>9_._4<br>100_._0<br>99_._8<br>49_._2<br>100_._0<br>100_._0<br>97_._8<br>100_._0<br>100_._0<br>65_._8<br>100_._0<br>100_._0<br>96_._0|99_._8<br>100_._0<br>8_._8<br>87_._6<br>94_._0<br>62_._0<br>99_._2<br>97_._8<br>90_._2<br>93_._8<br>97_._0<br>9_._6<br>99_._8<br>97_._2<br>56_._8|

were truncated to match the training context length. The synthetic NIAH tasks (Hsieh et al. 2024) were also run with LM Evaluation Harness.

**State-Tracking Synthetics.** Training follows a sequence length curriculum that sets the minimum length to 3 and progresses the maximum length from 40 to 160. Final models are evaluated at 256 length. Each curriculum runs for 10[4] steps with batch size 256. We use one-layer models for Parity and three-layer models for Modular-arithmetic tasks. The state size is chosen to be 64, and we sweep _𝑑_ model ∈{32 _,_ 64} and 8 learning rates logarithmically spaced between 10[−][4] and 10[−][2] , reporting the best validation accuracy.

## **E Additional Experimental Results**

## Context Length Extrapolation

**==> picture [353 x 200] intentionally omitted <==**

Train length = 2K<br>10.8 Gated DeltaNet<br>Mamba-2<br>Mamba-3<br>10.6<br>10.4<br>10.2<br>10.0<br>1K 2K 4K 8K 16K 32K<br>Context length<br>Perplexity<br>**----- End of picture text -----**<br>

Figure 4: Pretrained 1.5B models’ performance on the held-out FineWeb-Edu test set at varying context lengths. Mamba-3 exhibits strong length extrapolation while Mamba-2 falters at longer contexts.

**==> picture [368 x 198] intentionally omitted <==**

Validation Perplexity for 1B Pretraining Runs<br>14.0<br>13.5<br>13.0<br>12.5<br>12.0<br>11.5<br>GatedDeltaNet<br>11.0<br>Mamba-2<br>Mamba-3 SISO<br>10.5<br>Mamba-3 MIMO<br>10.0<br>20000 30000 40000 50000 60000 70000 80000 90000<br>Global Step<br>Perplexity<br>**----- End of picture text -----**<br>

Figure 5: Mamba-3 demonstrates better pretraining performance compared to strong baselines like Mamba-2 and Gated DeltaNet. These are the validation perplexity on FineWeb-Edu of our fully pretrained 1.5B models.

We also compare the effectiveness of state size usage of Mamba variants to a Gated DeltaNet baseline in Figure 6. We highlight the difficulty of directly comparing GDN versus Mamba-style models due to the differing head structure (multihead for GDN compared to multi-value for Mamba). Our experiments hold GDN’s _𝑣𝑒𝑥𝑝𝑎𝑛𝑑_ to 2 and decrease the head dimension accordingly to vary the relative total state size. Similar to Figure 3, we train 440M models to 2× Chinchilla tokens (40× token-to-parameter ratio) and sweep across _𝑑_ state = {32 _,_ 64 _,_ 128} for the Mamba models and _𝑑_ head dim = {32 _,_ 64 _,_ 128} for GDN. We parameter match all models.

**==> picture [368 x 185] intentionally omitted <==**

Relative Total State Size vs Pretraining Perplexity<br>15.0<br>Mamba-2<br>Mamba-3<br>Mamba-3 MIMO<br>14.8<br>Gated DeltaNet<br>14.6<br>10 [5]<br>Relative Total State Size<br>Pretraining Perplexity<br>**----- End of picture text -----**<br>

Figure 6: Exploration of state size (inference speed proxy) versus pretraining perplexity (performance proxy). Mamba-3 and Mamba-3 MIMO continue to set the Pareto frontier.

## **F Architecture Ablations**

We explore our model architecture ablations in this section. All models are trained at the 440M scale to Chinchilla optimal number of tokens (20× tokens to parameters) with the same experimental procedures as our pretrained models as covered in Appendix D unless otherwise stated.

_**B** ,_ _**C**_ **Bias Parameterization.** The Mamba-3 model’s separate _𝐵_ and _𝐶_ biases are head-specific and channel-wise and added to both _**B**_ and _**C**_ after the QK-Norm. While the biases in the final Mamba-3 model are trainable, data-independent parameters and initialized to all ones, we explore various bias parameterizations in Table 10a. We find our models are not very sensitive to the initialization of the biases as long as they are positive. We choose the all-ones initialization due to its simplicity.

We also explore the impact of removing the _𝐵_ or _𝐶_ bias on performance in Table 10b (bias is initialized with our default parameterization when utilized). Unlike in Yu and Erichson (2025), which finds that _𝐵_ bias by itself is able to improve performance on Mamba-1, our experiments find that only having _𝐵_ bias hurts performance slightly and that _𝐵_ and _𝐶_ biases have synergistic properties.

|||
|---|---|
|**Bias Init.**<br>**Trainable**<br>**ppl**↓<br>1.0<br>✓<br>15.72<br>0.0<br>✓<br>16.57<br>1.0<br>×<br>15.80<br>U(0_,_1)<br>✓<br>15.76<br>U(−1_,_1)<br>✓<br>16.07|**_B_ Bias**<br>**_C_ Bias**<br>**ppl**↓|
||×<br>×<br>16.52<br>✓<br>×<br>16.68<br>×<br>✓<br>15.98<br>✓<br>✓<br>15.69|

(a) Effect of parameterization of the _𝐵_ and _𝐶_ bias on model performance, measured by pretraining perplexity. We find our default initialization of all-ones (first row) provides the best performance, but performance is not sensitive as long as biases are positive.

(b) Applying a bias to both _𝐵_ and _𝐶_ leads to the best performance. Only applying _𝐵_ bias (Block-Biased (Yu and Erichson 2025) Mamba-3 variant) does not provide significant gains over the no-bias baseline.

Table 10: Ablations on _𝐵,𝐶_ bias initialization (left) and presence (right) for Mamba-3.

## **G Inference Kernel Latency Analysis**

## **G.1 Kernel Implementations and Fusion Structure**

In Table 6, we detail the DSL (Triton, TileLang, CuTe, PyTorch) and the fusion level of the kernels used in our latency analysis. For Mamba-2 and Gated DeltaNet (GDN), we directly use the publicly released Triton kernels from the respective authors. For Mamba-3, we implement new inference kernels with a comparable fusion structure: the forward SISO uses a Triton kernel fused with rotary position embeddings and the forward MIMO uses a TileLang kernel with the same fusion level while the decode path uses a CuTe kernel fused with gating and MIMO projection.

In Tables 11 and 12, we abbreviate IP = input projection, Conv = 1D convolution, Gate = gating, OP = output projection. Colors indicate implementation backend (Torch, Triton, TileLang, CuTe).

Table 11: Kernel DSL and fusion structure for **forward** (prefill) kernels.

|**Model (Forward)**|**Kernel DSL**|**Fusion Level**|
|Mamba-2|Triton|IP,Conv,SSM,Gate,OP|
|Gated DeltaNet|Triton|IP,Conv,Chunked Delta,Gate,OP|
|Mamba-3 (SISO)|Triton|IP,SSM+Rotary+Gate,OP|
|Mamba-3 (MIMO)|TileLang|IP,SSM+Rotary+Gate,OP|

## **G.2 Extended Prefill and Prefill+Decode Latency Measurements**

**Models.** We benchmark Mamba-3 1.5B (SISO), Mamba-2 1.5B, Gated DeltaNet 1.5B, and a strong Transformer baseline implemented via the vLLM engine (v0.11.0) with Llama-3.2 1B.[6] All recurrent models are trained at the 1.5B scale with

> 6https://huggingface.co/meta-llama/Llama-3.2-1B.

Table 12: Kernel DSL and fusion structure for **decode** kernels.

|**Model (Decode)**|**Kernel DSL**|**Fusion Level**|
|Mamba-2|Triton|IP,Conv,SSM,Gate,OP|
|Gated DeltaNet|Triton|IP,Conv,Recurrent Delta,Gate,OP|
|Mamba-3 (SISO)|CuTe + Triton|IP,Rotary,SSM+Gate,OP|
|Mamba-3 (MIMO)|CuTe + Triton|IP,Rotary,SSM+Gate,OP|

_𝑑_ model = 2048 and 24 layers. For Mamba variants we set state size as 128 and head dimension 64; for GDN we use QK head dimension as 128.

**Setting.** Sequence lengths were swept over _𝐿_ ∈{512 _,_ 1024 _,_ 2048 _,_ 4096 _,_ 16384} for prefill, with an equal number of tokens decoded. For all sequence lengths, we use a batch size of 128. To report vLLM numbers at sequence length 16384, we measure performance at the same sequence length with batch size 16. We then scale the result by a factor of 8 to approximate performance at batch size 128 since direct measurement at this setting exceeds GPU memory. This provides a reasonable estimate because each batch is processed independently by each SM on the GPU, so we expect performance of Transformer models to scale linearly with batch size. For recurrent models, when the size of input and output tensors exceeds GPU memory at sequence length 16384, we utilize a state passing approach that processes the sequence in two halves while propagating the hidden state between segments to avoid materializing the entire sequence at once. We use a single H100-SXM 80GB GPU and report wall-clock times (in seconds) over three repetitions.

We observe that (i) Mamba-3 adds minimal forward-pass cost, showing that the exponential-trapezoidal update, complex state tracking, and MIMO parameterization remain lightweight; (ii) decode latency is competitive across recurrent models; and (iii) recurrent mixers scale more gently with context length than vLLM Llama-3.2-1B, which grows much faster with _𝐿_ due to KV-cache overhead.
