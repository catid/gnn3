**==> picture [29 x 29] intentionally omitted <==**

# **Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces**

**Chloe Wang**[1 2] **Oleksii Tsepa**[1 2] **Jun Ma**[2 3 4] **Bo Wang**[1 2 3 4 5]

## **Abstract**

## **1. Introduction**

Attention mechanisms have been widely used to capture long-range dependencies among nodes in Graph Transformers. Bottlenecked by the quadratic computational cost, attention mechanisms fail to scale in large graphs. Recent improvements in computational efficiency are mainly achieved by attention sparsification with random or heuristic-based graph subsampling, which falls short in data-dependent context reasoning. State space models (SSMs), such as Mamba, have gained prominence for their effectiveness and efficiency in modeling long-range dependencies in sequential data. However, adapting SSMs to non-sequential graph data presents a notable challenge. In this work, we introduce GraphMamba, the first attempt to enhance long-range context modeling in graph networks by integrating a Mamba block with the input-dependent node selection mechanism. Specifically, we formulate graph-centric node prioritization and permutation strategies to enhance context-aware reasoning, leading to a substantial improvement in predictive performance. Extensive experiments on ten benchmark datasets demonstrate that GraphMamba outperforms state-of-the-art methods in long-range graph prediction tasks, with a fraction of the computational cost in both FLOPs and GPU memory consumption. The code and models are publicly available at https://github.com/ bowang-lab/Graph-Mamba.

> 1Department of Computer Science, University of Toronto, Toronto, Canada 2Vector Institute for Artificial Intelligence, Toronto, Canada[3] Peter Munk Cardiac Centre, University Health Network, Toronto, Canada[4] Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada[5] AI Hub, University Health Network, Toronto, Canada. Correspondence to: Bo Wang _<_ bowang@vectorinstitute.ai _>_ .

Graph modeling has been widely used to handle complex data structures and relationships, such as social networks (Fan et al., 2019), molecular interactions (Tsepa et al., 2023), and brain connectivity (Li et al., 2021). Recently, Graph Transformers have gained increasing popularity because of their strong capability in modeling long-range connections between nodes (Yun et al., 2019; Dwivedi & Bresson, 2012; Kreuzer et al., 2021a; Chen et al., 2022). The typical Graph Transformer attention allows each node in the graph to interact with all the other nodes (Vaswani et al., 2017). This serves as the complement to local message-passing approaches that primarily encode edge-based neighborhood topology (Kipf & Welling, 2016; Xu et al., 2018). To streamline the construction of Graph Transformers, GraphGPS devised a unified framework that combines an attention module, a message passing neural network (MPNN), and structural/positional encodings (SE/PE). These components collaboratively update node and edge embeddings for downstream prediction tasks. Such decoupled pipelines offer users great flexibility to incorporate various attention modules in a plug-and-play manner (Ramp´aˇsek et al., 2022).

Although Transformers demonstrate notable enhancements of modeling capabilities, their application to long sequences is hindered by the quadratic computational cost associated with attention mechanism. This limitation has prompted further research into linear-time attention approaches. For example, BigBird (Zaheer et al., 2020) and Performer (Choromanski et al., 2020) attempted to approximate the full attention with sparse attention or lower-dimensional matrices. However, designed for sequential inputs, BigBird does not generalize well to non-sequential inputs such as graphs, leading to performance deterioration in GraphGPS (Shirzad et al., 2023). Exphormer tailored such attention sparsification principles to graph input, by incorporating local connectivity defined by edges as local attention (Shirzad et al., 2023). These adaptations led to improved performance comparable to full graph attention.

However, approximating the full attention, or encoding all contexts, may not be ideal for long-range dependencies. In empirical observations, many sequence models do not improve with increasing context length (Gu & Dao, 2023).

Mamba, a selective state space model (SSM), was proposed to solve data-dependent context compression in sequence modeling (Gu & Dao, 2023). Instead of attention computation, Mamba inherits the construct of state space models that encode context using hidden states during recurrent scans. The selection mechanism allows control over which part of the input can flow into the hidden states, as part of the context that influence subsequent embedding updates. In graph modeling, this can be viewed as a data-dependent node selection process. By filtering relevant nodes at each step of recurrence and “attending” only to this selected context, Mamba helps achieve the same objectives as attention sparsification, serving as an alternative to random subsampling. Moreover, the Mamba module is optimized for linear time complexity and reduced memory, offering improved efficiency for large-graph training tasks. However, challenges present in effectively adapting Mamba designed for sequence modeling to non-sequential graph input.

**Contributions.** Motivated by the exceptional long-sequence modeling ability of Mamba, we propose Graph-Mamba to alleviate the high computational cost associated with Graph Transformers, as a data-dependent alternative for attention sparsification. In particular, this work presents the following key contributions:

- **Innovative graph network design:** Graph-Mamba represents a new type of graph network pioneering integration with selective state space models, which performs input-dependent node filtering and adaptive context selection. The selection mechanism captures long-range dependencies and improves on the existing subsampling-based attention sparsification techniques.

- **Adaptation of SSMs for non-sequential graph data:** We designed an elegant way to extend state space models to handle non-sequential graph data. Specifically, we introduced a node prioritization technique to prioritize important nodes for more access to context, and employed a permutation-based training recipe to minimize sequence-related biases.

- **Superior performance and efficiency:** Comprehensive experiments on ten public datasets demonstrate that Graph-Mamba not only outperforms baselines but also achieves linear-time computational complexity. Remarkably, Graph-Mamba reduces GPU memory consumption by up to 74% on large graphs, highlighting its efficiency in long-range graph datasets.

## **2. Related Work**

## **2.1. Graph Neural Networks**

Graph Neural Networks (GNNs) leverage message passing as a key mechanism for graph modeling, enabling

nodes to communicate and iteratively aggregate information from their neighbors. Graph Convolutional Networks (GCN) (Kipf & Welling, 2017; Defferrard et al., 2017) pioneered GNNs, influencing subsequent works like GraphSage (Hamilton et al., 2018), GIN (Xu et al., 2018), GAT (Velickoviˇ c et al.´ , 2018), and GatedGCN (Bresson & Laurent, 2018). Despite the significance in aggregating node features based on graph topology, MPNNs have limited expressive power upper bounded by the 1-dimensional Weisfeiler-Lehman (1-WL) isomorphism test (Xu et al., 2018). Additionally, aggregated node features are prone to over-smoothing in the local neighborhood (Alon & Yahav, 2021; Topping et al., 2022).

## **2.2. Graph Transformers**

Transformers with attention mechanism have achieved unprecedented success in various domains, including natural language processing (NLP) (Vaswani et al., 2017; Kalyan et al., 2021) and computer vision (D’Ascoli et al., 2021; Guo et al., 2022; Dosovitskiy et al., 2021). Graph Transformers typically compute full attention, allowing each node to attend to all others, regardless of the edge connectivity. This enables Graph Transformers to effectively capture long-range dependencies, avoiding over-aggregation in local neighborhoods like MPNNs. However, full attention, with its _O_ ( _N_[2] ) complexity, fails to scale in large graphs.

Analogous to positional embeddings in transformers for NLP, the first Graph Transformer (Dwivedi & Bresson, 2021) introduced graph Laplacian eigenvectors as node PE. Subsequently, SAN (Kreuzer et al., 2021b) incorporated invariant PE aggregation, integrating conditional attention for both real and virtual graph edges. Concurrently, Graphormer (Ying et al., 2021) integrated relative PE into the attention mechanism using centrality and spatial encodings. GraphiT (Mialon et al., 2021) utilized relative PE based on diffusion kernels to simulate the attention mechanism. Lastly, GraphTrans (Wu et al., 2022) proposed a two-stage architecture, employing a graph transformer on local embeddings derived from MPNNs.

## **2.3. GraphGPS**

GraphGPS (Rampa´sekˇ et al., 2022) employed a modular framework that integrates SE, PE, MPNN, and a graph transformer. Users have the flexibility to choose the methods for each component within this framework. Given an input graph, GraphGPS computes SE and PE, concatenates them with node and edge embeddings, and passes these embeddings into the GPS layers. In the GPS layers, a graph transformer and MPNN collaboratively update the node and edge embeddings. The GraphGPS framework allows the replacement of fully-connected Transformer attention with its sparse alternatives, resulting in _O_ ( _N_ + _E_ ) complexity.

## **2.4. Sparse Graph Attention**

BigBird (Zaheer et al., 2020) and Performer (Choromanski et al., 2020) are the two sparse attention methods supported by GraphGPS. Performer improved computation efficiency by using lower-dimensional positive orthogonal random features to approximate softmax kernels in regular attention. BigBird employed graph subsampling and sequence heuristics to approximate full attention, combining randomly subsampled attention, local attention among adjacent tokens, and global attention with global tokens (Zaheer et al., 2020). Randomly subsampled graphs, or expanders, are known to approximate the spectral properties of full graphs (Spielman & Teng, 2011; Yun et al., 2020). However, BigBird’s local attention is specifically designed for sequence input with a sliding window on adjacent tokens, making it unsuitable for modeling graph input. Exphormer (Shirzad et al., 2023) proposed a graph adaptation of BigBird by incorporating local neighborhood attention among neighbors defined by edges, and global attention connecting virtual nodes to all nodes, to expanders. These adaptations further improved model performance while benefiting from the linear complexity of sparse attention. However, the random node subsampling process suggests potential room for improvement. Incorporating methods that allow informed context selection could serve as further enhancement.

## **2.5. State Space Models**

General state space models involve recurrent updates over a sequence through hidden states. Implementations range from hidden Markov models to recurrent neural networks (RNNs) in deep learning. Utilizing a recurrent scan, SSM stores context in its hidden states, and updates the output by combining these hidden states with input. Structured state space models (S4) enhance computational efficiency with reparameterization (Gu et al., 2021), offering an efficient alternative to attention computation. Recent S4 variants for linear-time attention include H3 (Fu et al., 2022), Gated State Space (Mehta et al., 2022), Hyena (Nguyen et al., 2023), and RWKV (Peng et al., 2023). Mamba further introduces a data-dependent selection mechanism to S4 to capture long-range context with increasing sequence length (Gu & Dao, 2023). Notably, Mamba demonstrates lineartime efficiency in long-sequence modeling, and outperforms Transformers on various benchmarks. Mamba has also been successfully adapted for non-sequential input such as images on segmentation tasks to enhance long-range dependencies (Ma et al., 2024; Zhu et al., 2024; Liu et al., 2024).

## **3. Graph-Mamba**

Graph-Mamba employs a selective SSM to achieve inputdependent graph sparsification. In particular, we have designed a Graph-Mamba block (GMB) and incorporated it

into the popular GraphGPS framework, enabling fair comparisons with other graph attention implementations. GMB leverages the recurrent scan in sequence modeling with a selection mechanism to achieve two levels of graph sparsification. The first level involves the selection mechanism in Mamba module, which effectively filters relevant information within the long-range context. The second level is achieved through the proposed node prioritization approach, allowing important nodes in the graph to access more context. Consequently, these sequence modeling features present a promising avenue of combining datadependent and heuristic-informed selection for graph sparsification. Morever, Graph-Mamba implementation using the Mamba module ensures linear-time complexity, as an efficient alternative to dense graph attention.

To contextualize SSMs in graph modeling, we first review SSMs followed by the selection mechanism in Section 3.1 and 3.2. Next, we introduce the Graph-Mamba implementation in Section 3.3, and detail GMB’s specialized graph adaptation techniques in Section 3.4 and 3.5. Finally, we discuss the computational efficiency of GMB in Section 3.6.

## **3.1. Structured state space models for sequence modeling**

SSM is a type of sequence model that defines a linear Ordinary Differential Equation (ODE) to map input sequence _x_ ( _t_ ) _∈_ R _[N]_ to output sequence _y_ ( _t_ ) _∈_ R _[N]_ by a latent state _h_ ( _t_ ) _∈_ R _[N]_ :

**==> picture [169 x 27] intentionally omitted <==**

where _**A** ∈_ R _[N][×][N]_ and _**B** ,_ _**C** ∈_ R _[N]_ denote the state matrix, input matrix, and output matrix, respectively. To obtain the output sequence _y_ ( _t_ ) at time _t_ , we need to find _h_ ( _t_ ) which is difficult to solve analytically. In contrast, real-world data is usually discrete rather than continuous. As an alternative, we discretize the system Equation (1) as follows:

**==> picture [161 x 27] intentionally omitted <==**

where _**A**_ **[¯]** := exp(∆ _·_ _**A**_ ) and _**B**_ **[¯]** := (∆ _·_ _**A**_ ) _[−]_[1] (exp(∆ _·_ _**A**_ ) _− I_ ) _·_ ∆ _**B**_ are the discretized state parameters and ∆ is the discretization step size. SSMs have rich theoretical properties but suffer from high computational cost and numerical instability. Structured state space sequence models (S4) addressed these limitations by imposing structure on the state matrix _**A**_ based on HIPPO matrices, which significantly improved the performance and efficiency. In particular, S4 surpassed Transformers by a large margin on the Long Range Arena benchmark, which requires effective modeling of the long-range dependencies (Gu et al., 2021).

**==> picture [487 x 269] intentionally omitted <==**

A B<br>MPNN<br>Edge Embeddings Message Passing<br>Neural Network A A<br>(MPNN) + MLP B G B G<br>C C<br>Node Embeddings E F E F<br>Graph-Mamba D D<br>Block (GMB)<br>C D<br>Graph Flattening State Space Model (SSM) + Selection<br>A B C D E F G Input Node Embeddings Output Node Embeddings<br>Default Order G E B F A D C B A D C G E B F A D C<br>Node Prioritization Degree Low Degree High Degree Low Degree High Degree Low Degree High<br>E G A B F C D Mamba Module<br>Degree Low Degree High Linear Conv SSM<br>Linear<br>Permutation<br>Linear<br>G E B F A D C<br>Degree Low Degree High<br>**----- End of picture text -----**<br>

_Figure 1._ **Overview of Graph-Mamba architecture** , by incorporating GMB to replace the attention module in the GraphGPS framework. A) The GMB layer, an adaptation of GPS layer that combines an edge-based MPNN and a node-focused GMB to output updated node and edge embeddings. B) Graph-Mamba employs the GatedGCN model as default for MPNN. C) GMB’s specialized training recipe with node prioritization and permutation techniques that perform informed graph sparsification. D) The selection mechanism with Mamba module that facilitates input-dependent context filtering.

## **3.2. Graph-dependent selection mechanism**

S4 has demonstrated better suitability for modeling long sequences, but underperforms when content-aware reasoning is needed, attributed to its time-invariant nature. More specifically, _**A** ,_ _**B**_ , and _**C**_ are the same for all input tokens in a sequence. Mamba (Gu & Dao, 2023) addressed this issue by introducing the selection mechanism, allowing the model to adaptively select relevant information from the context. This can be achieved by simply making the SSM parameters _**B** ,_ _**C**_ , and ∆ as functions of the input _x_ . Furthermore, a GPU-friendly implementation is designed for efficient computing of the selection mechanism, which significantly reduces the number of memory IOs and avoids saving the intermediate states.

We use the reparameterized discretization step size ∆ as an example to illustrate the intuition behind Mamba’s selection mechanism. Revisiting Gu & Dao (2023)’s main Theorem, ∆ _t_ assumes a generalized role related to gating mechanism in RNN to facilitate input-dependent selection, where _gt_ = _σ_ ( _Linear_ ( _xt_ )) and _ht_ = (1 _− gt_ ) _ht−_ 1 + _gtxt_ , as detailed in Appendix C. Intuitively, the current input _xt_ is able to control the balance between current input and previous context _ht−_ 1 with _gt_ when updating the hidden state

_ht_ . This is achieved by parameterizing ∆ as a function of input in the discretization step to obtain data-dependent _**A**_ **[¯]** and _**B**_ **[¯]** , which acts as the main selection mechanism similar to gating in RNN. Additionally, the projection matrices _**B**_ and _**C**_ are parameterized as linear projections of the input _x_ , to further control how much _xt_ updates the hidden states and how much _ht_ influences the output _yt_

In graph learning, with nodes as input sequence, the selection mechanism allows the hidden states to update based on relevant nodes from prior sequence, gated by the current input node, and subsequently influencing the current node’s output embeddings. _gt_ ranges between 0 and 1, allowing the model to filter out irrelevant context entirely when needed. The ability to select and reset enables Mamba to distill relevant dependencies given long-range context, while minimizing the influence of unimportant nodes at each step of recurrence. It hence offers a context-aware alternative for sparsifying graph attention, by retaining relevant dependencies only in the long input sequences.

## **3.3. Graph-Mamba workflow**

Graph-Mamba incorporates Mamba’s selection mechanism from Section 3.2 into the GraphGPS framework. Figure 1

## **Algorithm 1** GMB Forward Pass

**Input:** Node embeddings _**X** ∈_ R _[L][×][D]_ ; Node heuristic _**H** ∈_ R _[L][×]_[1] . **Output:** Updated node embeddings _**X[′]** ∈_ R _[L][×][D]_ .

- 1: _**H[′]**_ = _**H**_ + _Noise_ (0 _,_ 1) _{_ Add noise to heuristic for node shuffling _}_

- 2: _**Isorted** ← Argsort_ ( _**H[′]**_ )

- 3: _**Ireverse** ← Argsort_ ( _**Isorted**_ )

|3:|**_Ireverse_** _←Argsort_(**_Isorted_**)|
|4:|**_Xsorted_** : (_L, D_)_←_**_X_[****_Isorted_]**_{_Sort nodes by node|
||heuristic_}_|
|5: <br>6: <br>7: <br>8: <br>9: <br>10: <br>11: <br>12:<br>13: <br>14:|**_Xnorm_** : (_L, D_) _←LayerNorm_(**_Xsorted_**) _{_Input<br>to SSM + Selection_}_<br> **_x_**: (_L, D′_)_←Linear_0(**_Xnorm_**)<br> **_x′_** : (_L, D′_)_←SiLU_(_Linear_1(**_Xnorm_**))<br> **_xSSM_**(_L, D′_)_←SiLU_(_Conv_1_d_(**_x_**))<br> **_B_** : (_L, N_)_←LinearB_(**_xSSM_**)<br> **_C_** : (_L, N_)_←LinearC_(**_xSSM_**)<br> **∆**: (_L, D′_)_←softplus_(_Linear_∆(**_xSSM_**))<br>**¯**<br>**_A_**: (_L, D′, N_)_←discretizeA_(**∆**_,_**_A_**)<br> **¯**<br>**_B_** : (_L, D′, N_)_←discretizeB_(**∆**_,_**_A_**_,_**_B_**)<br> **_y_** : (_L, D′_)_←SSM_( **¯**<br>**_A_**_,_ **¯**<br>**_B_**_,_**_C_**)(**_xSSM_**)|
|15: <br>16:|**_y′_** : (_L, D′_)_←_**_y_**_⊙_**_x′_**<br> **_X′_**<br>**_sorted_** : (_L, D_)_←Linear_2(**_y′_**)|
|17:|**_X′_** : (_L, D_)_←_**_X′_**<br>**_sorted_**[**_Ireverse_**]_{_Reverse sort out-|
||put_}_|

A illustrates Graph-Mamba’s adaptation of the GPS layer, where the attention module is replaced by GMB, denoted as a GMB layer. We used the GatedGCN model as the default for MPNN for local context selection, as shown in Figure 1 B. The GatedGCN model aggregates information from neighboring nodes defined by edge connections, and employs a gating mechanism to decide on how much of that information to incorporate, inspired by RNNs. GatedGCN and GMB collectively contribute to the overarching theme of recurrence-based context selection in Graph-Mamba, facilitating node filtering within the local neighborhood and among the global connections. The graph feature computation with SE and PE prior to the GMB layers remain consistent. The GMB layers thus receive the SE/PE-aware node and edge embeddings as input.

A Graph-Mamba framework consists of _K_ stacked GMB layers. Algorithm 1 defines the GMB function (more explanations in Sections 3.4-3.6), and Algorithm 2 illustrates the forward pass through _K_ GMB layers. In Algorithm 2, each GMB layer performs two round of embedding updates using MPNN and GMB, given an input graph of _L_ nodes, _E_ edges, and embedding size _D_ . Specifically, an MPNN updates both node and edge embeddings (line 2), while GMB updates the node embeddings only (line 3). The updated node embeddings from an MPNN ( _**X** M[k]_[+1] ) and GMB ( _**X** GMB[k]_[+1][) are combined through an MLP layer to pro-]

duce the output node embeddings (line 6). Using the output from the previous layer as the input for the next layer, this process iterates through _L_ GMB layers to obtain the final output node embeddings, which are subsequently used for downstream tasks.

## **3.4. Node prioritization strategy for non-sequential graph input**

## **Algorithm 2** Graph-Mamba with K GMB Layers

**Input:** Node embeddings _**X**_[0] _∈_ R _[L][×][D]_ ; Edge embeddings _**E**_[0] _∈_ R _[E][×][D]_ ; Adjacency matrix _**A** ∈_ R _[L][×][L]_ . **Output:** Returns _**X**[K] ∈_ R _[L][×][D]_ , and _**E**[K] ∈_ R _[E][×][D]_ , used for downstream prediction tasks.

2:1:3: **for** _**XX** k_ ˆˆ _MGMB_ = 0, 1, · · · , K-1 _[k][k]_[+1][+1] _,_ _**E**[←][k]_[+1] _[GMB] ← MPNN[k]_ **do**[(] _**[X]**[k]_[)] _[k]_ ( _**X**[k] ,_ _**E**[k] ,_ _**A**_ ) 4: _**X** M[k]_[+1] _← Dropout_ ( _**X**_[ˆ] _M[k]_[+1] + _**X**[k]_ ) 5: _**X** GMB[k]_[+1] _[←][Dropout]_[(] _**X**_[ˆ] _GMB[k]_[+1][+] _**[ X]**[k]_[)] 6: _**X**[k]_[+1] _← MLP[k]_ ( _**X** M[k]_[+1] + _**X** GMB[k]_[+1][)] 7: **end for** 8: return _**X**[K] ,_ _**E**[K]_

A major challenge of adapting sequence models such as Mamba to graphs stems from the unidirectionality of recurrent scan and update. In dense attention, all nodes attend to one another. However, due to the recurrent nature of the sequence modeling, in Mamba, each node gets updated based on nodes that come before them from the hidden states, not vice versa. For example, in an input sequence of length _L_ , the last node has access to hidden states that incorporate most context including all prior nodes 0 to _L−_ 2. In contrast, node 1 only has access to limited context via hidden states that encode node 0 only. This restricted information flow removes connections between nodes based on its position in the sequence, allowing GMB to prioritize specific nodes of higher importance at the end of the sequence for informed sparsification.

To achieve informed sparsification in GMB, we explored an input node prioritization strategy by node heuristics that are proxy of node importance, as illustrated in Figure 1 C. When we first flatten a graph into a sequence, the nodes do not assume any particular order. The input nodes are then sorted in ascending order by node heuristic such as node degree. The intuition behind is that more important nodes should have access to more context (i.e., a longer history of prior nodes), and therefore to be placed at the end of the sequence. In in Algorithm 1, lines 1-4 illustrate the sequence sorting procedure for an input graph of _L_ nodes, where node heuristic _**H[′]**_ determines the node order in the flattended sequence. Lines 5-16 compute the selective SSM using Mamba, explained in more details in subsequent

_Table 1._ **Benchmark of Graph-Mamba on Long-Range Graph Datasets** with existing methods. These five datasets feature large input graphs with 150 to 1,400 nodes. Best results are colored in **first** , **second** , **third** .

|**Model**|**Peptides-Func**|**Peptides-Struct**|**PascalVOC-SP**|**COCO-SP**|**MALNET-TINY**|
|---|---|---|---|---|---|
||AP _↑_|MAE _↓_|F1 score _↑_|F1 score _↑_|Accuracy _↑_|
|GCN|0.5930_±_0.0023|0.3496_±_0.0013|0.1268_±_0.0060|0.0841_±_0.0010|0.8100|
|GIN|0.5498_±_0.0079|0.3547_±_0.0045|0.1265_±_0.0076|0.1339_±_0.0044|0.8898_±_0.0055|
|GatedGCN|0.5864_±_0.0077|0.3420_±_0.0013|0.2873_±_0.0219|0.2641_±_0.0045|0.9223_±_0.0065|
|GPS+Transformer|**0.6575**_±_**0.0049**|**0.2510**_±_**0.0015**|**0.3689**_±_**0.0131**|**0.3774**_±_**0.0150**|OOM (bs=8)|
|GPS+Performer|**0.6475**_±_**0.0056**|0.2558_±_0.0012|**0.3724**_±_**0.0131**|**0.3761**_±_**0.0101**|**0.9264**_±_**0.0078**|
|GPS+BigBird|0.5854_±_0.0079|0.2842_±_0.0130|0.2762_±_0.0069|0.2622_±_0.0008|0.9234_±_0.0034|
|Exphormer|0.6258_±_0.0092|**0.2512**_±_**0.0025**|0.3446_±_0.0064|0.3430_±_0.0108|**0.9422**_±_**0.0024**|
|Graph-Mamba|**0.6739**_±_**0.0087**|**0.2478**_±_**0.0016**|**0.4191**_±_**0.0126**|**0.3960**_±_**0.0175**|**0.9340**_±_**0.0027**|

Section 3.6. In line 17, the SSM output is reverse sorted to return updated _**X[′]**_ in the original order. More details about other choices of node heurstics and rationale behind node prioritization are summarized in Appendix D.

## **3.5. Permutation-based training and inference recipe**

Following the input node prioritization strategy, GraphMamba uses a permutation-focused training and inference recipe to promote permutation invariance, as illustrated in Figure 1 C. Intuitively, when ordering the nodes by heuristics such as node degree, nodes within the same degree are deemed equally important in the graph. Therefore, nodes of the same degree are randomly shuffled during training to minimize bias towards any particular order. Line 1 in Algorithm 1 illustrates the permutation implementation. Specifically, random noise _∈_ [0 _,_ 1) is added to node heuristic _**H**_ , and the jittered _**H[′]**_ determines the input node order.

In the training stage of Graph-Mamba, GMB is called once to output updated node embeddings from a random permutation of input node sequence. At inference time, the _m_ GMB outputs _**X**_[ˆ] _GMB[k]_[+1][are averaged and passed on to subsequent] computation. The _m_ -fold average at inference time aims to provide stability, and makes the output node embeddings invariant to the permutations applied.

## **3.6. GMB with improved computation efficiency**

GMB’s selection mechanism is illustrated in Figure 1 D, with corresponding Mamba implementation detailed in Algorithm 1 lines 5-16. The sorted node sequence _**Xsorted**_ consists of _L_ nodes, each with a node embedding of size _D_ . The Mamba computation consists of linear projection of normalized input to _D[′]_ dimensions (line 5, 6), followed by 1-D convolution and SiLU activation (line 8), and SSM computation (lines 9-14). The SSM output _y_ is gated by a projection of the original input (line 7, 15), before the final projection back the original size as output (line 16).

expansion in parameters in _**A**_ **[¯]** , _**B**_ **[¯]** , and _**C**_ would lead to increased computational cost in SSM. Mamba implements an efficient hardware-aware algorithm that leverages the hierarchy in GPU memory to alleviate this overhead. Specifically, with input batch size _B_ , Mamba reads the _O_ ( _BLD[′]_ + _ND[′]_ ) of input _**A**_ , _**B**_ , _**C**_ , and ∆ from HBM, computes the intermediate states of size _O_ ( _BLD[′] N_ ) in SRAM and writes the final output of size of _O_ ( _BLD[′]_ ) to HBM, thus reducing IOs by a factor of _N_ . Not storing the intermediate states also lowers memory consumption, where intermediates states are recomputed for gradient calculation in the backward pass. With the GPU-aware implementation of Mamba, GMB achieves linear time complexity ( _O_ ( _L_ )) to input sequence length, which is significantly faster than the dense attention computation in transformers with quadratic time complexity ( _O_ ( _L_[2] )).

## **4. Experiments**

## **4.1. Benchmark on graph-based prediction tasks**

We benchmarked Graph-Mamba on ten datasets from the Long Range Graph Benchmark (LRGB) (Dwivedi et al., 2022) and GNN Benchmark (Dwivedi et al., 2023) as a comprehensive evaluation. These benchmarks evaluate model performance on various graph-based prediction tasks, including graph, node, and link-level classification and regression. For each dataset, we reported the test metric across multiple runs to ensure robustness. The dataset and task descriptions are summarized in Appendix A. Details about experiment setup are summarized in Appendix F and G.

We focused the comparison on the choice of attention modules within the GraphGPS framework. Specifically, we evaluated Graph-Mamba’s performance against GraphGPS with dense attention (Transformer) and various implementations of sparse attention (i.e., Exphormer, Performer, and BigBird) (Rampa´sek et al.ˇ , 2022; Shirzad et al., 2023; Choromanski et al., 2020; Zaheer et al., 2020).

With the data-dependent selection mechanism, the _L_ -fold

**==> picture [439 x 146] intentionally omitted <==**

MalNet-Tiny FLOPs Benchmark MalNet-Tiny GPU Memory Benchmark<br>OOM OOM<br>GPS+Transformer GPS+Transformer<br>Graph-Mamba Graph-Mamba<br>Avg # Nodes Avg # Nodes<br>9)<br>9)<br>Avg FLOPs (10<br>Avg GPU Memory (MB)<br>Avg FLOPs (10<br>**----- End of picture text -----**<br>

_Figure 2._ **FLOPs and Memory Benchmark of Graph-Mamba with GPS+Transformer** on the MalNet-Tiny dataset, subsampled at various ratios.

Table 1 highlights Graph-Mamba’s superior performance in capturing long-range dependencies from the top five datasets with the largest input lengths, ranging from 150 to 1,400 nodes per graph respectively. In four out of five datasets, Graph-Mamba offered considerable improvement (up to 5%) to the other sparse attention methods. Graph-Mamba also compared favorably to Transformer with dense attention, which underscores the importance of context-aware node selection in graphs with long-range dependencies. On the other five datasets with small to medium-sized graphs, Graph-Mamba further demonstrated its robustness by showcasing comparable performance to the state-of-the-art sparse and dense attention methods, summarized in Appendix Table 4. These results endorsed Graph-Mamba’s ability to capture long-range context with the input-dependent node selection mechanism, while generalizing well to common graph-based prediction tasks.

## **4.2. FLOPs and memory consumption**

Graph-Mamba offers significant improvement in efficiency in addition to performance gain. We benchmarked GraphMamba’s Floating Point Operations (FLOPs) and memory consumption during training stage against existing methods on five datasets, as detailed in Appendix G.

Figure 2 illustrates the computational cost on the MalNetTiny dataset with an average of 1,410.3 nodes, subsampled at increasing ratios. Graph-Mamba demonstrates linear complexity in both FLOPs and memory with respect to input length, whereas GPS-Transformer’s cost grows quadratically. GPS-Transformer encounters out-of-memory issues with input sizes below 700 nodes at a batch size of 16, impeding efficient model training. In contrast, Graph-Mamba supports the training of full graphs with twice the number of nodes, at batch size up to 256.

In Figure 3, we further compared Graph-Mamba with

the sparse attention variants in GPS. For each benchmark dataset, the x-axis represents the average number of FLOPs per training example, while the y-axis showcases the average GPU consumption. Graph-Mamba consistently occupies the lower left corner, indicating the fewest FLOPs and least memory usage across all datasets. Specifically, in the Peptides-func dataset, Graph-Mamba achieves a 74% reduction in memory usage and a 66% reduction in FLOPs compared to Transformer. Moreover, Graph-Mamba demonstrates a 40% decrease in both FLOPs and memory usage against the state-of-the-art sparse graph attention implementation, Exphormer. These results highlight the computational efficiency of Graph-Mamba, opening up the potential of efficient training on larger graphs.

## **4.3. Ablation of Graph-Mamba’s training and inference recipes**

We demonstrate the effectiveness of the proposed training and inference recipe for Graph-Mamba with an ablation study. Table 2 showcases the predictive performance of Graph-Mamba from three training and inference settings on two datasets Peptides-Func and PascalVOC-SP. The baseline training procedure did not use any node prioritization or permutation techniques. The permutation-only setting introduced random permutation at the node-level during training time. At inference time, the output was obtained by averaging five runs of corresponding permutation. The node prioritization setting injected node degree as the node heuristics to sort the input sequence, while restricting the permutation to a smaller range. Specifically, the node degree prioritization setting only allows nodes with the same degree to randomly permute among themselves.

The permutation strategy led to a significant performance gain. The node-level permutation setting saw a 3% increase in average precision on the Peptides-Func dataset and a 10% increase in F1 scores on the PascalVOC-SP dataset,

**==> picture [487 x 173] intentionally omitted <==**

PCQM-Contact CLUSTER Peptides-func PascalVOC-SP<br>Avg FLOPs (10 [9] ) Avg FLOPs (10 [9] ) Avg FLOPs (10 [9] ) Avg FLOPs (10 [9] )<br>Avg # Nodes 30.1 117.2 150.9 476.9<br>Method GPS+Transformer GPS+Performer GPS+BigBird Exphormer Graph-Mamba<br>Avg GPU Memory (MB) Avg GPU Memory (MB) Avg GPU Memory (MB) Avg GPU Memory (MB)<br>**----- End of picture text -----**<br>

_Figure 3._ **FLOPs and Memory Benchmark of Graph-Mamba** with existing methods, with emphasis on **sparse attention variants** , on four datasets of increasing input size.

_Table 2._ **Ablation Study of Permutation and Node Prioritization Strategies** on the Peptides-Func and PascalVOC-SP datasets.

|**y of Permutation a**|**nd Node Prioritization S**|**trategies**on the Peptides-Func and Pasca|
|---|---|---|
|**Permutation**|**Node Prioritization**|**Peptides-Func**<br>**PascalVOC-SP**<br>AP _↑_<br>F1 score _↑_|
|-|-|0.6581<br>0.3105|
|Node Level|-|0.6821<br>0.4193|
|Node Level|Degree|**06834**<br>**04314**|
|||**.**<br>**.**|

compared to the baseline training strategy. Combining node prioritization by degree with permutation further improved the scores on the PascalVOC-SP dataset. These results underscore the importance of devising graph-focused adaptations of sequence modeling methods, tailoring towards the characteristics of non-sequential data. These elegant sequence design techniques offer improved modeling power to Mamba well beyond simple plug-and-play. We therefore recommend using the combination of input node prioritization by node degree and node-level permutation as the default training and inference recipe for Graph-Mamba. Table 1 and 4 showcased model performance from this recommended recipe.

## **5. Conclusion**

We propose Graph-Mamba, a novel graph model that leverages SSM for efficient data-dependent context selection, as an alternative to graph attention sparsification. GMB’s selection mechanism filters relevant nodes as context, effectively compressing and propagating long-range dependencies during node embedding updates. Through recurrent scan with context compression, Graph-Mamba achieves linear-time complexity and reduced memory consumption. The specialized training and inference recipe, combining permutation and node prioritization, further adapts Mamba, a se-

quence modeling technique, to non-sequential graph input with significant performance improvement. In empirical experiments, Graph-Mamba demonstrated the state-of-the-art or comparable performance across ten datasets with various graph prediction tasks. Graph-Mamba thus presents a promising option to replace traditional dense or sparse graph attention, offering competitive predictive power and longrange context awareness at a fraction of the computational cost.

In future works, exploring alternative model architectures beyond the GraphGPS framework is crucial for enhancing predictive performance. The architecture-agnostic nature of GMB offers flexibility for such applications. Furthermore, this study highlights the significance of sequence construction and training strategies in facilitating sequence model learning with non-sequential graph input. Beyond permutation and node heuristics, effective ways to inject graph topology into input sequences remain unexplored. Ultimately, learning the optimal strategy of flattening a graph into sequences from data is essential. SSM-based sequence modeling offers new perspectives for causality analysis beyond prediction, presenting a promising direction for graph data analysis. The improved efficiency further supports the development of graph foundation models, opening up the possibility of large-scale pre-training.

## **Impact Statement**

Graph Mamba was designed as a general graph representation learning method. Therefore, it can not have any direct negative societal outcomes. Nonetheless, adverse or malicious application of the proposed algorithm in various domains, including drug discovery and healthcare, may lead to undesirable effects.

## **References**

- Alon, U. and Yahav, E. On the bottleneck of graph neural networks and its practical implications, 2021.

- Bresson, X. and Laurent, T. Residual gated graph convnets, 2018.

- Chen, D., O’Bray, L., and Borgwardt, K. Structure-aware transformer for graph representation learning. In _International Conference on Machine Learning_ , pp. 3469–3489. PMLR, 2022.

- Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., et al. Rethinking attention with performers. _arXiv preprint arXiv:2009.14794_ , 2020.

- D’Ascoli, S., Touvron, H., Leavitt, M. L., Morcos, A. S., Biroli, G., and Sagun, L. Convit: Improving vision transformers with soft convolutional inductive biases. In Meila, M. and Zhang, T. (eds.), _Proceedings of the 38th International Conference on Machine Learning_ , volume 139 of _Proceedings of Machine Learning Research_ , pp. 2286–2296. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/ v139/d-ascoli21a.html.

- Defferrard, M., Bresson, X., and Vandergheynst, P. Convolutional neural networks on graphs with fast localized spectral filtering, 2017.

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale, 2021.

- Dwivedi, V. and Bresson, X. A generalization of transformer networks to graphs. _arXiv preprint arXiv:2012.09699_ , 2012.

- Dwivedi, V. P. and Bresson, X. A generalization of transformer networks to graphs, 2021.

- Dwivedi, V. P., Rampa´sek, L., Galkin, M., Parviz, A., Wolf,ˇ G., Luu, A. T., and Beaini, D. Long range graph benchmark. _Advances in Neural Information Processing Systems_ , 35:22326–22340, 2022.

- Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio, Y., and Bresson, X. Benchmarking graph neural networks. _Journal of Machine Learning Research_ , 24 (43):1–48, 2023.

- Fan, W., Ma, Y., Li, Q., He, Y., Zhao, E., Tang, J., and Yin, D. Graph neural networks for social recommendation, 2019.

- Freitas, S., Dong, Y., Neil, J., and Chau, D. H. A large-scale database for graph representation learning, 2021.

- Fu, D. Y., Dao, T., Saab, K. K., Thomas, A. W., Rudra, A., and Re,´ C. Hungry hungry hippos: Towards language modeling with state space models. _arXiv preprint arXiv:2212.14052_ , 2022.

- Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ , 2023.

- Gu, A., Goel, K., and Re,´ C. Efficiently modeling long sequences with structured state spaces. _arXiv preprint arXiv:2111.00396_ , 2021.

- Guo, J., Han, K., Wu, H., Tang, Y., Chen, X., Wang, Y., and Xu, C. Cmt: Convolutional neural networks meet vision transformers, 2022.

- Hamilton, W. L., Ying, R., and Leskovec, J. Inductive representation learning on large graphs, 2018.

- Kalyan, K. S., Rajasekharan, A., and Sangeetha, S. Ammus : A survey of transformer-based pretrained models in natural language processing, 2021.

- Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. _arXiv preprint arXiv:1609.02907_ , 2016.

- Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks, 2017.

- Kreuzer, D., Beaini, D., Hamilton, W., Letourneau, V., and´ Tossou, P. Rethinking graph transformers with spectral attention. _Advances in Neural Information Processing Systems_ , 34:21618–21629, 2021a.

- Kreuzer, D., Beaini, D., Hamilton, W., Letourneau, V., and´ Tossou, P. Rethinking graph transformers with spectral attention. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), _Advances in Neural Information Processing Systems_ , volume 34, pp. 21618–21629. Curran Associates, Inc., 2021b.

- Li, X., Zhou, Y., Dvornek, N., Zhang, M., Gao, S., Zhuang, J., Scheinost, D., Staib, L. H., Ventola, P., and Duncan, J. S. Braingnn: Interpretable brain graph neural network for fmri analysis. _Medical_

_Image Analysis_ , 74:102233, 2021. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2021.102233. URL https://www.sciencedirect.com/ science/article/pii/S1361841521002784.

- Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., and Liu, Y. Vmamba: Visual state space model. _arXiv preprint arXiv:2401.10166_ , 2024.

- Ma, J., Li, F., and Wang, B. U-mamba: Enhancing longrange dependency for biomedical image segmentation. _arXiv preprint arXiv:2401.04722_ , 2024.

- Mehta, H., Gupta, A., Cutkosky, A., and Neyshabur, B. Long range language modeling via gated state spaces. _arXiv preprint arXiv:2206.13947_ , 2022.

- Mialon, G., Chen, D., Selosse, M., and Mairal, J. Graphit: Encoding graph structure in transformers, 2021.

- Nguyen, E., Poli, M., Faizi, M., Thomas, A., Birch-Sykes, C., Wornow, M., Patel, A., Rabideau, C., Massaroli, S., Bengio, Y., et al. Hyenadna: Long-range genomic sequence modeling at single nucleotide resolution. _arXiv preprint arXiv:2306.15794_ , 2023.

- Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Cao, H., Cheng, X., Chung, M., Grella, M., GV, K. K., et al. Rwkv: Reinventing rnns for the transformer era. _arXiv preprint arXiv:2305.13048_ , 2023.

- Rampa´sek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf,ˇ G., and Beaini, D. Recipe for a general, powerful, scalable graph transformer. _Advances in Neural Information Processing Systems_ , 35:14501–14515, 2022.

- Wu, Z., Jain, P., Wright, M. A., Mirhoseini, A., Gonzalez, J. E., and Stoica, I. Representing long-range context for graph neural networks with global attention, 2022.

- Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? _arXiv preprint arXiv:1810.00826_ , 2018.

- Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., and Liu, T.-Y. Do transformers really perform bad for graph representation?, 2021.

- Yun, C., Chang, Y.-W., Bhojanapalli, S., Rawat, A. S., Reddi, S., and Kumar, S. O (n) connections are expressive enough: Universal approximability of sparse transformers. _Advances in Neural Information Processing Systems_ , 33:13783–13794, 2020.

- Yun, S., Jeong, M., Kim, R., Kang, J., and Kim, H. J. Graph transformer networks. _Advances in neural information processing systems_ , 32, 2019.

- Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big bird: Transformers for longer sequences. _Advances in neural information processing systems_ , 33:17283–17297, 2020.

- Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., and Wang, X. Vision mamba: Efficient visual representation learning with bidirectional state space model. _arXiv preprint arXiv:2401.09417_ , 2024.

- Shirzad, H., Velingker, A., Venkatachalam, B., Sutherland, D. J., and Sinop, A. K. Exphormer: Sparse transformers for graphs. _arXiv preprint arXiv:2303.06147_ , 2023.

- Spielman, D. A. and Teng, S.-H. Spectral sparsification of graphs. _SIAM Journal on Computing_ , 40(4):981–1025, 2011.

- Topping, J., Giovanni, F. D., Chamberlain, B. P., Dong, X., and Bronstein, M. M. Understanding over-squashing and bottlenecks on graphs via curvature, 2022.

- Tsepa, O., Naida, B., Goldenberg, A., and Wang, B. Congfu: Conditional graph fusion for drug synergy prediction, 2023.

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017.

- Velickoviˇ c, P., Cucurull, G., Casanova, A., Romero, A., Li´ o,` P., and Bengio, Y. Graph attention networks, 2018.

## **A. Dataset Description.**

We evaluate Graph-Mamba on ten datasets from two popular graph benchmarks, the Long-Range Graph Benchmark (LRGB) (Dwivedi et al., 2023) and GNN Benchmark (Dwivedi et al., 2022). Table 3 summarizes the dataset characteristics and associated prediction tasks. The first five datasets in bold feature long input size (i.e., Avg. Nodes), corresponding to Table 1. The other five datasets have small to medium input size, corresponding to Table 4.

**CIFAR10 and MNIST** (Dwivedi et al., 2023) introduce the graph equivalents of the image classification datasets. Each image is represented by the 8-nearest neighbor graph derived from the SLIC superpixels. Both datasets feature 10 classification labels and adhere to the standard dataset splits established by the original image datasets.

**MalNet-Tiny** (Freitas et al., 2021) consists of function call graphs extracted from Android APKs. It is a subset of the larger MalNet collection, containing 5,000 graphs each with a maximum of 5,000 nodes. This dataset contains 5 classification labels, including 1 benign software and 4 types of malware. In this benchmarking version, the original node and edge features are removed, and each node is represented with its local degree profile instead. This modification presents a challenging classification task relying solely on graph structures.

**PATTERN and CLUSTER** (Dwivedi et al., 2023) are synthetic graph datasets of community structures simulated by the Stochastic Block Model (SBM). Both datasets represent node-level classification tasks with an inductive focus. In PATTERN, the goal is to distinguish nodes belonging to 100 distinct sub-graph patterns randomly generated with different SBM parameters from the remaining nodes. In CLUSTER, each graph contains 6 clusters from the same distribution, with 6 test nodes representing each unique cluster. The objective is to predict the cluster identity of these test nodes.

**Peptides-func and Peptides-struct** (Dwivedi et al., 2022) are graph representations of peptides with large diameters. Peptides-Func involves graph-level classification with 10 functional labels. Peptides-Struct focuses on a graph-level regression task, predicting 11 structural properties of the molecules.

**PCQM-Contact** (Dwivedi et al., 2022) introduces a link prediction task based on the PCQM4Mv2 dataset of 3D molecular structures. A contact link is defined between pairs of distant nodes that are spatially close in three-dimensional space. The evaluation metric used is the Mean Reciprocal Rank (MRR), a ranking-based measure.

**PascalVOC-SP and COCO-SP** (Dwivedi et al., 2022) are the graph representations of the image datasets upon SLIC superpixelization. The node-level classification task involves classifying superpixels into corresponding object classes similar to semantic segmentation.

_Table 3._ Dataset description.

||_Table 3._ Dataset description.|
|**Dataset**|**Prediction Task**<br>**Prediction Level**<br>**Graphs**<br>**Avg. Nodes**<br>**Avg. Edges**<br>**Benchmark**|
|**Peptides-func**<br>**Peptides-struct**<br>**PascalVOC-SP**<br>**COCO-SP**<br>**MalNet-Tiny**|Classifcation<br>Graph<br>15,535<br>150.9<br>307.3<br>LRGB<br>Regression<br>Graph<br>15,535<br>150.9<br>307.3<br>LRGB<br>Classifcation<br>Node<br>11,355<br>479.4<br>2,710.5<br>LRGB<br>Classifcation<br>Node<br>123,286<br>476.9<br>2,693.7<br>LRGB<br>Classifcation<br>Graph<br>5,000<br>1,410.3<br>2,859.9<br>GNN|
|PCQM-Contact<br>CIFAR10<br>MNIST<br>CLUSTER<br>PATTERN|Link Ranking<br>Link<br>529,434<br>30.1<br>61.0<br>LRGB<br>Classifcation<br>Graph<br>60,000<br>117.6<br>941.1<br>GNN<br>Classifcation<br>Graph<br>70,000<br>70.6<br>564.5<br>GNN<br>Classifcation<br>Node<br>12,000<br>117.2<br>2,150.9<br>GNN<br>Classifcation<br>Node<br>14,000<br>118.9<br>3,039.3<br>GNN|

## **B. Additional Benchmark Results.**

Table 4 presents benchmark results on the five datasets with small to medium input length, ranging from 30 to 120 nodes in the graph. Graph-Mamba demonstrates predictive performance comparable to GraphGPS with full Transformer and Exphormer, endorsing its generalizability to common graph tasks.

_Table 4._ **Benchmark on Short to Medium-Range Graph Datasets** with existing methods. Best results are colored in **first** , **second** ,

**third** .

|**ird**.||||||
|---|---|---|---|---|---|
|**Model**|**CIFAR10**|**MNIST**|**CLUSTER**|**PATTERN**|**PCQM-Contact**|
||Accuracy _↑_|Accuracy _↑_|Accuracy _↑_|Accuracy _↑_|MRR _↑_|
|GCN|0.5571_±_0.0038|0.9071_±_0.0021|0.6850_±_0.0097|0.7189_±_0.0033|0.3234_±_0.0006|
|GIN|0.5526_±_0.0152|0.9649_±_0.0025|0.6472_±_0.0155|0.8539_±_0.0013|0.3180_±_0.0027|
|GatedGCN|0.6731_±_0.0031|0.9734_±_0.0014|0.7384_±_0.0032|0.8557_±_0.0008|0.3218_±_0.0011|
|GPS+Transformer|**0.7226**_±_**0.0031**|0.9811_±_0.0011|**0.7799**_±_**0.0017**|**0.8664**_±_**0.0011**|**0.3442**_±_**0.0009**|
|GPS+Performer|0.7067_±_0.0033|**0.9834**_±_**0.0003**|**0.7829**_±_**0.0004**|0.8334_±_0.0029|**0.3437**_±_**0.0005**|
|GPS+BigBird|0.7048_±_0.0010|0.9817_±_0.0001|0.7746_±_0.0002|0.8600_±_0.0014|0.3391_±_0.0002|
|Exphormer|**0.7413**_±_**0.0050**|**0.9843**_±_**0.0004**|**0.7802**_±_**0.0011**|**0.8670**_±_**0.0003**|**0.3587**_±_**0.0025**|
|Graph-Mamba|**0.7370**_±_**0.0034**|**0.9842**_±_**0.0008**|0.7680_±_0.0036|**0.8671**_±_**0.0005**|0.3395_±_0.0013|

## **C. Proof of Theorem.**

We revisit the proof of the main Theorem in Mamba as presented by Gu & Dao (2023) for further reference for the SSM selection mechanism.

_**Theorem 1** . Consider N = 1,_ _**A** = -1,_ _**B** = 1, and_ ∆ _t_ = _softplus_ ( _Linear_ ( _xt_ )) _for selective SSM, the discretized recurrence output is defined as_

**==> picture [296 x 26] intentionally omitted <==**

_Proof._ Substitute the expression of ∆ _t_ into the zero-order hold discretization formulas _**A**_ **[¯]** = exp(∆ _t_ _**A**_ ) and _**B**_ **[¯]** = (∆ _t_ _**A**_ ) _[−]_[1] (exp(∆ _t ·_ _**A**_ ) _− I_ ) _·_ ∆ _t_ _**B**_ .

**==> picture [326 x 145] intentionally omitted <==**

## **D. Input Node Prioritization and Permutation Strategies.**

To elaborate on the rationale behind node prioritization, we first assume a sequence of _L_ nodes in random order, denoted as _ND_ 0, _ND_ 1, ... , _NDL−_ 1. The selection mechanism implemented as SSM computation in the Mamba module performs a unidirectional scan over the sequence of nodes, updating the hidden states at each step of recurrence. This defines different levels of access to context (i.e., other nodes) based on node position in the sequence. For example, _ND_ 1 has limited access to context and updates itself based on hidden states that encode _ND_ 0 only. Its connections to the other nodes are removed entirely. In contrast, _NDL−_ 1 has access to most context including all prior nodes _ND_ 0 to _NDL−_ 2. It reatins connections to all other nodes. Intuitively, with a randomly ordered node sequence, SSM creates a similar effect as random subsampling. Instead, we would like to create a biased sampling procedure that favor connections for important nodes in a graph. This is achieved by placing the important nodes at the end of the sequence.

In addition to the recommended recipe, we explored a few variants of node prioritization and permutation techniques summarized as follows:

- **Node prioritization by eigenvector centrality.** We explored eigenvector centrality as an alternative proxy for node importance. Eigenvector centrality measures a node’s influence by adding the centrality of its neighbors. A node with high centrality indicates connections to other influential nodes in the graph. The eigenvalue centrality of each node is defined by the principal eigenvector that corresponds to the largest eigenvalue. In Graph-Mamba, the eigenvector centrality scores were calculated using the eigenvector ~~c~~ entrality ~~n~~ umpy() function from NetworkX.

- **Permutation by clusters.** The input sequence consists of nodes grouped by clusters defined by edge connectivity. The permutation happens first on the cluster level. The nodes within each cluster are then randomly permuted among themselves. The intuition behind is that the local structure and topology are essential and not represented in dense attention. This approach aims to inject the local structure into the input sequence. We used the Louvain algorithm for unsupervised graph partitioning.

Table 5 presents the predictive performance from these two aforementioned variants, highlighted in blue. Cluster-level permutation leads to a significant performance gain compared to the baseline, but slightly less compared to node-level permutation. Similarly, node prioritization by eigenvector centrality is not as effective as by degree as demonstrated in these two datasets.

_Table 5._ **Comparison with Alternative Node Prioritization and Permutation Strategies** on the Peptides-Func and PascalVOC-SP datasets.

|**Permutation**|**Node Prioritization**|**Peptides-Func**<br>**PascalVOC-SP**<br>AP _↑_<br>F1 score _↑_|
|---|---|---|
|-|-|0.6581<br>0.3105|
|Node Level<br>Cluster Level|-<br>-|0.6821<br>0.4193<br>0.6769<br>0.3802|
|Node Level<br>Node Level|Eigenvector Centrality<br>Degree|0.6739<br>0.3961<br>**0.6834**<br>**0.4314**|

## **E. Binning Technique for Large Graphs.**

For large graph datasets, we devised a binning technique that randomly divides the long sequence of input nodes into _n_ bins and applies GMB to each of the sub-sequence individually. Node prioritization and permutation happen within each sub-sequence. We then obtain the GMB output for each sub-sequence in _n_ passes, and combine the updated sub-sequences to match the node order in the original sequence. Conceptually, the binning technique further sparsifies the node connections for embedding updates, since only nodes within a sub-sequence can interact. This technique helps further reduce memory usage.

## **F. Hyperparameters.**

We followed the hyperparamter suggestions in the Exphormer benchmark (Shirzad et al., 2023). We matched the number of parameters in the Mamba/attention module and the whole model between Graph-Mamba and GPS+Transformer. The model size, positional encodings, and batch size remained consistent with the default in the Exphormer benchmark across all models. For Graph-Mamba, we used a Mamba block with state dimension of 16, convolution kernel size of 4, and expansion factor of 1. We adjusted the Adam optimizer setting by increasing the initial learning rate and introducing weight decay, following Mamba’s training recipe (Gu & Dao, 2023).

Tables 6 and 7 summarize the hyperparameters used for Graph-Mamba training. For trainable parameters, “Num Params Mamba” indicates the number of parameters in a single Mamba block in one GMB layer, and “Num Params Total” reports the total number of trainable parameters in the model. The binning technique described in Appendix E is only applied to large graph datasets with close to or more than 500 nodes.

_Table 6._ **Hyperparameters used for Graph-Mamba on Long-Range Graph Datasets.**

|_Table 6._|**Hyperparameters used for Graph-Mamba on Long-Range Graph Datasets.**|
|**Dataset**|**Peptides-Func**<br>**Peptides-Struct**<br>**PascalVOC-SP**<br>**COCO-SP**<br>**MalNet-Tiny**|
|Num Layers<br>Hidden Dim<br>PE|4<br>4<br>4<br>4<br>5<br>96<br>96<br>96<br>96<br>64<br>LapPE<br>LapPE<br>LapPE<br>LapPE<br>LapPE|
|Batch Size<br>Learning Rate (LR)<br>Weight Decay<br>Num Epochs|128<br>128<br>32<br>32<br>16<br>0.001<br>0.001<br>0.0015<br>0.0015<br>0.0015<br>0.01<br>0.01<br>0.001<br>0.001<br>0.00001<br>200<br>200<br>300<br>300<br>150|
|Num Bins|-<br>-<br>2<br>2<br>3|
|Num Params Mamba<br>Num Params Total|34,080<br>34,080<br>34,080<br>34,080<br>16,320<br>373,018<br>491,867<br>497,781<br>497,781<br>285,125|

_Table 7._ **Hyperparameters used for Graph-Mamba on Short to Medium-Range Graph Datasets.**

|_Table 7._ **Hyperparam**|**eters used for Graph-Mamba on Short to Medium-Range Graph Datasets.**|
|**Dataset**|**CIFAR10**<br>**MNIST**<br>**CLUSTER**<br>**PATTERN**<br>**PCQM-Contact**|
|Num Layers<br>Hidden Dim<br>PE|3<br>3<br>16<br>6<br>4<br>52<br>52<br>48<br>64<br>96<br>LapPE<br>LapPE<br>LapPE<br>LapPE<br>LapPE|
|Batch Size<br>Learning Rate (LR)<br>LR Weight Decay<br>Num Epochs|16<br>16<br>16<br>32<br>128<br>0.005<br>0.005<br>0.001<br>0.001<br>0.002<br>0.01<br>0.01<br>0.0001<br>0.0001<br>0.01<br>100<br>100<br>100<br>100<br>100|
|Num Bins|-<br>-<br>-<br>-<br>-|
|Num Params Mamba<br>Num Params Total|11,388<br>11,388<br>9,840<br>16,320<br>34,080<br>113,818<br>116,486<br>508,966<br>335,281<br>500,112|

## **G. Benchmarking Experiments.**

For the GNN and LRGB benchmarks, we reported the average over 5 runs of random seeds 0-4 for Graph-Mamba, GPS+Transformer, and Exphormer. For the earlier methods and some of the GPS+Performer and GPS+BigBird runs, we consolidated the scores from the Exphormer benchmark (Shirzad et al., 2023). We reported the standard evaluation metrics for each dataset using the same pipelines as GraphGPS and Exphormer. Note that we capped the maximum GPU memory at 24GB for this benchmark, and an OOM case was reported when reducing the standard batch size by half still led to an OOM error. For the ablation study on training and inference recipes, we reported the scores from a single run with random seed 0.

For the FLOPs and memory benchmark, the statistics were collected from a single epoch in training phase. Specifically, we reported the average FLOPs performed per sample by summing the total number of FLOPs performed in all forward passes in one epoch, and dividing it by the number of training examples. Similarly for memory, we reported the peak GPU memory usage divided by the batch size as an estimate for average memory usage per training example. For FLOPs profiling experiments, we used the FlopsProfiler implementation from the deepspeed package. We obtained the peak GPU memory usage in one epoch from the torch.cuda.max ~~m~~ emory ~~a~~ llocated() function. The MalNet-Tiny dataset features the largest input graphs with an average of 1,410.3 nodes. To assess how the models scale with input size, we subsampled the Malnet-Tiny dataset at various ratios including 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9 to simulate increasing number of nodes in the input. Specifically, for each input graph, we randomly selected a fraction of input nodes at the desired ratio, and retained the edges associated with these selected nodes only. To ensure a fair comparison, we matched the model size for GPS+Transformer (286,725 parameters) and Graph-Mamba (285,125 parameters), consistent with the predictive performance benchmark. For the subsequent benchmark including the GPS sparse attention variants, the original datasets PCQM-Contact, CLUSTER, Peptides-Func, and PascalVOC-SP were used without any subsampling. These datasets feature increasing input length ranging from 30 up to 480 nodes.

## **H. Implementation Details.**

The Graph-Mamba model is implemented in PyTorch framework using mamba-ssm and PyTorch Geometric. The model was trained on a single RTX6000 or A100 GPU.
