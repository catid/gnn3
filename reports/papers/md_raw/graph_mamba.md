# **Graph Mamba: Towards Learning on Graphs with State Space Models** 

**Ali Behrouz**[* 1] **Farnoosh Hashemi**[* 1] 

## **Abstract** 

Graph Neural Networks (GNNs) have shown promising potential in graph representation learning. The majority of GNNs define a local message-passing mechanism, propagating information over the graph by stacking multiple layers. These methods, however, are known to suffer from two major limitations: over-squashing and poor capturing of long-range dependencies. Recently, Graph Transformers (GTs) emerged as a powerful alternative to Message-Passing Neural Networks (MPNNs). GTs, however, have quadratic computational cost, lack inductive biases on graph structures, and rely on complex Positional/Structural Encodings (SE/PE). In this paper, we show that while Transformers, complex message-passing, and SE/PE are sufficient for good performance in practice, neither is necessary. Motivated by the recent success of State Space Models (SSMs), such as Mamba, we present Graph Mamba Networks (GMNs), a general framework for a new class of GNNs based on selective SSMs. We discuss and categorize the new challenges when adapting SSMs to graph-structured data, and present four required and one optional steps to design GMNs, where we choose (1) Neighborhood Tokenization, (2) Token Ordering, (3) Architecture of Bidirectional Selective SSM Encoder, (4) Local Encoding, and dispensable (5) PE and SE. We further provide theoretical justification for the power of GMNs. Experiments demonstrate that despite much less computational cost, GMNs attain an outstanding performance in long-range, small-scale, large-scale, and heterophilic benchmark datasets. The code is in this link. 

## **1. Introduction** 

Recently, graph learning has become an important and popular area of study due to its impressive results in a wide range of applications, like neuroscience (Behrouz et al., 2023), social networks (Fan et al., 2019), molecular graphs (Wang et al., 2021), etc. In recent years, Message-Passing Neural Networks (MPNNs), which iteratively aggregate neighborhood information to learn the node/edge representations, have been the dominant paradigm in machine learning on graphs (Kipf & Welling, 2016; Velickoviˇ c et al.´ , 2018; Wu et al., 2020; Gutteridge et al., 2023). They, however, have some inherent limitations, including over-squashing (Di Giovanni et al., 2023), over-smoothing (Rusch et al., 2023), and poor capturing of long-range dependencies (Dwivedi et al., 2022). With the rise of Transformer architectures (Vaswani et al., 2017) and their success in diverse applications such as natural language processing (Wolf et al., 2020) and computer vision (Liu et al., 2021), their graph adaptations, so-called Graph Transformers (GTs), have gained popularity as the alternatives of MPNNs (Yun et al., 2019; Kim et al., 2022; Ramp´aˇsek et al., 2022). 

Graph transformers have shown promising performance in various graph tasks, and their variants have achieved top scores in several graph learning benchmarks (Hu et al., 2020; Dwivedi et al., 2022). The superiority of GTs over MPNNs is often explained by MPNNs’ bias towards encoding local structures (Muller et al.¨ , 2023), while a key underlying principle of GTs is to let nodes attend to all other nodes through a global attention mechanism (Kim et al., 2022; Yun et al., 2019), allowing direct modeling of long-range interactions. Global attention, however, has weak inductive bias and typically requires incorporating information about nodes’ positions to capture the graph structure (Rampa´sek et al.ˇ , 2022; Kim et al., 2022). To this end, various positional and structural encoding schemes based on spectral and graph features have been introduced (Kreuzer et al., 2021; Kim et al., 2022; Lim et al., 2023a). 

Despite the fact that GTs with proper positional encodings (PE) are universal approximators and provably more powerful 

> *Equal contribution 1Cornell University, Ithaca, USA. Correspondence to: Ali Behrouz _<_ ab2947@cornell.edu _>_ , Farnoosh Hashemi _<_ sh2574@cornell.edu _>_ . 

Preprint. Under Review. 

1 

**Graph Mamba** 

than any Weisfeiler-Lehman isomorphism test (WL test) (Kreuzer et al., 2021), their applicability to large-scale graphs is hindered by their poor scalability. That is, the standard global attention mechanism on a graph with _n_ nodes incurs both time and memory complexity of _O_ ( _n_[2] ), quadratic in the input size, making them infeasible on large graphs. To overcome the high computational cost, inspired by linear attentions (Zaheer et al., 2020), sparse attention mechanisms on graphs attracts attention (Rampa´sek et al.ˇ , 2022; Shirzad et al., 2023). For example, Exphormer (Shirzad et al., 2023) suggests using expander graphs, global connectors, and local neighborhoods as three patterns that can be incorporated in GTs, resulting in a sparse and efficient attention. Although sparse attentions partially overcome the memory cost of global attentions, GTs based on these sparse attentions (Rampa´sek et al.ˇ , 2022; Shirzad et al., 2023) still might suffer from quadratic time complexity. That is, they require costly PE (e.g., Laplacian eigen-decomposition) and structural encoding (SE) to achieve their best performance, which can take _O_ ( _n_[2] ) to compute. 

Another approach to improve GTs’ high computational cost is to use subgraph tokenization (Chen et al., 2023; Zhao et al., 2021; Kuang et al., 2021; Baek et al., 2021; He et al., 2023), where tokens (a.k.a patches) are small subgraphs extracted with a pre-defined strategy. Typically, these methods obtain the initial representations of the subgraph tokens by passing them through an MPNN. Given _k_ extracted subgraphs (tokens), the time complexity of these methods is _O_ ( _k_[2] ), which is more efficient than typical GTs with node tokenization. Also, these methods often do not rely on complex PE/SE, as their tokens (subgraphs) inherently carry inductive bias. These methods, however, have two major drawbacks: (1) To achieve high expressive power, given a node, they usually require at least a subgraph per each remaining node (Zhang et al., 2023a; Bar-Shalom et al., 2023), meaning that _k ∈O_ ( _n_ ) and so the time complexity is _O_ ( _n_[2] ). (2) Encoding subgraphs via MPNNs can transmit all their challenges of over-smoothing and over-squashing, limiting their applicability to heterophilic and long-range graphs. 

Recently, Space State Models (SSMs), as an alternative of attention-based sequence modeling architectures like Transformers have gained increasing popularity due to their efficiency (Zhang et al., 2023b; Nguyen et al., 2023). They, however, do not achieve competitive performance with Transformers due to their limits in input-dependent context compression in sequence models, caused by their time-invariant transition mechanism. To this end, Gu & Dao (2023) present Mamba, a selective state space model that uses recurrent scans along with a selection mechanism to control which part of the sequence can flow into the hidden states. This selection can simply be interpreted as using data-dependent state transition mechanism (See §2.3 for a detailed discussion). Mamba outstanding performance in language modeling, outperforming Transformers of the same size and matching Transformers twice its size, motivates several recent studies to adapt its architecture for different data modalities (Liu et al., 2024b; Yang et al., 2024; Zhu et al., 2024; Ahamed & Cheng, 2024). 

Mamba architecture is specifically designed for sequence data and the complex non-causal nature of graphs makes directly applying Mamba on graphs challenging. Further, natural attempts to replace Transformers with Mamba in existing GTs frameworks (e.g., GPS (Rampa´sekˇ et al., 2022), TokenGT (Kim et al., 2022)) results in suboptimal performance in both effectiveness and time efficiency (See §5 for evaluation and §3 for a detailed discussion). The reason is, contrary to Transformers that allows each node to interact with all the other nodes, Mamaba, due to its recurrent nature, only incorporates information about previous tokens (nodes) in the sequence. This introduces new challenges compared to GTs: (1) The new paradigm requires token ordering that allows the model take advantage of the provided positional information as much as possible. (2) The architecture design need to be more robust to permutation than a pure sequential encoder (e.g., Mamba). (3) While the quadratic time complexity of attentions can dominate the cost of PE/SE in GTs, complex PE/SE (with _O_ ( _n_[2] ) cost) can be a bottleneck for scaling Graph Mamba on large graphs. 

**Contributions.** To address all the abovementioned limitations, we present Graph Mamba Networks (GMNs), a new class of machine learning on graphs based on state space models (Figure 1 shows the schematic of the GMNs). In summary our contributions are: 

- **Recipe for Graph Mamba Networks.** We discuss new challenges of GMNs compared to GTs in architecture design and motivate our recipe with four required and one optional steps to design GMNs. In particular, its steps are (1) Tokenization, (2) Token Ordering, (3) Local Encoding, (4) Bidirectional Selective SSM Encoder and dispensable (5) PE and SE. 

- **An Efficient Tokenization for Bridging Frameworks.** Literature lacks a common foundation about what constitutes a good tokenization. Accordingly, architectures are required to choose either node- or subgraph-level tokenization, while each of which has its own (dis)advantages, depending on the data. We present a graph tokenization process that not only is fast and efficient, but it also bridges the node- and subgraph-level tokenization methods using a single parameter. 

2 

**Graph Mamba** 

**==> picture [535 x 253] intentionally omitted <==**

**----- Start of picture text -----**<br>
Tokenization PE/SE Local Encoding Token Ordering Bidirectional Mamba<br>Random Walk   For each node and !𝑚= 1, … , 𝑚,  PE/SE  (1) Sum over the rows of  MPNNs   To vectorize  𝒎≥𝟏 Tokens have  Robust to Permutation  We<br>we sample 𝑀 walks with length !𝑚 and consider  non-diagonal elements of the  each token one can use  implicit order due to  scan the sequence of tokens<br>their induced subgraph as a token. For each !𝑚,  random walk matrix.  message-passing to  hierarchical structure. in two directions.<br>𝒎 =  𝟎we repeat the process : Each node is an independent token.𝑠 times. (2) Eigenvectors of the Laplacian. (3) Anonymous random walk encoding, i.e., counting the  incorporate local information. 𝒎= 𝟎 Sort nodes  ⊕<br>number of times a node appears at  RWF   Given the walks  based on PRR/Degree.<br>Allows switching between node and subgraph tokenization using a single parameter making the choice of tokenization a tunable hyperparameter during training. 𝑚,  Relative PE/SE   difference of PE/SE as edge features.  a certain position.Using pair-wise  that corresponds to a subgraph, one can use local identity relation of nodes to vectorize it. ( Implicit order  $𝑚= 𝑚) ( $𝑚= 1) × × +<br>For each Token: … … …<br>𝜎<br>s s 𝜎 𝜎<br>… … … … 𝒎= 𝟎<br>Concatenation<br>…<br>s subgraphs  s subgraphs  s  subgraphs  small large<br>   ( !𝑚= 1)   ( !𝑚= 2)   ( !𝑚= 𝑚) PRR/Degree/… MPNN<br>PNA<br>⊕ Gated GCN<br>Long Sequence  Mamba shows performance  Optional PE/SE   When using  Features   When  node  Domain Knowledge  Sum/Concatenation GINE<br>improvement with longer sequences,  and so we  subgraph tokens (i.e., !𝑚≥1), PE/SE  or edge features are  One can use domain  𝜎 Activation Function<br>use parameter   s  to control the length of the  is optional. That is, tokens have  available, one can  knowledge (when<br>subgraph sequence. Based on the dataset, one  their own inductive bias, and do not  concatenate them with  adapting GMs to  Linear Layer<br>can tune 𝑠 to achieve better results. need additional information about  the PE/SE, before the  specific domain) or  1-d Convolution<br>the graph structure.  local encoding step.  structural properties  Selective SSM<br>like Personalized  Required Step<br>PageRank or degree. Optional Step<br>Key Points<br>⊕<br>**----- End of picture text -----**<br>


_Figure 1._ Schematic of the GMNs with four required and one optional steps: (1) Tokenization: the graph is mapped into a sequence of tokens ( _m ≥_ 1: subgraph and _m_ = 0: node tokenization) (2) (Optional Step) PE/SE: inductive bias is added to the architecture using information about the position of nodes and the strucutre of the graph. (3) Local Encoding: local structures around each node are encoded using a subgraph vectorization mechanism. (4) Token Ordering: the sequence of tokens are ordered based on the context. (Subgraph tokenization ( _m ≥_ 1) has implicit order and does not need this step). (5) (Stack of) Bidirectional Mamba: it scans and selects relevant nodes or subgraphs to flow into the hidden states. _[†]_ In this figure, the last layer of bidirectional Mamba, which performs as a readout on all nodes, is omitted for simplicity. 

Moreover, the presented tokenization has implicit order, which is specially important for sequential encoders like SSMs. 

- **New Bidirectional SSMs for Graphs.** Inspired by Mamba, we design a SSM architecture that scans the input sequence in two different directions, making the model more robust to permutation, which is particularly important when we do not use implicitly ordered tokenization on graphs. 

- **Theoretical Justification.** We provide theoretical justification for the power of GMNs and show that they are universal approximator of any functions on graphs. We further show that GMNs using proper PE/SE is more expressive than any WL test, matching GTs in this manner. 

- **Outstanding Performance and New Insights.** Our experimental evaluations demonstrate that GMNs attain an outstanding performance in long-range, small-scale, large-scale, and heterophilic benchmark datasets, while consuming less GPU memory. These results show that while Transformers, complex message-passing, and SE/PE are sufficient for good performance in practice, neither is necessary. We further perform ablation study and validate the contribution of each architectural choice. 

## **2. Related Work and Backgrounds** 

To situate GMNs in a broader context, we discuss four relevant types of machine learning methods: 

## **2.1. Message-Passing Neural Networks** 

Message-passing neural networks are a class of GNNs that iteratively aggregate local neighborhood information to learn the node/edge representations (Kipf & Welling, 2016). MPNNs have been the dominant paradigm in machine learning on graphs, and attracts much attention, leading to various powerful architectures, e.g., GAT (Velickoviˇ c et al.´ , 2018), GCN (Henaff 

3 

**Graph Mamba** 

et al., 2015; Kipf & Welling, 2016), GatedGCN (Bresson & Laurent, 2017), GIN (Xu et al., 2019), etc. Simple MPNNs, however, are known to suffer from some major limitations including: (1) limiting their expressivity to the 1-WL isomorphism test (Xu et al., 2019), (2) over-smoothing (Rusch et al., 2023), and (3) over-squashing (Alon & Yahav, 2021; Di Giovanni et al., 2023). Various methods have been developed to augment MPNNs and overcome such issues, including higher-order GNNs (Morris et al., 2019; 2020), graph rewiring (Gutteridge et al., 2023; Arnaiz-Rodr´ıguez et al., 2022), adaptive and cooperative GNNs (Errica et al., 2023; Finkelshtein et al., 2023), and using additional features (Sato et al., 2021; Murphy et al., 2019). 

## **2.2. Graph Transformers** 

With the rise of Transformer architectures (Vaswani et al., 2017) and their success in diverse applications such as natural language processing (Wolf et al., 2020) and computer vision (Liu et al., 2021), their graph adaptations have gained popularity as the alternatives of MPNNs (Yun et al., 2019; Kim et al., 2022; Rampa´sek et al.ˇ , 2022). Using a full global attention, GTs consider each pair of nodes connected (Yun et al., 2019) and so are expected to overcome the problems of oversquashing and over-smoothing in MPNNs (Kreuzer et al., 2021). GTs, however, have weak inductive bias and needs proper positional/structural encoding to learn the structure of the graph (Kreuzer et al., 2021; Rampa´sek et al.ˇ , 2022). To this end, various studies have focused on designing powerful positional and structural encodings (Wang et al., 2022; Ying et al., 2021; Kreuzer et al., 2021; Shiv & Quirk, 2019). 

**Sparse Attention.** While GTs have shown outstanding performance in different graph tasks on small-scale datasets (up to 10K nodes), their quadratic computational cost, caused by their full global attention, has limited their applicability to large-scale graphs (Rampa´sek et al.ˇ , 2022). Motivated by linear attention mechanisms (e.g., BigBird (Zaheer et al., 2020) and Performer (Choromanski et al., 2021)), which are designed to overcome the same scalability issue of Transformers on long sequences, using sparse Transformers in GT architectures has gained popularity (Rampa´sek et al.ˇ , 2022; Shirzad et al., 2023; Kong et al., 2023; Liu et al., 2023; Wu et al., 2023). The main idea of sparse GTs models is to restrict the attention pattern, i.e., the pairs of nodes that can interact with each other. As an example, Shirzad et al. (2023) present Exphormer, the graph adaption of BigBird that uses three sparse patterns of (1) expander graph attention, (2) local attention among neighbors, and (3) global attention by connecting virtual nodes to all non-virtual nodes. 

**Subgraph Tokenization.** Another method to overcome GTs’ high computational cost is to use subgraph tokenization (Chen et al., 2023; Zhao et al., 2021; Baek et al., 2021; He et al., 2023), where tokens are small subgraphs extracted with a pre-defined strategy. These subgraph tokenization strategies usually are _k_ -hop neighborhood (given a fixed _k_ ) (Nguyen et al., 2022a; Hussain et al., 2022; Park et al., 2022), learnable sample of neighborhood (Zhang et al., 2022), ego-networks (Zhao et al., 2021), hierarchical _k_ -hop neighborhoods (Chen et al., 2023), graph motifs (Rong et al., 2020), and graph partitions (He et al., 2023). To vectorize each token, subgraph-based GT methods typically rely on MPNNs, making them vulnerable to over-smoothing and over-squashing. Most of them also use a fixed neighborhood of each node, missing the hierarchical structure of the graph. The only exception is NAGphormer (Chen et al., 2023) that uses all _k_ = 1 _, . . . , K_ -hop neighborhoods of each node as its corresponding tokens. Although this tokenization lets the model learn the hierarchical structure of the graph, by increasing the hop of the neighborhood, its tokens become exponentially larger, limiting its ability to scale to large graphs. 

## **2.3. State Space Models** 

State Space Models (SSMs), a type of sequence models, are usually known as linear time-invariant systems that map input sequence _x_ ( _t_ ) _∈_ R _[L]_ to response sequence _y_ ( _t_ ) _∈_ R _[L]_ (Aoki, 2013). Specifically, SSMs use a latent state _h_ ( _t_ ) _∈_ R _[N][×][L]_ , evolution parameter **A** _∈_ R _[N][×][N]_ , and projection parameters **B** _∈_ R _[N][×]_[1] _,_ **C** _∈_ R[1] _[×][N]_ such that: 

**==> picture [297 x 26] intentionally omitted <==**

Due to the hardness of solving the above differential equation in deep learning settings, discrete space state models (Gu et al., 2020; Zhang et al., 2023b) discretize the above system using a parameter **∆** : 

**==> picture [289 x 26] intentionally omitted <==**

4 

**Graph Mamba** 

where 

**==> picture [324 x 29] intentionally omitted <==**

Gu et al. (2020) shows that discrete-time SSMs are equivalent to the following convolution: 

**==> picture [318 x 27] intentionally omitted <==**

and accordingly can be computed very efficiently. Structured state space models (S4), another type of SSMs, are efficient alternatives of attentions and have improved efficiency and scalability of SSMs using reparameterization (Gu et al., 2022; Fu et al., 2023; Nguyen et al., 2023). SSMs show promising performance on timeseries data (Zhang et al., 2023b; Tang et al., 2023), Genomic sequence (Nguyen et al., 2023), healthcare domain (Gu et al., 2021), and computer vision (Gu et al., 2021; Nguyen et al., 2022b). They, however, lack selection mechanism, causing missing the context as discussed by Gu & Dao (2023). Recently, Gu & Dao (2023) introduce an efficient and powerful selective structured state space architecture, called MAMBA, that uses recurrent scans along with a selection mechanism to control which part of the sequence can flow into the hidden states. The selection mechanism of Mamba can be interpreted as using data-dependent state transition mechanisms, i.e., making **B** _,_ **C** _,_ and **∆** as function of input _xt_ . Mamba outstanding performance in language modeling, outperforming Transformers of the same size and matching Transformers twice its size, motivates several recent studies to adapt its architecture for different data modalities and tasks (Liu et al., 2024b; Yang et al., 2024; Zhu et al., 2024; Ahamed & Cheng, 2024; Xing et al., 2024; Liu et al., 2024a; Ma et al., 2024). 

## **3. Challenges & Motivations: Transformers vs Mamba** 

Mamba architecture is specifically designed for sequence data and the complex non-causal nature of graphs makes directly applying Mamba on graphs challenging. Based on the common applicability of Mamba and Transformers on tokenized sequential data, a straightforward approach to adapt Mamba for graphs is to replace Transformers with Mamba in GTs frameworks, e.g., TokenGT (Kim et al., 2022) or GPS (Rampa´sek et al.ˇ , 2022). However, this approach might not fully take advantage of selective SSMs due to ignoring some of their special traits. In this section, we discuss new challenges for GMNs compared to GTs. 

**Sequences vs 2-D Data.** It is known that the self-attentive architecture corresponds to a family of permutation equivariant functions (Lee et al., 2019; Liu et al., 2020). That is, the attention mechanism in Transformers (Vaswani et al., 2017) assumes a connection between each pair of tokens, regardless of their positions in the sequence, making it permutation equivariant. Accordingly, Transformers lack inductive bias and so properly positional encoding is crucial for their performance, whenever the order of tokens matter (Vaswani et al., 2017; Liu et al., 2020). On the other hand, Mamba is a sequential encoder and scans tokens in a recurrent manner (potentially less sensitive to positional encoding). Thus, it expects causal data as an input, making it challenging to be adapted to 2-D (e.g., images) (Liu et al., 2024b) or complex graph-structured data. Accordingly, while in graph adaption of Transformers mapping the graph into a sequence of tokens along with a positional/structural encodings were enough, sequential encoders, like SSMs, and more specifically Mamba, require an ordering mechanism for tokens. 

Although this sensitivity to the order of tokens makes the adaption of SSMs to graphs challenging, it can be more powerful whenever the order matters. For example, learning the hierarchical structures in the neighborhood of each node ( _k_ -hops for _k_ = 1 _, . . . , K_ ), which is implicitly ordered, is crucial in different domains (Zhong et al., 2022; Lim et al., 2023b). Moreover, it provides the opportunity to use domain knowledge when the order matters (Yu et al., 2020). In our proposed framework, we provide the opportunity for both cases: (1) using domain knowledge or structural properties (e.g., Personalized PageRank (Page et al., 1998)) when the order matters, or (2) using implicitly ordered subgraphs (no ordering is needed). Furthermore, our bidirectional encoder scans nodes in two different directions, being capable of learning equivariance functions on the input, whenever it is needed. 

**Long-range Sequence Modeling.** In graph domain, the sequence of tokens, either node, edge, or subgraph, can be counted as the context. Unfortunately, Transformer architecture, and more specifically GTs, are not scalable to long sequence. Furthermore, intuitively, more context (i.e., longer sequence) should lead to better performance; however, recently it has been empirically observed that many sequence models do not improve with longer context in language modeling (Shi et al., 

5 

**Graph Mamba** 

2023). Mamba, because of its selection mechanism, can simply filter irrelevant information and also reset its state at any time. Accordingly, its performance improves monotonically with sequence length (Gu & Dao, 2023). To this end, and to fully take advantage of Mamba, one can map a graph or node to long sequences, possibly bags of various subgraphs. Not only the long sequence of tokens can provide more context, but it also potentially can improve the expressive power (Bevilacqua et al., 2022). 

**Scalability.** Due to the complex nature of graph-structured data, sequential encoders, including Transformers and Mamba, require proper positional and structural encodings (Rampa´sek et al.ˇ , 2022; Kim et al., 2022). These PEs/SEs, however, often have quadratic computational cost, which can be computed once before training. Accordingly, due to the quadratic time complexity of Transformers, computing these PEs/SEs was dominated and they have not been the bottleneck for training GTs. GMNs, on the other hand, have linear computational cost (with respect to both time and memory), and so constructing complex PEs/SEs can be their bottleneck when training on very large graphs. This bring a new challenge for GMNs, as they need to either (1) do not use PEs/SEs, or (2) use their more efficient variants to fully take advantage of SSMs efficiency. Our architecture design make the use of PE/SE optional and our empirical evaluation shows that GMNs without PE/SE can achieve competitive performance compared to methods with complex PEs/SEs. 

**Node or Subgraph** ? In addition to the above new challenges, there is a lack of common foundation about what constitutes a good tokenization, and what differentiates them, even in GT frameworks. Existing methods use either node/edge (Shirzad et al., 2023; Rampa´sek et al.ˇ , 2022; Kim et al., 2022), or subgraph tokenization methods (Chen et al., 2023; Zhao et al., 2021; He et al., 2023). While methods with node tokenization are more capable of capturing long-range dependencies, methods with subgraph tokens have more ability to learn local neighborhoods, are less rely on PE/SE (Chen et al., 2023), and are more efficient in practice. Our architecture design lets switching between node and subgraph tokenization using a single parameter _m_ , making the choice of tokenization a tunable hyperparameter during training. 

## **4. Graph Mamba Networks** 

In this section, we provide our five-step recipe for powerful, flexible, and scalable Graph Mamba Networks. Following the discussion about the importance of each step, we present our architecture. The overview of the GMN framework is illustrated in Figure 1. 

Throughout this section, we let _G_ = ( _V, E_ ) be a graph, where _V_ = _{v_ 1 _, . . . , vn}_ is the set of nodes and _E ⊆ V × V_ is the set of edges. We assume each node _v ∈ V_ has a feature vector **x**[(0)] _v ∈_ **X** , where **X** _∈_ R _[n][×][d]_ is the feature matrix describing the attribute information of nodes and _d_ is the dimension of feature vectors. Given _v ∈ V_ , we let _N_ ( _v_ ) = _{u|_ ( _v, u_ ) _∈ E}_ be the set of _v_ ’s neighbors. Given a subset of nodes _S ⊆ V_ , we use _G_ [ _S_ ] to denote the induced subgraph constructed by nodes in _S_ , and **X** _S_ to denote the feature matrix describing the attribute information of nodes in _S_ . 

## **4.1. Tokenization and Encoding** 

Tokenization, which is the process of mapping the graph into a sequence of tokens, is an inseparable part of adapting sequential encoders to graphs. As discussed earlier, existing methods use either node/edge (Shirzad et al., 2023; Rampa´sekˇ et al., 2022; Kim et al., 2022), or subgraph tokenization methods (Chen et al., 2023; Zhao et al., 2021; He et al., 2023), each of which has its own (dis)advantages. In this part, we present a new simple but flexible and effective neighborhood sampling for each node and discuss its advantages over existing subgraph tokenization. The main and high-level idea of our tokenization is to first, sample some subgraphs for each node that can represent the node’s neighborhood structure as well as its local, and global positions in the graph. Then we vectorize (encode) these subgraphs to obtain the node representations. 

**Neighborhood Sampling.** Given a node _v ∈ V_ , and two integers _m, M ≥_ 0, for each 0 _≤ m_ ˆ _≤ m_ , we sample _M_ random walks started from _v_ with length _m_ ˆ . Let _T_ ˆ _m,i_ ( _v_ ) for _i_ = 0 _, . . . , M_ be the set of visited nodes in the _i_ -th walk. We define the token corresponds to all walks with length _m_ ˆ as: 

**==> picture [308 x 31] intentionally omitted <==**

which is the union of all walks with length _m_ ˆ . One can interpret _G_ [ _T_ ˆ _m_ ( _v_ )] as the induced subgraph of a sample of _m_ ˆ -hop neighborhood of node _v_ . At the end, for each node _v ∈ V_ we have the sequence of _G_ [ _T_ 0( _v_ )] _, . . . , G_ [ _Tm_ ( _v_ )] as its corresponding tokens. 

6 

**Graph Mamba** 

Using random walks (with fixed length) or _k_ -hop neighborhood of a node as its representative tokens has been discussed in several recent studies (Ding et al., 2023; Zhang et al., 2022; Chen et al., 2023; Zhao et al., 2021). These methods, however, suffer from a subset of these limitations: (1) they use a fixed-length random walk (Kuang et al., 2021), which misses the hierarchical structure of the node’s neighborhood. This is particularly important when the long-range dependencies of nodes are important. (2) they use all nodes in all _k_ -hop neighborhoods (Chen et al., 2023; Ding et al., 2023), resulting in a trade-off between long-range dependencies and over-smoothing or over-squashing problems. Furthermore, the _k_ -hop neighborhood of a well-connected node might be the whole graph, resulting in considering the graph as a token of a node, which is inefficient. Our neighborhood sampling approach addresses all these limitations. It sampled the fixed number of random walks with different lengths for all nodes, capturing hierarchical structure of the neighborhood while avoiding both inefficiency, caused by considering the entire graph, and over-smoothing and over squashing, caused by large neighborhood aggregation. 

**Why Not More Subgraphs** ? As discussed earlier, empirical evaluation has shown that the performance of _selective_ state space models improves monotonically with sequence length (Gu & Dao, 2023). Furthermore, their linear computational cost allow us to use more tokens, providing them more context. Accordingly, to fully take advantage of selective state space models, given an integer _s >_ 0, we repeat the above neighborhood sampling process for _s_ times. Accordingly, for each node _v ∈ V_ we have a sequence of 

**==> picture [263 x 25] intentionally omitted <==**

as its corresponding sequence of tokens. Here, we can see another advantage of our proposed neighborhood sampling compared to Chen et al. (2023); Ding et al. (2023). While in NAGphormer (Chen et al., 2023) the sequence length of each node is limited by the diameter of the graph, our method can produce a long sequence of diverse subgraphs. 

**Theorem 4.1.** _With large enough M, m, and s >_ 0 _, GMNs’ neighborhood sampling is strictly more expressive than k-hop neighborhood sampling._ 

**Structural/Positional Encoding** . To further augment our framework for Graph Mamba, we consider an optional step, when we inject structural and positional encodings to the initial features of nodes/edges. PE is meant to provide information about the position of a given node within the graph. Accordingly, two close nodes within a graph or subgraph are supposed to have close PE. SE, on the other hand, is meant to provide information about the structure of a subgraph. Following Rampa´sekˇ et al. (2022), we concatenate either eigenvectors of the graph Laplacian or Random-walk structural encodings to the nodes’ feature, whenever PE/SE are needed: i.e., 

**==> picture [280 x 12] intentionally omitted <==**

where _pv_ is the corresponding positional encoding to _v_ . For the sake of consistency, we use **x** _v_ instead of **x**[(new)] _v_ throughout the paper. 

**Neighborhood Encoding.** Given a node _v ∈ V_ and its sequence of tokens (subgraphs), we encode the subgraph via encoder _ϕ_ ( _._ ). That is, we construct **x**[1] _v[,]_ **[ x]**[2] _v[, . . . ,]_ **[ x]** _[ms] v[−]_[1] _,_ **x** _[ms] v ∈_ R _[d]_ as follows: 

**==> picture [322 x 19] intentionally omitted <==**

where 1 _≤ i ≤ m_ and 1 _≤ j ≤ s_ . In practice, this encoder can be an MPNN, (e.g., Gated-GCN (Bresson & Laurent, 2017)), or RWF (Tonshoff et al.¨ , 2023b) that encodes nodes with respect to a sampled set of walks into feature vectors with four parts: (1) node features, (2) edge features along the walk, and (3, 4) local structural information. 

**Token Ordering.** By Equation 7, we can calculate the neighborhood embeddings for various sampled neighborhoods of a node and further construct a sequence to represent its neighborhood information, i.e., **x**[1] _v[,]_ **[ x]**[2] _v[, . . . ,]_ **[ x]** _[ms] v[−]_[1] _,_ **x** _[ms] v_[.][As discussed] in §3, adaption of sequence models like SSMs to graph-structured data requires an order on the tokens. To understand what constitutes a good ordering, we need to recall selection mechanism in Mamba (Gu & Dao, 2023) (we will discuss selection mechanism more formally in §4.2). Mamba by making **B** _,_ **C** _,_ and **∆** as functions of input _xt_ (see §2.3 for notations) lets the model filter irrelevant information and select important tokens in a recurrent manner, meaning that each token gets updated based on tokens that come before them in the sequence. Accordingly, earlier tokens have less information about the context of sequence, while later tokens have information about almost entire sequence. This leads us to order tokens based on either their needs of knowing information about other tokens or their importance to our task. 

7 

**Graph Mamba** 

When _m ≥_ 1: For the sake of simplicity first let _s_ = 1. In the case that _m ≥_ 1, interestingly, our architecture design provides us with an implicitly ordered sequence. That is, given _v ∈ V_ , the _i_ -th token is a samples from _i_ -hop neighborhood of node _v_ , which is the subgraph of all _j_ -hop neighborhoods, where _j ≥ i_ . This means, given a large enough _M_ (number of sampled random walks), our _Tj_ ( _v_ ) has enough information about _Ti_ ( _v_ ), not vice versa. To this end, we use the reverse of initial order, i.e., **x** _[m] v[,]_ **[ x]** _[m] v[−]_[1] _, . . . ,_ **x**[2] _v[,]_ **[ x]**[1] _v_[.][Accordingly, inner subgraphs can also have information about the global structure.] When _s ≥_ 2, we use the same procedure as above, and reverse the initial order, i.e., **x** _[sm] v[,]_ **[ x]** _[sm] v[−]_[1] _, . . . ,_ **x**[2] _v[,]_ **[ x]**[1] _v_[.][To make our] model robust to the permutation of subgraphs with the same walk length _m_ ˆ , we randomly shuffle them. We will discuss the ordering in the case of _m_ = 0 later. 

## **4.2. Bidirectional Mamba** 

As discussed in §3, SSMs are recurrent models and require ordered input, while graph-structured data does not have any order and needs permutation equivariant encoders. To this end, inspired by Vim in computer vision (Zhu et al., 2024), we modify Mamba architecture and use two recurrent scan modules to scan data in two different directions (i.e., forward and backward). Accordingly, given two tokens _ti_ and _tj_ , where _i > j_ and indices show their initial order, in forward scan _ti_ comes after _tj_ and so has the information about _tj_ (which can be flown into the hidden states or filtered by the selection mechanism). In backward pass _tj_ comes after _ti_ and so has the information about _ti_ . This architecture is particularly important when _m_ = 0 (node tokenization), which we will discuss later. 

More formally, in forward pass module, let **Φ** be the input sequence (e.g., given _v_ , **Φ** is a matrix whose rows are **x** _[sm] v[,]_ **[ x]** _[sm] v[−]_[1] _, . . . ,_ **x**[1] _v_[, calculated in Equation][ 7][),] **[ A]**[ be the relative positional encoding of tokens, we have:] 

**==> picture [390 x 87] intentionally omitted <==**

where **W** _,_ **WB** _,_ **WC** _,_ **W∆** _,_ **W** forward _,_ 1 and **W** forward _,_ 2 are learnable parameters, _σ_ ( _._ ) is nonlinear function (e.g., SiLU), LayerNorm( _._ ) is layer normalization (Ba et al., 2016), SSM( _._ ) is the state space model discussed in Equations 2 and 4, and Discrete( _._ ) is discretization process discussed in Equation 3. We use the same architecture as above for the backward pass (with different weights) but instead we use **Φ** inverse as the input, which is a matrix whose rows are **x**[1] _v[,]_ **[ x]**[2] _v[, . . . ,]_ **[ x]** _[sm] v_[.][Let] _**y**_ backward be the output of this backward module, we obtain the final encodings as 

**==> picture [319 x 13] intentionally omitted <==**

In practice, we stack some layers of the bidirectional Mamba to achieve good performance. Note that due to our ordering mechanism, the last state of the output corresponds to the walk with length _m_ ˆ = 0, i.e., the node itself. Accordingly, the last state represents the updated node encoding. 

**Augmentation with MPNNs** . We further use an optional MPNN module that simultaneously performs message-passing and augments the output of the bidirectional Mamba via its inductive bias. Particularly this module is very helpful when there are rich edge features and so an MPNN can help to take advantage of them. While in our empirical evaluation we show that this module is not necessary for the success of GMNs in several cases, it can be useful when we avoid complex PE/SE and strong inductive bias is needed. 

**How Does Selection Work on Subgraphs** ? As discussed earlier, the selection mechanism can be achieved by making **B** _,_ **C** _,_ and **∆** as the functions of the input data (Gu & Dao, 2023). Accordingly, in recurrent scan, based on the input, the model can filter the irrelevant context. The selection mechanism in Equation 9 is implemented by making **B** _,_ **C** _,_ and **∆** as functions of **Φ** input, which is matrix of the encodings of neighborhoods. Therefore, as model scans the sampled subgraphs from the _i_ -hop neighborhoods in descending order of _i_ , it filters irrelevant neighborhoods to the context (last state), which is the node encoding. 

**Last Layer(s) of Bidirectional Mamba.** To capture the long-range dependencies and to flow information across the nodes, we use the node encodings obtained from the last state of Equation 9 as the input of the last layer(s) of bidirectional Mamba. 

8 

**Graph Mamba** 

Therefore, the recurrent scan of nodes (in both directions) can flow information across nodes. This design not only helps capturing long-range dependencies in the graph, but it also is a key to the flexibility of our framework to bridge node and subgraph tokenization. 

## **4.3. Tokenization When** _m_ = 0 

In this case, for each node _v ∈ V_ we only consider _v_ itself as its corresponding sequence of tokens. Based on our architecture, in this case, the first layers of bidirection Mamba become simple projection as the length of the sequence is one. However, the last layers, where we use node encodings as their input, treats nodes as tokens and become an architecture that use a sequential encoder (e.g., Mamba) with node tokenization. More specifically, in this special case of framework, the model is the adaption of GPS (Ramp´aˇsek et al., 2022) framework, when we replace its Transformer with our bidirectional Mamba. 

This architecture design lets switching between node and subgraph tokenization using a single parameter _m_ , making the choice of tokenization a tunable hyperparameter during training. Note that this flexibility comes more from our architecture rather than the method of tokenization. That is, in practice one can use only 0-hop neighborhood in NAGphormer (Chen et al., 2023), resulting in only considering the node itself. However, in this case, the architecture of NAGphormer becomes a stack of MLPs, resulting in poor performance. 

**Token Ordering.** When _m_ = 0: One remaining question is how one can order nodes when we use node tokenization. As discussed in §4.1, tokens need to be ordered based on either (1) their needs of knowing information about other tokens or (2) their importance to our task. When dealing with nodes and specifically when long-range dependencies matter, (1) becomes a must for all nodes. Our architecture overcomes this challenge by its bidirectional scan process. Therefore, we need to order nodes based on their importance. There are several metrics to measure the importance of nodes in a graph. For example, various centrality measures (Latora & Marchiori, 2007; Ruhnau, 2000), degree, _k_ -core (Lick & White, 1970; Hashemi et al., 2022), Personalized PageRank or PageRank (Page et al., 1998), etc. In our experiments, for the sake of efficiency and simplicity, we sort nodes based on their degree. 

**How Does Selection Work on Nodes** ? Similar to selection mechanism on subgraphs, the model based on the input data can filter irrelevant tokens (nodes) to the context (downstream tasks). 

## **4.4. Theoretical Analysis of GMNs** 

In this section, we provide theoretical justification for the power of GMNs. More specifically, we first show that GMNs are universal approximator of any function on graphs. Next, we discuss that given proper PE and enough parameters, GMNs are more powerful than any WL isomorphism test, matching GTs (with the similar assumptions). Finally, we evaluate the expressive power of GMNs when they do not use any PE or MPNN and show that their expressive power is unbounded (might be incomparable). 

**Theorem 4.2** (Universality) **.** _Let_ 1 _≤ p < ∞, and ϵ >_ 0 _. For any continues function f_ : [0 _,_ 1] _[d][×][n] →_ R _[d][×][n] that is permutation equivariant, there exists a GMN with positional encoding, gp, such that ℓ[p]_ ( _f, g_ ) _< ϵ_[1] _._ 

**Theorem 4.3** (Expressive Power w/ PE) **.** _Given the full set of eigenfunctions and enough parameters, GMNs can distinguish any pair of non-isomorphic graphs and are more powerful than any WL test._ 

We prove the above two theorems based on the recent work of Wang & Xue (2023), where they prove that SSMs with layer-wise nonlinearity are universal approximators of any sequence-to-sequence function. 

**Theorem 4.4** (Expressive Power w/o PE and MPNN) **.** _With enough parameters, for every k ≥_ 1 _there are graphs that are distinguishable by GMNs, but not by k-WL test, showing that their expressive power is not bounded by any WL test._ 

We prove the above theorem based on the recent work of Tonshoff et al.¨ (2023b), where they prove a similar theorem for CRaWl (Tonshoff et al.¨ , 2023b). Note that this theorem does not rely on the Mamba’s power, and the expressive power comes from the choice of neighborhood sampling and encoding. 

> 1 _ℓp_ ( _._ ) is the _p_ -norm 

9 

**Graph Mamba** 

_Table 1._ Benchmark on Long-Range Graph Datasets (Dwivedi et al., 2022). Highlighted are the top **first** , **second** , and **third** results. 

|**Model**|**COCO-SP**<br>F1 score_↑_|**PascalVOC-SP**<br>F1 score_↑_|**Peptides-Func**<br>AP_↑_|**Peptides-Struct**<br>MAE_↓_|
|---|---|---|---|---|
|GCN|0_._0841_±_0_._0010|0_._1268_±_0_._0060|0_._5930_±_0_._0023|0_._3496_±_0_._0013|
|GIN|0_._1339_±_0_._0044|0_._1265_±_0_._0076|0_._5498_±_0_._0079|0_._3547_±_0_._0045|
|Gated-GCN|0_._2641_±_0_._0045|0_._2873_±_0_._0219|0_._5864_±_0_._0077|0_._3420_±_0_._0013|
|CRaWl|0_._3219_±_0_._00106|0_._4088_±_0_._0079|0_._6963_±_0_._0079|0_._2506_±_0_._0022|
|SAN+LapPE|0_._2592_±_0_._0158|0_._3230_±_0_._0039|0_._6384_±_0_._0121|0_._2683_±_0_._0043|
|NAGphormer|0_._3458_±_0_._0070|0_._4006_±_0_._0061|-|-|
|Graph ViT|-|-|0_._6855_±_0_._0049|0_._2468_±_0_._0015|
|GPS|0_._3774_±_0_._0150|0_._3689_±_0_._0131|0_._6575_±_0_._0049|0_._2510_±_0_._0015|
|GPS (BigBird)|0_._2622_±_0_._0008|0_._2762_±_0_._0069|0_._5854_±_0_._0079|0_._2842_±_0_._0130|
|Exphormer|0_._3430_±_0_._0108|0_._3975_±_0_._0037|0_._6527_±_0_._0043|0_._2481_±_0_._0007|
|GPS + Mamba|0_._3895_±_0_._0125|0_._4180_±_0_._012|0_._6624_±_0_._0079|0_._2518_±_0_._0012|
|GMN-|0_._3618_±_0_._0053|0_._4169_±_0_._0103|0_._6860_±_0_._0012|0_._2522_±_0_._0035|
|GMN|0_._3974_±_0_._0101|0_._4393_±_0_._0112|0_._7071_±_0_._0083|0_._2473_±_0_._0025|



## **5. Experiments** 

In this section, we evaluate the performance of GMNs in long-range, small-scale, large-scale, and heterophilic benchmark datasets. We further discuss its memory efficiency and perform ablation study to validate the contribution of each architectural choice. The detailed statistics of datasets and additional experiments are available in the appendix. 

## **5.1. Experimental Setup** 

**Dataset.** We use three most commonly used benchmark datasets with long-range, small-scale, large-scale, and heterophilic properties. For long-range datasets, we use Longe Range Graph Benchmark (LRGB) dataset (Dwivedi et al., 2022). For small and large-scale datasets, we use GNN benchmark (Dwivedi et al., 2023). To evaluate the GMNs on heterophilic graphs, we use four heterophilic datasets from the work of Platonov et al. (2023). Finally, we use a large dataset from Open Graph Benchmark (Hu et al., 2020). We evaluate the performance of GMNs on various graph learning tasks (e.g., graph classification, regression, node classification and link classification). Also, for each datasets we use the propose metrics in the original benchmark and report the metric across multiple runs, ensuring the robustness. We discuss datasets, their statistics and their tasks in Appendix A. 

**Baselines.** We compare our GMNs with (1) MPNNs, e.g., GCN (Kipf & Welling, 2016), GIN (Xu et al., 2019), and Gated-GCN (Bresson & Laurent, 2017), (2) Random walk based method CRaWl (Tonshoff et al.¨ , 2023b), (3) state-of-the-art GTs, e.g., SAN (Kreuzer et al., 2021), NAGphormer (Chen et al., 2023), Graph ViT (He et al., 2023), two variants of GPS (Rampa´sek et al.ˇ , 2022), GOAT (Kong et al., 2023), and Exphormer (Shirzad et al., 2023), and (4) our baselines (i) GPS + Mamba: when we replace the transformer module in GPS with bidirectional Mamba. (ii) GMN-: when we do not use PE/SE and MPNN. The details of baselines are in Appendix B. 

## **5.2. Long Range Graph Benchmark** 

Table 1 reports the results of GMNs and baselines on long-range graph benchmark. GMNs consistently outperform baselines in all datasets that requires long-range dependencies between nodes. The reason for this superior performance is three folds: (1) GMNs based on our design use long sequence of tokens to learn node encodings and then use another selection mechanism to filter irrelevant nodes. The provided long sequence of tokens enables GMNs to learn long-range dependencies, without facing scalability or over-squashing issues. (2) GMNs using their selection mechanism are capable of filtering the neighborhood around each node. Accordingly, only informative information flows into hidden states. (3) The random-walk based neighborhood sampling allow GMNs to have diverse samples of neighborhoods, while capturing the hierarchical nature of _k_ -hop neighborhoods. Also, it is notable that GMN consistently outperforms our baseline GPS + Mamba, which shows the importance of paying attention to the new challenges. That is, replacing the transformer module with Mamba, while improves the performance, cannot fully take advantage of the Mamba traits. Interestingly, GMN-, a variant of GMNs 

10 

**Graph Mamba** 

_Table 2._ Benchmark on GNN Benchmark Datasets (Dwivedi et al., 2023). Highlighted are the top **first** , **second** , and **third** results. 

|**Model**|**MNIST**<br>Accuracy_↑_|**CIFAR10**<br>Accuracy_↑_|**PATTERN**<br>Accuracy_↑_|**MalNet-Tiny**<br>Accuracy_↑_|
|---|---|---|---|---|
|GCN|0_._9071_±_0_._0021|0_._5571_±_0_._0038|0_._7189_±_0_._0033|0_._8100_±_0_._0000|
|GIN|0_._9649_±_0_._0025|0_._5526_±_0_._0152|0_._8539_±_0_._0013|0_._8898_±_0_._0055|
|Gated-GCN|0_._9734_±_0_._0014|0_._6731_±_0_._0031|0_._8557_±_0_._0008|0_._9223_±_0_._0065|
|CRaWl|0_._9794_±_0_._050|0_._6901_±_0_._0259|-|-|
|NAGphormer|-|-|0_._8644_±_0_._0003|-|
|GPS|0_._9811_±_0_._0011|0_._7226_±_0_._0031|0_._8664_±_0_._0011|0_._9298_±_0_._0047|
|GPS (BigBird)|0_._9817_±_0_._0001|0_._7048_±_0_._0010|0_._8600_±_0_._0014|0_._9234_±_0_._0034|
|Exphormer|0_._9855_±_0_._0003|0_._7469_±_0_._0013|0_._8670_±_0_._0003|0_._9402_±_0_._0020|
|GPS + Mamba|0_._9821_±_0_._0004|0_._7341_±_0_._0015|0_._8660_±_0_._0007|0_._9311_±_0_._0042|
|GMN|0_._9839_±_0_._0018|0_._7576_±_0_._0042|0_._8714_±_0_._0012|0_._9415_±_0_._0020|



_Table 3._ Benchmark on heterophilic datasets (Platonov et al., 2023). Highlighted are the top **first** , **second** , and **third** results. 

|**Model**|**Roman-empire**<br>Accuracy_↑_|**Amazon-ratings**<br>Accuracy_↑_|**Minesweeper**<br>ROC AUC_↑_|**Tolokers**<br>ROC AUC_↑_|
|---|---|---|---|---|
|GCN|0_._7369_±_0_._0074|0_._4870_±_0_._0063|0_._8975_±_0_._0052|0_._8364_±_0_._0067|
|Gated-GCN|0_._7446_±_0_._0054|0_._4300_±_0_._0032|0_._8754_±_0_._0122|0_._7731_±_0_._0114|
|NAGphormer|0_._7434_±_0_._0077|0_._5126_±_0_._0072|0_._8419_±_0_._0066|0_._7832_±_0_._0095|
|GPS|0_._8200_±_0_._0061|0_._5310_±_0_._0042|0_._9063_±_0_._0067|0_._8371_±_0_._0048|
|Exphormer|0_._8903_±_0_._0037|0_._5351_±_0_._0046|0_._9074_±_0_._0053|0_._8377_±_0_._0078|
|GOAT|0_._7159_±_0_._0125|0_._4461_±_0_._0050|0_._8109_±_0_._0102|0_._8311_±_0_._0104|
|GPS + Mamba|0_._8310_±_0_._0028|0_._4513_±_0_._0097|0_._8993_±_0_._0054|0_._8370_±_0_._0105|
|GMN|0_._8769_±_0_._0050|0_._5407_±_0_._0031|0_._9101_±_0_._0023|0_._8452_±_0_._0021|



without Transformer, MPNN, and PE/SE that we use to evaluate the importance of these elements in achieving good performance, can achieve competitive performance with other complex methods, showing that while Transformers, complex message-passing, and SE/PE are sufficient for good performance in practice, neither is necessary. 

## **5.3. Comparison on GNN Benchmark** 

We further evaluate the performance of GMNs in small and large datasets from the GNN benchmark. The results of GMNs and baseline performance are reported in Table 2. GMN and Exphormer achieve competitive performance each outperforms the other two times. On the other hand again, GMN consistently outperforms GPS + Mamba baseline, showing the importance of designing a new framework for GMNs rather then using existing frameworks of GTs. 

## **5.4. Heterophilic Datasets** 

To evaluate the performance of GMNs on the heterophilc data as well as evaluating their robustness to over-squashing and over-smoothing, we compare their performance with the state-of-the-art baselines and report the results in Table 3. Our GMN outperforms baselines in 3 out of 4 datasets and achieve the second best result in the remaining dataset. These results show that the selection mechanism in GMN can effectively filter irrelevant information and also consider long-range dependencies in heterophilic datasets. 

## **5.5. Ablation Study** 

To evaluate the contribution of each component of GMNs in its performance, we perform ablation study. Table 4 reports the results. The first row, reports the performance of GMNs with its full architecture. Then in each row, we modify one the elements while keeping the other unchanged: Row 2 remove the bidirectional Mamba and use a simple Mamba. Row 3 remove the MPNN and Row 4 use PPR ordering. Finally the last row remove PE. Results show that all the elements of GMN contributes to its performance with most contribution from bidirection Mamba. 

11 

## **Graph Mamba** 

_Table 4._ Ablation study on GMN architecture. 

|**Model**|**Roman-empire**<br>Accuracy_↑_|**Amazon-ratings**<br>Accuracy_↑_|**Minesweeper**<br>ROC AUC_↑_|
|---|---|---|---|
|GMN|0_._8769_±_0_._0050|0_._5407_±_0_._0031|0_._9101_±_0_._0023|
|w/o bidirectional Mamba|0_._8327_±_0_._0062|0_._5016_±_0_._0045|0_._8597_±_0_._0028|
|w/o MPNN|0_._8620_±_0_._0043|0_._5312_±_0_._0044|0_._8983_±_0_._0031|
|PPR ordering|0_._8612_±_0_._0019|0_._5299_±_0_._0037|0_._8991_±_0_._0021|
|w/o PE|0_._8591_±_0_._0054|0_._5308_±_0_._0026|0_._9011_±_0_._0025|



_Figure 2._ Efficiency evaluation and accuracy of GMNs and baselines on OBGN-Arxiv and MalNet-Tiny. Highlighted are the top **first** , **second** , and **third** results. OOM: Out of Memory. 

|**Method**<br>Gated-GCN<br>GPS<br>NAGphormer<br>Exphormer_†_<br>GOAT|Ours<br>GPS+Mamba<br>GMN|
|---|---|
|OGBN-Arxiv||
|Training/Epoch (s)<br>**0.68**<br>OOM<br>5.06<br>1.97<br>13.09<br>Memory (GB)<br>11.09<br>OOM<br>**6.24**<br>36.18<br>8.41<br>Accuracy<br>0.7141<br>OOM<br>0.7013<br>**0.7228**<br>0.7196|**1.18**<br>**1.30**<br>**5.02**<br>**3.85**<br>**0.7239**<br>**0.7248**|
|MalNet-Tiny||
|Training/Epoch (s)<br>**10.3**<br>148.99<br>-<br>57.24<br>-<br>Accuracy<br>0.9223<br>**0.9234**<br>-<br>0.9224<br>-|**36.07**<br>**41.00**<br>**0.9311**<br>**0.9415**|



_†_ We follow the original paper (Shirzad et al., 2023) and use one virtual node in efficiency evaluation. 

**==> picture [145 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
GPS<br>1500 GMN<br>1000<br>500<br>0<br>1 300 600 900 1200<br>Average number of nodes<br>GPU Memory (MB)<br>**----- End of picture text -----**<br>


_Figure 3._ Memory of GPS and GMN on MalNet-Tiny dataset. 

## **5.6. Efficiency** 

As we discussed earlier, one of the main advantages of our model is its efficiency and memory usage. We evaluate this claim on OGBN-Arxiv (Hu et al., 2020) and MalNet-Tiny (Dwivedi et al., 2023) datasets and report the results in Figure 2. Our variants of GMNs are the most efficienct methods while achieving the best performance. To show the trend of scalability, we use MalNet-Tiny and plot the memory usage of GPS and GMN in Figure 3. While GPS, as a graph transformer framework, requires high computational cost (GPU memory usage), GMNs’s memory scales linearly with respect to the input size. 

## **6. Conclusion** 

In this paper, we present Graph Mamba Networks (GMNs) as a new class of graph learning based on State Space Model. We discuss and categorize the new challenges when adapting SSMs to graph-structured data, and present four required and one optional steps to design GMNs, where we choose (1) Neighborhood Tokenization, (2) Token Ordering, (3) Architecture of Bidirectional Selective SSM Encoder, (4) Local Encoding, and dispensable (5) PE and SE. We further provide theoretical justification for the power of GMNs and conduct several experiments to empirically evaluate their performance. 

## **Potential Broader Impact** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. 

12 

**Graph Mamba** 

## **References** 

- Ahamed, M. A. and Cheng, Q. Mambatab: A simple yet effective approach for handling tabular data. _arXiv preprint arXiv:2401.08867_ , 2024. 

- Alon, U. and Yahav, E. On the bottleneck of graph neural networks and its practical implications. In _International Conference on Learning Representations_ , 2021. URL https://openreview.net/forum?id=i80OPhOCVH2. 

- Aoki, M. _State space modeling of time series_ . Springer Science & Business Media, 2013. 

- Arnaiz-Rodr´ıguez, A., Begga, A., Escolano, F., and Oliver, N. M. Diffwire: Inductive graph rewiring via the lovasz bound.´ In _The First Learning on Graphs Conference_ , 2022. URL https://openreview.net/forum?id=IXvfIex0mX6f. 

- Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization, 2016. 

- Baek, J., Kang, M., and Hwang, S. J. Accurate learning of graph representations with graph multiset pooling. In _International Conference on Learning Representations_ , 2021. URL https://openreview.net/forum?id=JHcqXGaqiGn. 

- Bar-Shalom, G., Bevilacqua, B., and Maron, H. Subgraphormer: Subgraph GNNs meet graph transformers. In _NeurIPS 2023 Workshop: New Frontiers in Graph Learning_ , 2023. URL https://openreview.net/forum?id=e8ba9Hu1mM. 

- Behrouz, A., Delavari, P., and Hashemi, F. Unsupervised representation learning of brain activity via bridging voxel activity and functional connectivity. In _NeurIPS 2023 AI for Science Workshop_ , 2023. URL https://openreview.net/forum?id= HSvg7qFFd2. 

- Bevilacqua, B., Frasca, F., Lim, D., Srinivasan, B., Cai, C., Balamurugan, G., Bronstein, M. M., and Maron, H. Equivariant subgraph aggregation networks. In _International Conference on Learning Representations_ , 2022. URL https://openreview. net/forum?id=dFbKQaRk15w. 

- Bresson, X. and Laurent, T. Residual gated graph convnets. _arXiv preprint arXiv:1711.07553_ , 2017. 

- Chen, J., Gao, K., Li, G., and He, K. NAGphormer: A tokenized graph transformer for node classification in large graphs. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id= 8KYeilT3Ow. 

- Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In _International Conference on Learning Representations_ , 2021. URL https://openreview.net/forum?id=Ua6zuk0WRH. 

- Deng, C., Yue, Z., and Zhang, Z. Polynormer: Polynomial-expressive graph transformer in linear time. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=hmv1LpNfXa. 

- Di Giovanni, F., Giusti, L., Barbero, F., Luise, G., Lio, P., and Bronstein, M. M. On over-squashing in message passing neural networks: The impact of width, depth, and topology. In _International Conference on Machine Learning_ , pp. 7865–7885. PMLR, 2023. 

- Ding, Y., Orvieto, A., He, B., and Hofmann, T. Recurrent distance-encoding neural networks for graph representation learning. _arXiv preprint arXiv:2312.01538_ , 2023. 

- Dwivedi, V. P., Rampa´sek, L., Galkin, M., Parviz, A., Wolf, G., Luu, A. T., and Beaini, D.ˇ Long range graph benchmark. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), _Advances in Neural Information Processing Systems_ , volume 35, pp. 22326–22340. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/ paper ~~f~~ iles/paper/2022/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets ~~a~~ nd ~~B~~ enchmarks.pdf. 

- Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio, Y., and Bresson, X. Benchmarking graph neural networks. _Journal of Machine Learning Research_ , 24(43):1–48, 2023. 

- Errica, F., Christiansen, H., Zaverkin, V., Maruyama, T., Niepert, M., and Alesiani, F. Adaptive message passing: A general framework to mitigate oversmoothing, oversquashing, and underreaching. _arXiv preprint arXiv:2312.16560_ , 2023. 

13 

**Graph Mamba** 

- Fan, W., Ma, Y., Li, Q., He, Y., Zhao, E., Tang, J., and Yin, D. Graph neural networks for social recommendation. In _The world wide web conference_ , pp. 417–426, 2019. 

- Finkelshtein, B., Huang, X., Bronstein, M., and Ceylan,[˙] I.[˙] I. Cooperative graph neural networks. _arXiv preprint arXiv:2310.01267_ , 2023. 

- Fu, D. Y., Dao, T., Saab, K. K., Thomas, A. W., Rudra, A., and Re, C. Hungry hungry hippos: Towards language modeling with state space models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=COZDy0WYGg. 

- Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ , 2023. 

- Gu, A., Dao, T., Ermon, S., Rudra, A., and Re, C.´ Hippo: Recurrent memory with optimal polynomial projections. _Advances in neural information processing systems_ , 33:1474–1487, 2020. 

- Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., and Re, C.´ Combining recurrent, convolutional, and continuoustime models with linear state space layers. _Advances in neural information processing systems_ , 34:572–585, 2021. 

- Gu, A., Goel, K., and Re, C. Efficiently modeling long sequences with structured state spaces. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum?id=uYLFoz1vlAC. 

- Gutteridge, B., Dong, X., Bronstein, M. M., and Di Giovanni, F. Drew: Dynamically rewired message passing with delay. In _International Conference on Machine Learning_ , pp. 12252–12267. PMLR, 2023. 

- Hashemi, F., Behrouz, A., and Lakshmanan, L. V. Firmcore decomposition of multilayer networks. In _Proceedings of the ACM Web Conference 2022_ , WWW ’22, pp. 1589–1600, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450390965. doi: 10.1145/3485447.3512205. URL https://doi.org/10.1145/3485447.3512205. 

- He, X., Hooi, B., Laurent, T., Perold, A., LeCun, Y., and Bresson, X. A generalization of vit/mlp-mixer to graphs. In _International Conference on Machine Learning_ , pp. 12724–12745. PMLR, 2023. 

- Henaff, M., Bruna, J., and LeCun, Y. Deep convolutional networks on graph-structured data. _arXiv preprint arXiv:1506.05163_ , 2015. 

- Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., and Leskovec, J. Open graph benchmark: Datasets for machine learning on graphs. _Advances in neural information processing systems_ , 33:22118–22133, 2020. 

- Hussain, M. S., Zaki, M. J., and Subramanian, D. Global self-attention as a replacement for graph convolution. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , pp. 655–665, 2022. 

- Kim, J., Nguyen, D., Min, S., Cho, S., Lee, M., Lee, H., and Hong, S. Pure transformers are powerful graph learners. _Advances in Neural Information Processing Systems_ , 35:14582–14595, 2022. 

- Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. _arXiv preprint arXiv:1609.02907_ , 2016. 

- Kong, K., Chen, J., Kirchenbauer, J., Ni, R., Bruss, C. B., and Goldstein, T. Goat: A global transformer on large-scale graphs. In _International Conference on Machine Learning_ , pp. 17375–17390. PMLR, 2023. 

- Kreuzer, D., Beaini, D., Hamilton, W., Letourneau, V., and Tossou, P.´ Rethinking graph transformers with spectral attention. _Advances in Neural Information Processing Systems_ , 34:21618–21629, 2021. 

- Kuang, W., Zhen, W., Li, Y., Wei, Z., and Ding, B. Coarformer: Transformer for large graph via graph coarsening. 2021. 

- Latora, V. and Marchiori, M. A measure of centrality based on network efficiency. _New Journal of Physics_ , 9(6):188, 2007. 

- Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y. W. Set transformer: A framework for attention-based permutation-invariant neural networks. In _International conference on machine learning_ , pp. 3744–3753. PMLR, 2019. 

- Lick, D. R. and White, A. T. k-degenerate graphs. _Canadian Journal of Mathematics_ , 22(5):1082–1096, 1970. doi: 10.4153/CJM-1970-125-1. 

14 

**Graph Mamba** 

- Lim, D., Robinson, J. D., Zhao, L., Smidt, T., Sra, S., Maron, H., and Jegelka, S. Sign and basis invariant networks for spectral graph representation learning. In _The Eleventh International Conference on Learning Representations_ , 2023a. URL https://openreview.net/forum?id=Q-UHqMorzil. 

- Lim, H., Joo, Y., Ha, E., Song, Y., Yoon, S., Lyoo, I. K., and Shin, T. Brain age prediction using multi-hop graph attention module(MGA) with convolutional neural network. In _Medical Imaging with Deep Learning, short paper track_ , 2023b. URL https://openreview.net/forum?id=brK-VVoDpqo. 

- Liu, C., Zhan, Y., Ma, X., Ding, L., Tao, D., Wu, J., and Hu, W. Gapformer: Graph transformer with graph pooling for node classification. In _Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI-23)_ , pp. 2196–2205, 2023. 

- Liu, J., Yang, H., Zhou, H.-Y., Xi, Y., Yu, L., Yu, Y., Liang, Y., Shi, G., Zhang, S., Zheng, H., et al. Swin-umamba: Mamba-based unet with imagenet-based pretraining. _arXiv preprint arXiv:2402.03302_ , 2024a. 

- Liu, X., Yu, H.-F., Dhillon, I., and Hsieh, C.-J. Learning to encode position for transformer with continuous dynamical model. In _International conference on machine learning_ , pp. 6327–6335. PMLR, 2020. 

- Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., and Liu, Y. Vmamba: Visual state space model. _arXiv preprint arXiv:2401.10166_ , 2024b. 

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 10012–10022, 2021. 

- Ma, J., Li, F., and Wang, B. U-mamba: Enhancing long-range dependency for biomedical image segmentation. _arXiv preprint arXiv:2401.04722_ , 2024. 

- Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., and Grohe, M. Weisfeiler and leman go neural: Higher-order graph neural networks. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 33, pp. 4602–4609, 2019. 

- Morris, C., Rattan, G., and Mutzel, P. Weisfeiler and leman go sparse: Towards scalable higher-order graph embeddings. _Advances in Neural Information Processing Systems_ , 33:21824–21840, 2020. 

- Muller, L., Galkin, M., Morris, C., and Ramp¨ a´sek, L.ˇ Attending to graph transformers. _arXiv preprint arXiv:2302.04181_ , 2023. 

- Murphy, R., Srinivasan, B., Rao, V., and Ribeiro, B. Relational pooling for graph representations. In _International Conference on Machine Learning_ , pp. 4663–4673. PMLR, 2019. 

- Nguyen, D. Q., Nguyen, T. D., and Phung, D. Universal graph transformer self-attention networks. In _Companion Proceedings of the Web Conference 2022_ , pp. 193–196, 2022a. 

- Nguyen, E., Goel, K., Gu, A., Downs, G., Shah, P., Dao, T., Baccus, S., and Re, C.´ S4nd: Modeling images and videos as multidimensional signals with state spaces. _Advances in neural information processing systems_ , 35:2846–2861, 2022b. 

- Nguyen, E., Poli, M., Faizi, M., Thomas, A. W., Wornow, M., Birch-Sykes, C., Massaroli, S., Patel, A., Rabideau, C. M., Bengio, Y., Ermon, S., Re, C., and Baccus, S. HyenaDNA: Long-range genomic sequence modeling at single nucleotide resolution. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https: //openreview.net/forum?id=ubzNoJjOKj. 

- Page, L., Brin, S., Motwani, R., and Winograd, T. The pagerank citation ranking: Bring order to the web. Technical report, Technical report, stanford University, 1998. 

- Park, J., Yun, S., Park, H., Kang, J., Jeong, J., Kim, K.-M., Ha, J.-w., and Kim, H. J. Deformable graph transformer. _arXiv preprint arXiv:2206.14337_ , 2022. 

- Platonov, O., Kuznedelev, D., Diskin, M., Babenko, A., and Prokhorenkova, L. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=tJbbQfw-5wv. 

15 

**Graph Mamba** 

- Rampa´sek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., and Beaini, D.ˇ Recipe for a general, powerful, scalable graph transformer. _Advances in Neural Information Processing Systems_ , 35:14501–14515, 2022. 

- Rong, Y., Bian, Y., Xu, T., Xie, W., Wei, Y., Huang, W., and Huang, J. Self-supervised graph transformer on large-scale molecular data. _Advances in Neural Information Processing Systems_ , 33:12559–12571, 2020. 

- Ruhnau, B. Eigenvector-centrality—a node-centrality? _Social networks_ , 22(4):357–365, 2000. 

- Rusch, T. K., Bronstein, M. M., and Mishra, S. A survey on oversmoothing in graph neural networks. _arXiv preprint arXiv:2303.10993_ , 2023. 

- Sato, R., Yamada, M., and Kashima, H. Random features strengthen graph neural networks. In _Proceedings of the 2021 SIAM international conference on data mining (SDM)_ , pp. 333–341. SIAM, 2021. 

- Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E. H., Scharli, N., and Zhou, D.¨ Large language models can be easily distracted by irrelevant context. In _International Conference on Machine Learning_ , pp. 31210–31227. PMLR, 2023. 

- Shirzad, H., Velingker, A., Venkatachalam, B., Sutherland, D. J., and Sinop, A. K. Exphormer: Sparse transformers for graphs. _arXiv preprint arXiv:2303.06147_ , 2023. 

- Shiv, V. and Quirk, C. Novel positional encodings to enable tree-based transformers. _Advances in neural information processing systems_ , 32, 2019. 

- Song, Y., Huang, S., Cai, J., Wang, X., Zhou, C., and Lin, Z. S4g: Breaking the bottleneck on graphs with structured state spaces, 2024. URL https://openreview.net/forum?id=0Z6lN4GYrO. 

- Tang, S., Dunnmon, J. A., Liangqiong, Q., Saab, K. K., Baykaner, T., Lee-Messer, C., and Rubin, D. L. Modeling multivariate biosignals with graph neural networks and structured state space models. In Mortazavi, B. J., Sarker, T., Beam, A., and Ho, J. C. (eds.), _Proceedings of the Conference on Health, Inference, and Learning_ , volume 209 of _Proceedings of Machine Learning Research_ , pp. 50–71. PMLR, 22 Jun–24 Jun 2023. URL https://proceedings.mlr.press/v209/tang23a.html. 

- Tonshoff, J., Ritzert, M., Rosenbluth, E., and Grohe, M.¨ Where did the gap go? reassessing the long-range graph benchmark. _arXiv preprint arXiv:2309.00367_ , 2023a. 

- Tonshoff, J., Ritzert, M., Wolf, H., and Grohe, M.¨ Walking out of the weisfeiler leman hierarchy: Graph learning beyond message passing. _Transactions on Machine Learning Research_ , 2023b. ISSN 2835-8856. URL https://openreview.net/ forum?id=vgXnEyeWVY. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Velickoviˇ c, P., Cucurull, G., Casanova, A., Romero, A., Li´ o, P., and Bengio, Y.` Graph attention networks. In _International Conference on Learning Representations_ , 2018. URL https://openreview.net/forum?id=rJXMpikCZ. 

- Wang, C., Tsepa, O., Ma, J., and Wang, B. Graph-mamba: Towards long-range graph sequence modeling with selective state spaces. _arXiv preprint arXiv:2402.00789_ , 2024. 

- Wang, H., Yin, H., Zhang, M., and Li, P. Equivariant and stable positional encoding for more powerful graph neural networks. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum?id=e95i1IHcWj. 

- Wang, S. and Xue, B. State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/ forum?id=i0OmcF14Kf. 

- Wang, Y., Min, Y., Shao, E., and Wu, J. Molecular graph contrastive learning with parameterized explainable augmentations. In _2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)_ , pp. 1558–1563. IEEE, 2021. 

- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations_ , pp. 38–45, 2020. 

16 

**Graph Mamba** 

- Wu, Y., Xu, Y., Zhu, W., Song, G., Lin, Z., Wang, L., and Liu, S. Kdlgt: a linear graph transformer framework via kernel decomposition approach. In _Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence_ , pp. 2370–2378, 2023. 

- Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., and Philip, S. Y. A comprehensive survey on graph neural networks. _IEEE transactions on neural networks and learning systems_ , 32(1):4–24, 2020. 

- Xing, Z., Ye, T., Yang, Y., Liu, G., and Zhu, L. Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation. _arXiv preprint arXiv:2401.13560_ , 2024. 

- Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? In _International Conference on Learning Representations_ , 2019. URL https://openreview.net/forum?id=ryGs6iA5Km. 

- Yang, Y., Xing, Z., and Zhu, L. Vivim: a video vision mamba for medical video object segmentation. _arXiv preprint arXiv:2401.14168_ , 2024. 

- Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., and Liu, T.-Y. Do transformers really perform badly for graph representation? _Advances in Neural Information Processing Systems_ , 34:28877–28888, 2021. 

- Yu, Z., Cao, R., Tang, Q., Nie, S., Huang, J., and Wu, S. Order matters: Semantic-aware neural networks for binary code similarity detection. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 1145–1152, 2020. 

- Yun, S., Jeong, M., Kim, R., Kang, J., and Kim, H. J. Graph transformer networks. _Advances in neural information processing systems_ , 32, 2019. 

- Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big bird: Transformers for longer sequences. _Advances in neural information processing systems_ , 33:17283–17297, 2020. 

- Zhang, B., Feng, G., Du, Y., He, D., and Wang, L. A complete expressiveness hierarchy for subgraph GNNs via subgraph weisfeiler-lehman tests. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 41019–41077. PMLR, 23–29 Jul 2023a. URL https://proceedings.mlr.press/v202/zhang23k.html. 

- Zhang, M., Saab, K. K., Poli, M., Dao, T., Goel, K., and Re, C. Effectively modeling time series with simple discrete state spaces. In _The Eleventh International Conference on Learning Representations_ , 2023b. URL https://openreview.net/ forum?id=2EpjkjzdCAa. 

- Zhang, Z., Liu, Q., Hu, Q., and Lee, C.-K. Hierarchical graph transformer with adaptive node sampling. _Advances in Neural Information Processing Systems_ , 35:21171–21183, 2022. 

- Zhao, J., Li, C., Wen, Q., Wang, Y., Liu, Y., Sun, H., Xie, X., and Ye, Y. Gophormer: Ego-graph transformer for node classification. _arXiv preprint arXiv:2110.13094_ , 2021. 

- Zhong, W., He, C., Xiao, C., Liu, Y., Qin, X., and Yu, Z. Long-distance dependency combined multi-hop graph neural networks for protein–protein interactions prediction. _BMC bioinformatics_ , 23(1):1–21, 2022. 

- Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., and Wang, X. Vision mamba: Efficient visual representation learning with bidirectional state space model. _arXiv preprint arXiv:2401.09417_ , 2024. 

17 

**Graph Mamba** 

_Table 5._ Dataset Statistics. 

|_Table 5._ Dataset Statistics.||
|---|---|
|Dataset<br>#Graphs<br>Average #Nodes<br>Average #Edges<br>#Class|Setup<br>Metric<br>Input Level<br>Task|
|Long-range Graph Benchmark (Dwivedi et|al.,2022)|
|COCO-SP<br>123,286<br>476.9<br>2693.7<br>81<br>PascalVOC-SP<br>11,355<br>479.4<br>2710.5<br>21<br>Peptides-Func<br>15,535<br>150.9<br>307.3<br>10<br>Peptides-Struct<br>15,535<br>150.9<br>307.3<br>11 (regression)|Node<br>Classifcation<br>F1 score<br>Node<br>Classifcation<br>F1 score<br>Graph<br>Classifcation<br>Average Precision<br>Graph<br>Regression<br>Mean Absolute Error|
|GNN Benchmark (Dwivedi et al.,2023)||
|MNIST<br>70,000<br>70.6<br>564.5<br>10<br>Graph<br>Classifcation<br>Accuracy<br>CIFAR10<br>60,000<br>117.6<br>941.1<br>10<br>Graph<br>Classifcation<br>Accuracy<br>Pattern<br>14,000<br>118.9<br>3,039.3<br>2<br>Node<br>Classifcation<br>Accuracy<br>MalNet-Tiny<br>5,000<br>1,410.3<br>2,859.9<br>5<br>Graph<br>Classifcation<br>Accuracy||
|Heterophilic Benchmark (Platonov et al.,2023)||
|Roman-empire<br>1<br>22,662<br>32,927<br>18<br>Node<br>Classifcation<br>Accuracy<br>Amazon-ratings<br>1<br>24,492<br>93,050<br>5<br>Node<br>Classifcation<br>Accuracy<br>Minesweeper<br>1<br>10,000<br>39,402<br>2<br>Node<br>Classifcation<br>ROC AUC<br>Tolokers<br>1<br>11,758<br>519,000<br>2<br>Node<br>Classifcation<br>ROC AUC||
|Very Large Dataset (Hu et al.,2020)||
|OGBN-Arxiv<br>1<br>169,343<br>1,166,243<br>40<br>Node<br>Classifcation<br>Accuracy||



## **A. Details of Datasets** 

The statistics of all the datasets are reported in Table 5. For additional details about the datasets, we refer to the Long-range graph benchmark (Dwivedi et al., 2022), GNN Benchmark (Dwivedi et al., 2023), Heterophilic Benchmark (Platonov et al., 2023), and Open Graph Benchmark (Hu et al., 2020). 

## **B. Experimental Setup** 

## **B.1. Hyperparameters** 

We use grid search to tune hyperparameters and the search space is reported in Table 6. Following previous studies, we use the same split of traning/test/validation as (Rampa´sek et al.ˇ , 2022). We report the results over the 4 random seeds. Also, for the baselines’ results (in Tables 1, 2, and 3), we have re-used and reported the benchmark results in the work by Shirzad et al. (2023); Deng et al. (2024); T¨onshoff et al. (2023a) and Wang et al. (2024).[2] . 

## **C. Details of GMN Architecture: Algorithms** 

Algorithm 1 shows the forward pass of the Graph Mamba Network with one layer. For each node, GMN first samples _M_ ˆ walks with length _m_ = 1 _, . . . , m_ and constructs its corresponding tokens, each of which as the induced subgraph of _M_ walks with length _m_ ˆ . We repeat this process _s_ times to have longer sequence and more samples from each hierarchy of the neighborhoods. This part of the algorithm, can be computed before the training process and in CPU. Next, GMNs for each node encode its tokens using an encoder _ϕ_ ( _._ ), which can be an MPNN (e.g., gated-GCN (Bresson & Laurent, 2017)) or RWF encoding (proposed by Tonshoff et al.¨ (2023b)). We then pass the encodings to a Bidirectional Mamba block, which we describe in Algorithm 2 (This algorithm is simple two Mamba block (Gu & Dao, 2023) such that we use one of the backward or forward ordering of inputs for each of them). At the end of line 15, we have the node encodings obtained from subgraph tokenization. Next, we treat each node as a token and pass the encoding to another bidirectional Mamba, with a specific order. We have used degree ordering in our experiments, but there are some other approaches that we have discussed in the main paper. 

> 2 ´ˇ In the previous version of this preprint (Feb 13, 2024), the reported results of Exphormer (Shirzad et al., 2023) and GPS (Rampasek et al., 2022) came from the results of Wang et al. (2024). 

18 

**Graph Mamba** 

_Table 6._ Search space of hyperparameters for each dataset _[†]_ . 

|Dataset|||_M_|_s_|#Layers|Max # Epochs|Learning Rate|
|---|---|---|---|---|---|---|---|
|||||Long-range Graph Benchmark||(Dwivedi et al.,2022)||
|COCO-SP|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_4_,_5_}_|300|0.001|
|PascalVOC-SP|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_4_,_5_}_|300|0.001|
|Peptides-Func|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_4_,_5_}_|300|0.001|
|Peptides-Struct|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_4_,_5_}_|300|0.001|
|||||GNN Benchmark (Dwivedi et al.,2023)||||
|MNIST|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|CIFAR10|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|Pattern|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|MalNet-Tiny|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|||||Heterophilic Benchmark (Platonov et al.,2023)||||
|Roman-empire|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|Amazon-ratings|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|Minesweeper|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|Tolokers|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|
|||||Very Large Dataset (Hu et al.,2020)||||
|OGBN-Arxiv|_{_1_,_|2_,_|4_,_8_,_16_,_32_}_|_{_0_,_1_,_2_,_4_,_8_,_16_}_|_{_3_,_4_,_6_}_|300|0.001|



_†_ This space is not fully searched and preliminary results are reported based on its subspace. We will update the results accordingly. 

## **D. Additional Experimental Results** 

## **D.1. Parameter Sensitivity** 

**The effect of** _M_ **.** Parameter _M_ is the number of walks that we aggregate to construct a subgraph token. To evaluate its effect on the performance of the GMN, we use two datasets of Roman-empire and PascalVOC-SP, from two different benchmarks, and vary the value of _M_ from 1 to 10. The results are reported in Figure 4 (Left). These results show that performance peaks at certain value of _M_ and the exact value varies with the dataset. The main reason is, parameter _M_ determines that how many walks can be a good representative of the neighborhood of a node, and so depends on the density, homophily score, and network topology this value can be different. 

**The effect of** _m_ **.** Similar to the above, we use two datasets of Roman-empire and PascalVOC-SP, from two different benchmarks, and vary the value of _m_ from 1 to 60. The results are reported in Figure 4 (Middle). The performance is non-decreasing with respect to the value of _m_ . That is, increasing the value of _m_ , i.e., considering far neighbors in the tokenization process, does not damage the performance (might lead to better results). Intuitively, using large values of _m_ is expected to damage the performance due to the over-smoothing and over-squashing; however, the selection mechanism in Bidirectional Mamba can select informative tokens (i.e., neighborhood), filtering information that cause performance damage. 

**The effect of** _s_ **.** Using the same setting as the above, we report the results when varying the value of _s_ in Figure 4 (Right). Result show that increasing the value of _s_ can monotonically improve the performance. As discussed earlier, longer sequences of tokens can provide more context for our model and due to the selection mechanism in Mamba (Gu & Dao, 2023), GMNs can select informative subgraphs/nodes and filter irrelevant tokens, resulting in better results with longer sequences. 

## **D.2. Comparison with GRED (Ding et al., 2023) and S4G (Song et al., 2024)** 

GRED (Ding et al., 2023) is a recent work on ArXiv that uses an RNN on the set of neighbors with distance _k_ = 1 _, . . . , K_ to a node of interest for the node classification task. S4G (Song et al., 2024) is a recent unpublished (without preprint but available on Openreview) work that uses the same approach as GRED but replaces the RNN block with a structured SSM. Since the code or models of GRED and S4G are not available, for the comparison of GMNs, S4G, and GRED, we run GMNs on the datasets used in the original papers (Ding et al., 2023; Song et al., 2024). The results are reported in Table 7. GMNs consistently outperforms GRED (Ding et al., 2023) in all datasets and outperforms S4G in 4 out of 5 datasets. The 

19 

**Graph Mamba** 

**Algorithm 1** Graph Mamba Networks (with one layer) 

**Input:** A graph _G_ = ( _V, E_ ), input features **X** _∈_ R _[n][×][d]_ , ordered array of nodes _V_ = _{v_ 1 _, . . . , vn}_ , and hyperparameters _M, m_ , and _s_ . **Optional:** Matrix **P** , whose rows are positional/structural encodings correspond to nodes, and/or a MPNN model Ψ( _._ ). **Output:** The updated node encodings **X** new. 

1: **for** _v ∈ V_ **do** _▷_ **This block can be done before the training.** 2: **for** _m_ ˆ = 0 _, . . . , m_ **do** 3: **for** ˆ _s_ = 1 _, . . . , s_ **do** 4: _Tm[s]_ ˆ[ˆ][(] _[v]_[)] _[ ←∅]_[;] 5: **for** _M_[ˆ] = 1 _, . . . , M_ **do** 6: W _←_ Sample a random walk with length _m_ ˆ starting from _v_ ; 7: _Tm[s]_ ˆ[ˆ][(] _[v]_[)] _[ ←][T] m_[ ˆ] _[s]_ ˆ[(] _[v]_[)] _[ ∪{][u][|][u][ ∈]_[W] _[}]_[;] 8: 9: _▷_ **Start the training:** 10: Initialize all learnable parameters; 11: **for** _v ∈ V_ **do** 12: **for** _j_ = 1 _, . . . , s_ **do** 13: **for** _i_ = 1 _, . . . , m_ **do** 14: **x**[(] _v[i][−]_[1)] _[s]_[+] _[j] ← ϕ_ � _G_ [ _Ti[j]_[(] _[v]_[)]] _[,]_ **[ X]** _Ti[j]_[(] _[v]_[)] _[ ||]_ **[ P]** _[T][ j] i_[(] _[v]_[)] �; **Φ** _v ←∥i[sm]_ =1 **[x]** _v[i]_[;] _▷_ **Φ** _v_ **is a matrix whose rows are x** _[i] v_ **[.]** 15: _**y**_ output( _v_ ) _←_ BiMamba ( **Φ** _v_ ); _▷_ **Using Algorithm 2.** 16: _▷_ **Each node is a token:** 17: _**Y** ←∥i[sm]_ =1 _**[y]**_ output[(] _[v]_[)][;] _▷_ _**y**_ **is a matrix whose rows are** _**y**_ **output** ( _v_ ) **.** 18: _**Y**_ output _←_ BiMamba ( ( _**Y**_ ) + Ψ ( Ψ ( _G,_ **X** _∥_ **P** );; Roman-empire PascalVOC-SP 0.8 0.8 0.8 Roman-empire Roman-empire PascalVOC-SP PascalVOC-SP 0.6 0.6 0.6 0.4 0.4 0.4 1 2 4 6 8 10 1 5 10 20 30 40 50 60 5 10 15 M m s 

18: _**Y**_ output _←_ BiMamba ( ( _**Y**_ ) + Ψ ( Ψ ( _G,_ **X** _∥_ **P** );; 

_Figure 4._ The effect of (Left) _M_ , (Middle) _m_ , and (Right) _s_ on the performance of GMNs. 

reason is two folds: (1) GMNs use sampled walks instead of all the nodes within _k_ -distance neighborhood. As discussed in Theorem 4.1, this approach with large enough length and samples is more expressive than considering all nodes within the neighborhood. (2) S4G and GRED use simple RNN and SSM to aggregate the information about all the different neighborhoods of a node, while GMNs use Mamba, which have a selection mechanism. This selection mechanism help the model to choose neighborhoods that are more informative and important than others. (3) GRED and S4G are solely based on distance encoding, meaning that they miss the connections between nodes in _k_ -distance and ( _k_ + 1)-distance. Figure 5 shows a failure example of these methods that solely are based on distance of nodes. To obtain the node encoding of node _A_ , these two methods group nodes wit respect to their distance to _A_ , either _d_ = 1 _,_ 2 _,_ and 3. In Figure 5, while these two graphs are non-isomorphism, the output of this step for both graphs are the same, meaning that these methods obtain the same node encoding for _A_ . 

## **E. Complexity Analysis of GNMs** 

_m ≥_ 1 **.** For each node _v ∈ V_ , we generate _M × s_ walks with length _m_ ˆ = 1 _, . . . , m_ , which requires _O_ ( _M × s ×_ ( _m_ + 1)) time. Given _K_ tokens, the complexity of bidirectional Mamba is 2 _×_ of Mamba (Gu & Dao, 2023), which is linear with respect to _K_ . Accordingly, since we have _O_ ( _M × s × m_ ) tokens, the final complexity for a given node _v ∈ V_ is _O_ ( _M × s ×_ ( _m_ + 1)). Repeating the process for all nodes, the time complexity is _O_ ( _M × s ×_ ( _m_ + 1) _× |V |_ + _|E|_ ), which is linear in terms of _|V |_ and _|E|_ (graph size). To compare to the quadratic time complexity of GTs, even for small 

20 

**Graph Mamba** 

## **Algorithm 2** Bidirectional Mamba 

**Input:** A sequence **Φ** (Ordered matrix, where each row is a token). **Output:** The updated sequence encodings **Φ** . 1: _▷_ **Forward Scan:** 

2: **Φ** f = _σ_ (Conv ( **W** input _,f_ LayerNorm ( **Φ** ))); 3: **B** f = **WB** f **Φ** f; 4: **C** f = **WC** f **Φ** f; 5: **∆** f = Softplus ( **W** ∆f **Φ** f); 6: **A**[¯] = Discrete **A** ( **A** _,_ **∆** ); 7: **B**[¯] f = Discrete **B** f ( **B** f _,_ **∆** ); 8: _**y**_ f = SSM ¯ **A** _,_ **B**[¯] f _,_ **C** f[(] **[Φ]**[f][)][;] 9: _**Y**_ f = **W** f _,_ 1 ( _**y**_ f _⊙ σ_ ( **W** f _,_ 2 LayerNorm ( **Φ** ))); 10: _▷_ **Backward Scan:** 11: **Φ** _←_ Reverse-rows ( **Φ** ); 12: **Φ** b = _σ_ (Conv ( **W** input, b LayerNorm ( **Φ** ))); 13: **B** b = **WB** b **Φ** b; 14: **C** b = **WC** b **Φ** b; 15: **∆** b = Softplus ( **W** ∆b **Φ** b); 16: **A**[¯] = Discrete **A** ( **A** _,_ **∆** ); 17: **B**[¯] b = Discrete **B** b ( **B** b _,_ **∆** ); 18: _**y**_ b = SSM **A** ¯ _,_ **B**[¯] b _,_ **C** b[(] **[Φ]**[b][)][;] 19: _**Y**_ b = **W** b _,_ 1 ( _**y**_ b _⊙ σ_ ( **W** b _,_ 2 LayerNorm ( **Φ** ))); 

_▷_ **Reverse the order of rows in the matrix.** 

20: _▷_ **Output:** 21: _**y**_ output _←_ **W** out ( _**Y**_ f + Reverse-row( _**Y**_ b)); 

22: **return** _**y**_ output; 

**==> picture [487 x 102] intentionally omitted <==**

_Figure 5._ Failure example for methods that are solely based on distance encoding. Solely considering the set of nodes in different distances to the target node misses the connections between them. While the structure of these two graphs are different, the set of nodes with the same distance to node _A_ are the same. Accordingly, GRED (Ding et al., 2023) and S4G (Song et al., 2024) achieve the same node encoding for _A_ , missing _A_ ’s neighborhood topology. 

networks, note that in practice, _M × s ×_ ( _m_ + 1) _≪|V |_ , and in our experiments usually _M × s ×_ ( _m_ + 1) _≤_ 200. Also, note that using MPNN as an optional step cannot affect the time complexity as the MPNN requires _O_ ( _|V |_ + _|E|_ ) time. 

_m_ = 0 **.** In this case, each node is a token and so the GMN requires _O_ ( _|V |_ ) time. Using MPNN in the architecture, the time complexity would be _O_ ( _|V |_ + _|E|_ ), dominating by the MPNN time complexity. 

As discussed above, based on the properties of Mamba architecture, longer sequence of tokens (larger value of _s ≥_ 1) can improve the performance of the method. Based on the abovementioned time complexity when _m ≥_ 1, we can see that there is a trade-off between time complexity and the performance of the model. That is, while larger _s_ result in better performance, it results in slower model. 

## **F. Discussion on a Concurrent Work** 

(Wang et al., 2024), in work concurrent to and independent of ours, replace Transformer architecture (attention block) with a Mamba block (Gu & Dao, 2023) in GPS framework (Rampa´sek et al.ˇ , 2022). Next, we discuss these two works in different aspects: 

21 

**Graph Mamba** 

_Table 7._ Comparison with GRED and S4G Models. Highlighted are the top **first** and **second** results. 

|**Model**|**MNIST**<br>Accuracy_↑_|**CIFAR10**<br>Accuracy_↑_|**PATTERN**<br>Accuracy_↑_|**Peptides-func**<br>AP_↑_|**Peptides-struct**<br>MAE_↓_|
|---|---|---|---|---|---|
|S4G (Song et al.,2024)_†_|0_._9637_±_0_._0017|0_._7065_±_0_._0033|0_._8687_±_0_._0002|0_._7293_±_0_._0004|0_._2485_±_0_._0017|
|GRED (Ding et al.,2023)_‡_|0_._9822_±_0_._0095|0_._7537_±_0_._6210|0_._8676_±_0_._0200|0_._7041_±_0_._0049|0_._2503_±_0_._0019|
|GMN(Ours)|0_._9839_±_0_._0018|0_._7576_±_0_._0042|0_._8714_±_0_._0012|0_._7071_±_0_._0083|0_._2473_±_0_._0025|
|_†_Results are reported bySong et al.(2024).||||||
|_‡_Results are reported byDing et al.(2023).||||||



**==> picture [438 x 177] intentionally omitted <==**

_Figure 6._ (A) An example of node tokenization and its information flow. Even using PE/SE, nodes at the beginning of the sequence do not have any information about the structure of the graph! (B) Potential failure example for using a simple one directional sequential encoder when each token is a node. Nodes in the right hand side do not have any information about the structure of the graph in the left hand side (Or vise versa depends on the direction of the order). 

**Architecture Design.** As mentioned above, the GMB model (Wang et al., 2024) replaces the attention module in GPS framework (Rampa´sek et al.ˇ , 2022) with Mamba block (Gu & Dao, 2023). Accordingly, it treats each node as a token, uses PE/SE as initial feature vectors, and ordered nodes based on their degree. Since this approach is based on node tokenization and uses one directional Mamba (Gu & Dao, 2023), it suffers from the limitations of GTs with node tokenization mentioned in § 3. More specifically: (1) Although it has more ability to learn long-range dependencies, this approach lacks inductive bias about the graph structure and requires complex PE/SE. (2) Using a simple one directional Mamba causes the lack of inductive bias about some structures in the graph. Figure 6 provides an example of this information loss. In part (A), we show an example of node tokenization and node ordering with respect to their degrees. Based on the information flow, using a one directional Mamba, nodes with high-degree do not have any information about the structure of the graph. For example, in Figure 6 (B), even with using complex PE/SE, nodes in the right hand side do not have any information about the global information in the left hand side, due to the one directional information flow of Mamba block. As discussed earlier in the paper, this is one of the new challenges of using SSMs (compare to GTs). The main reason is, attentions in Transformers consider all nodes connected and so the information could pass between each pair of nodes. In sequential encoders (even with selection mechanism), however, each token has the information about its previous tokens. Our neighborhood sampling and its reverse ordering can address this issue due to its implicit order of neighborhood hierarchy. 

Comparing to GMNs, GMB can be seen as a special case of GMNs when we use _m_ = 0 and replace bidirectional Mamba block with a one directional Mamba block. Our approach using parameter _m_ provides the flexibility of using either node or subgraph tokenization, whenever either inductive bias or long-range dependencies is more important to the task and the dataset. Furthermore, having these two special traits of GMNs compared to GMB results in provable expressive power of GMNs, which we discuss in the following. 

**Expressive Power.** As discussed earlier, due to the lack of inductive bias, GMB method requires complex PE/SE to learn about the structure of the graph. While GMNs _without_ PE/SE and MPNNs has unbounded expressive power with respect to 

22 

**Graph Mamba** 

isomorphic test (Theorem 4.4), GMB cannot distinguish graphs with the same sequence of degrees and its expressive power is bounded by the expressive power of its MPNN. That is, let _G_ 1 and _G_ 2 be two graphs with the same sequence of node degrees, Mamba block in GMB, in the worst case of having the same node feature vectors, cannot distinguish _G_ 1 and _G_ 2 since its input is the same for both of these graphs. Accordingly, the expressive power of GMB is bounded by the expressive power of its MPNN in the GPS framework (Ramp´aˇsek et al., 2022). 

## **G. Theoretical Analysis of GMNs** 

**Theorem G.1.** _With large enough M, m, and s >_ 0 _, GMNs’ neighborhood sampling is strictly more expressive than k-hop neighborhood sampling._ 

_Proof._ We first show that in this condition, the random walk neighborhood sampling is as expressive as _k_ -hop neighborhood sampling. To this end, given an arbitrary small _ϵ >_ 0, we show that the probability that _k_ -hop neighborhood sampling is more expressive than random walk neighborhood sampling is less than _ϵ_ . Let _m_ = _k_ , _s_ = 1, and _pu,v_ be the probability that we sample node _v_ in a walk with length _m_ = _k_ starting from node _u_ . This prbobality is zero if the shortest path of _u_ and _v_ ˆ is more than _k_ . To construct the subgraph token corresponds to _m_ = _k_ , we use _M_ samples and so the probability of not seeing node _v_ in these samples is _qu,v,M_ = (1 _− pu,v_ ) _[M] ≤_ 1. Now let _M →∞_ and _v ∈Nk_ ( _u_ ) (i.e., _pu,v_ = 0), we have lim _M →∞ qu,v,M_ = 0. Accordingly, with large enough _M_ , we have _qu,v,M ≤ ϵ_ . This means that with a large enough _M_ when _m_ = _k_ and _s_ = 1, we sample all the nodes within the _k_ -hop neighborhood, meaning that random walk neighborhood sampling at least provide as much information as _k_ -hop neighborhood sampling with arbitrary large probability. Next, we provide an example that _k_ -hop neighborhood sampling is not able to distinguish two non-isomorphism graphs, while random walk sampling can. Let _S_ = _{v_ 1 _, v_ 2 _, . . . , vℓ}_ be a set of nodes such that all nodes have shortest path less than _k_ to _u_ . Using hyperparamters _m_ = _k_ and arbitrary _M_ , let the probability that we get _G_ [ _S_ ] as the subgraph token be 1 _> qS >_ 0. Using _s_ samples, the probability that we do not have _G_ [ _S_ ] as one of the subgraph tokens is (1 _− qS_ ) _[s]_ . Now using large _s →∞_ , we have lim _s→∞_ (1 _− qS_ ) _[s]_ = 0 and so for any arbitrary _ϵ >_ 0 there is a large _s >_ 0 such that we see all non-empty subgraphs of the _k_ -hop neighborhood with probability more than 1 _− ϵ_ , which is more powerful than the neighborhood itself. 

Note that the above proof does not necessarily guarantee an efficient sampling, but it guarantees the expressive power. 

**Theorem G.2** (Universality) **.** _Let_ 1 _≤ p < ∞, and ϵ >_ 0 _. For any continues function f_ : [0 _,_ 1] _[d][×][n] →_ R _[d][×][n] that is permutation equivariant, there exists a GMN with positional encoding, gp, such that ℓ[p]_ ( _f, g_ ) _< ϵ._ 

_Proof._ Recently, Wang & Xue (2023) show that SSMs with layer-wise nonlinearity are universal approximators of any sequence-to-sequence function. We let _m_ = 0, meaning we use node tokenization. Using the universality of SSMs for sequence-to-sequence function, the rest of the proof is the same as Kreuzer et al. (2021), where they use the padded adjacency matrix of _G_ as a positional encoding to prove the same theorem for Transformers. In fact, the universality for sequence-to-sequence functions is enough to show the universality on graphs with a strong positional encoding. 

**Theorem G.3** (Expressive Power w/ PE) **.** _Given the full set of eigenfunctions and enough parameters, GMNs can distinguish any pair of non-isomorphic graphs and are more powerful than any WL test._ 

_Proof._ Due to the universality of GMNs in Theorem G.3, one can use a GMN with the padded adjacency matrix of _G_ as positional encoding and learn a function that is invariant under node index permutations and maps non-isomorphic graphs to different values. 

**Theorem G.4** (Expressive Power w/o PE and MPNN) **.** _With enough parameter, for every k ≥_ 1 _there are graphs that are distinguishable by GMNs, but not by k-WL test, showing that their expressivity power is not bounded by any WL test._ 

_Proof._ The proof of this theorem directly comes from the recent work of Tonshoff et al.¨ (2023b), where they prove a similar theorem for CRaWl (Tonshoff et al.¨ , 2023b). That is, using RWF as _ϕ_ ( _._ ) in GMNs (without MPNN), makes the GMNs as powerful as CRaWl. The reason is CRaWl uses 1-d CNN on top of RWF, while GMNs use Bidirectional Mamba block on top of RWF: using a broadcast SMM, this block becomes similar to 1-d CNN. 

23 

