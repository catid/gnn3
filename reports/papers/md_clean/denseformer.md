# **DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging**

**Matteo Pagliardini**[* 1] **Amirkeivan Mohtashami**[* 1] **Francois Fleuret**[2] **Martin Jaggi**[1]

## **Abstract**

The transformer architecture by Vaswani et al. (2017) is now ubiquitous across application domains, from natural language processing to speech processing and image understanding. We propose DenseFormer, a simple modification to the standard architecture that improves the perplexity of the model without increasing its size—adding a few thousand parameters for large-scale models in the 100B parameters range. Our approach relies on an additional averaging step after each transformer block, which computes a weighted average of current and past representations—we refer to this operation as Depth-Weighted-Average (DWA). The learned DWA weights exhibit coherent patterns of information flow, revealing the strong and structured reuse of activations from distant layers. Experiments demonstrate that DenseFormer is more data efficient, reaching the same perplexity of much deeper transformer models, and that for the same perplexity, these new models outperform transformer baselines in terms of memory efficiency and inference time.

## **1 Introduction**

The transformer architecture (Vaswani et al., 2017) is the workhorse of modern natural language processing. Recent leaps in the state of the art can be attributed in a large part to efforts scaling this architecture, from millions of parameters (Devlin et al., 2019) to large billion-parameter models (Radford et al., 2019; Brown et al., 2020; Touvron et al., 2023a;b; OpenAI, 2023). Unfortunately, those larger models come with an increased computational cost, and a large memory footprint. This renders them impractical to use in a wide range of use-cases, limiting who can benefit

> *Equal contribution (order is arbitrary) 1EPFL 2University of Geneva. Correspondence to: Matteo Pagliardini _<_ matteo.pagliardini@epfl.ch _>_ , Amirkeivan Mohtashami

> _<_ amirkeivan.mohtashami@epfl.ch _>_ .

from them to a handful of big corporations. As an attempt to mitigate this issue, Touvron et al. (2023a) propose training a smaller model for more steps. However, longer training requires larger datasets which becomes challenging as we are reaching scales where even extremely large datasets fall short of sufficient amounts of data (Villalobos et al., 2022).

Furthermore, recent observations suggest that we are reaching the state of diminishing returns where increasing the depth of the model beyond a certain point does not significantly improve performance (Petty et al., 2023a). Interestingly, a similar state of diminishing returns has been observed in the field of computer vision focused on the training of deep convolutional neural networks. Various solutions were proposed to address this issue, including DenseNets (Huang et al., 2017) which alleviated the problem by allowing subsequent layers to directly access outputs of earlier layers.

In this work, using a similar intuition as DenseNets, we propose the DenseFormer architecture. In particular, instead of only having skip connections from one block to the next, in DenseFormer, a weighted average of the outputs of all previous blocks is given as the input of the next block. The approach is visually summarized in Fig. 1a.

We show that DenseFormers can perform the same as a much deeper standard Transformer model while at the same time being smaller in size, faster, and consuming less memory at inference. More importantly, this is achieved without increasing the amount of required data. As such, DenseFormers are also more data efficient, obtaining much better performance when trained on the same amount of data than a standard model with a similar number of parameters. Our results establish the DenseFormer architecture as an improved version of Transformers for language modeling, encouraging their future use.

In addition to providing experimental results on DenseFormer’s performance, we also provide additional insights and intuition for their success. Looking at the learned weights of the DWA modules we observe a surprisingly stable pattern in the learned weights that emerges at multiple depths and generalizes across random seeds (see Fig. 5). Overall, similar to Huang et al. (2017), we hypothesize that the inter-block connectivity enables the model to more

**==> picture [442 x 99] intentionally omitted <==**

(a) DenseFormer Architecture. (b) DWA Weights with Dilation.<br>DWA DWA<br>=<br>=<br>+<br>+<br>Transformer Block Transformer Block +<br>next token prediction<br>**----- End of picture text -----**<br>

_Figure 1._ **DenseFormer architecture.** The diagram in **(a)** shows the DenseFormer architecture with two transformer layers and a dilation of 1. After the first (resp. second) block, the past and current intermediary representations _{X_ 0 _, X_ 1 _}_ (resp. _{X_ 0 _, X_ 1 _, X_ 2 _}_ ) are averaged using the first (resp. second) DWA weights [ _α_ 0 _,_ 0 _, α_ 0 _,_ 1] (resp. [ _α_ 1 _,_ 0 _, α_ 1 _,_ 1 _, α_ 1 _,_ 2]). The DWA weights are supported by red arrows. Those weights are represented in matrix form in **(b)** , for a 12 layers DenseFormer. A DWA module at depth _i_ has _i_ + 1 weights, represented in red. Increasing the dilation sparsifies this matrix, reducing the computational overhead without degrading the perplexity, see Section 3.2 for more details.

directly re-use early features, without requiring to allocate as much bandwidth to propagate them through multiple layers. Intuitively, this seems to help resolve an ambiguity caused by skip-connections which force the deeper representations to maintain the current token representation while at the same time having to predict the next token.

**Contributions.** Our contributions can be summarized as follows:

- Introducing DenseFormer architecture by adding a depthweighted-average module after each Transformer block.

- Demonstrating over different settings (e.g. datasets, batch sizes, sequence lengths) the significantly superior performance of DenseFormer over deeper Transformers, yielding a better speed-performance trade-off during both inference and training.

- Providing additional empirically grounded insights and intuition that support the benefits of using DenseFormer.

Our implementation of DenseFormer is available at https: //github.com/epfml/DenseFormer.

## **2 Related Work**

While larger models have shown great promise in delivering better capabilities, recent results suggest that the gain from using deeper models faces diminishing returns (Petty et al., 2023b). Interestingly, the same challenge presented itself earlier for scaling convolutional neural networks (He & Sun, 2015). Methods that allow a better flow of information from earlier to later layers such as Residual connections (He et al., 2016) and Highway Networks (Srivastava et al., 2015) have been proposed as a successful solution to tackle this challenge, stabilizing training and increasing the threshold where a gain can be observed from increasing depth. Taking it to the extreme, DenseNets (Huang et al., 2017) demonstrate the benefit of having access to the output of all

previous layers.We propose DenseFormer by building on a similar intuition, allowing each block to directly access the output of all previous blocks.

Some advantages of attending to representations from earlier layers have already been explored in prior work. Depthwise Attention (ElNokrashy et al., 2022) suggests adding an attention-like layer before Transformer’s final projection layer. This new layer applies attention across the outputs of the Transformer’s blocks for the current token (instead of over different tokens as in the standard attention layer). This operation is similar to the weighted averaging block introduced in this work which mixes the outputs of earlier blocks. However, whereas in our proposal the weights are learned during training, Depth-wise Attention computes the weights using the dot product (similar to the attention) which levies a higher overhead. In our experiments, we also show that only a single DWA step before the last layer does not yield the same performance as a full DenseFormer. In another recent relevant work Mohtashami et al. (2023) suggest interleaving current and past representations. This allows current tokens to attend to previous representations of themselves and past tokens. Our DenseFormer can be seen as a crude and much more efficient approximation of this mechanism in which we restrict each token to only attend to past representations of themselves, using static (as opposed to dynamic) attention weights.

Since its original design by Vaswani et al. (2017), the Transformer architecture used in most applications changed surprisingly little. LLM training is costly, and architecture choices are often conservative. Small variations include changing the activation function (Shazeer, 2020), adopting RMSNorm instead of LayerNorm, or computing the feedforward and attention in parallel (Wang & Komatsuzaki, 2021; Touvron et al., 2023a; Penedo et al., 2023). More progressive proposals have been made to alleviate computational challenges of the self attention module, such as using kernel methods or other linear approximations (Wang

et al., 2020; Katharopoulos et al., 2020; Kitaev et al., 2020), or removing redundant operations without impacting performance (He & Hofmann, 2023; Shazeer, 2019). These proposals only affect the internal structure of Transformer blocks. As DenseFormer only adds DWA modules that operate between blocks, we expect that it can be readily used together with these existing proposals.

Additionally, recent works focusing extensively on hardware-aware efficient implementations (Dao et al., 2022; Dao, 2023) provide significant gains in both memory and computational time. Similar to those works, we keep the hardware in mind when deriving an efficient DenseFormer implementation reducing the time overhead during both inference and training to a negligible amount.

Recent explorations also have shown gains from using multiple language models instead of one. An example is using a mixture of experts, which rely on a routing mechanism (Fedus et al., 2022) to select which expert(s) to use in a given context. Other examples include deploying the same instance of a model in different roles allowing them to debate or provide feedback to each other leading to performance improvements (Liang et al., 2023; Madaan et al., 2023). As these approaches mostly retain the structure of the Transformer architecture and focus on the communication structure between multiple models (or sub-modules), we also expect them to be adaptable to use DenseFormers.

## **3 Method**

**Setup & Notations.** We consider the standard Transformer architecture. Given a depth _d_ , it consists in a succession of _d_ Transformer blocks _B_ 1 _, . . . , Bd_ , each composed of a self-attention module followed by a single hidden layer Multi-Layer-Perceptron (MLP). We name _X_ 0 _, . . . , Xd_ the different intermediary representations, with _X_ 0 being the embedded token sequence, and _Xi_ for _i ≥_ 1 being the output of block _Bi_ .

We summarize the Transformer architecture as follows:

**==> picture [149 x 41] intentionally omitted <==**

**DenseFormer.** The only change to the original architecture is the addition of a **Depth Weighted Average module (DWA)** after each transformer block. A DWA module at depth _i_ performs a weighted average between (i) the output from the current block _Bi_ , (ii) the output of all previous blocks _Bj<i_ , and (iii) the embedded input _X_ 0. The weights of the weighted-average for the DWA _i_ module at depth _i_ are _αi,_ 0 _, . . . , αi,i_ . A visual summary can be seen in Fig 1a. The elements of the _α_ matrix are the only additional parameters of our method. More formally, our DenseFormer model can

be summarized as follows:

**==> picture [175 x 92] intentionally omitted <==**

**==> picture [101 x 11] intentionally omitted <==**

In Section 4 we demonstrate that the DenseFormer architecture can outperform the standard Transformer architecture. In particular, it obtains a much stronger performance (in terms of perplexity) than a model of the same depth, matching the performance of a much deeper model which is both slower at inference and larger in size, leading to a much higher memory footprint than DenseFormer. We further demonstrate the importance of the improved inter-block connectivity brought by the DWA modules in Section 5. We do so by comparing our architecture to a variety of baselines with constrained connections and show those do not perform as well as DenseFormers.

**Initializing the DWA modules.** We note that if _αi,i_ is set to 1 while others are set to 0, the DWA module acts as an identity function, reducing DenseFormer to the standard Transformer architecture. Therefore, we start our training from this initialization.

## **3.1 Impact on Resources**

**Negligible Model Size Overhead.** At depth _i_ the DWA module has _i_ + 1 weights. Therefore, for a DenseFormer of depth _d_ , the total number of additional parameters is � _dj_ =1[(] _[j]_[+ 1)][=] _d_ ( _d_ 2+3) . For typical model depths (less than 100 blocks), this represents at most an order of 10[3] parameters, which is negligible when compared to the full size of the models.

**Negligible Memory Overhead.** We also emphasize that while DWA requires access to the output of blocks and embedded input _X_ 0 _, . . . , Xd_ , these values are stored even when using the standard Transformer architecture. During training, the outputs of the blocks are kept in memory to allow backpropagation, while at inference, the outputs are stored in the KV cache to facilitate auto-regressive decoding. Therefore, the total memory overhead of DenseFormer is negligible.

**Computational Overhead.** Computing the output of the DWA modules increases the computational cost since it requires averaging over multiple large tensors of size (batch size _×_ sequence length _×_ hidden dimension). In this work, we provide an efficient implementation of DWA to

**==> picture [227 x 58] intentionally omitted <==**

_Figure 2._ **DWA weights with dilation** _**and**_ **DWA period.** For a 12 layers DenseFormer, the _α_ weights are sparsified using dilation _k_ and DWA periodicity _p_ (referred to as _k_ x _p_ ). Compared to Fig. 1b, only certain rows have some weights other than the upper diagonal weights (which correspond to the regular transformer information flow). Increasing the dilation and period sparsifies the _α_ matrix, reducing the computational overhead without degrading the perplexity, see Sections 3.2 and 3.3 for more details.

reduce the overhead and avoid unnecessary data movement. In addition, we introduce two architectural hyperparameters, which allow building a set of DenseFormer variants that approximate the full DenseFormer. These hyperparameters are DWA dilation and DWA periodicity, respectively introduced in Sections 3.2 and 3.3. We refer to a DenseFormer variant with the dilation factor _k_ and the DWA periodicity _p_ as _k_ x _p_ -DenseFormer. In this notation, the full DenseFormer is a 1 _x_ 1-DenseFormer.

## **3.2 Dilated DenseFormer**

In order to further reduce the computational overhead, we introduce a dilation parameter which sparsifies the DWA weights by periodically setting them to 0. In particular, each DWA module is now given the output of every _k_ -th block, where _k_ is called the dilation factor. More formally, given a DWA module at depth _i_ , a dilation factor of _k_ implies DWA _i_ is only computing a weighted average over _{Xj|j ≤ i, j ≡ i_ (mod _k_ ) _}_ ). See Fig. 1b for a visual explanation. Our dilated DenseFormer can be described as:

**==> picture [243 x 72] intentionally omitted <==**

As shown in Section 4, we observe no noticeable performance degradation for small values of _k_ (e.g. 2 or 4) while the computational overhead is significantly reduced, leading to much faster inference and training. More precisely, a dilation of _k_ reduces the computational overhead induced by DWA modules by a factor of 1 _/k_ .

## **3.3 Periodic DenseFormer**

An alternative method to dilation for sparsifying DWA weights is adding DWA modules to the architecture less

**==> picture [179 x 178] intentionally omitted <==**

4x1-DenseFormer<br>18 . 5 48<br>4x5-DenseFormer<br>Transformer<br>18 . 0 72 48<br>48<br>90<br>17 . 5<br>104 72<br>72<br>17 . 0 90<br>90<br>104<br>104<br>16 . 5<br>3 4 5 6<br>Batches Per Second<br>Perplexity<br>**----- End of picture text -----**<br>

_Figure 3._ **Speed and performance trade-off.** Comparison of speed and performance trade-off between the standard Transformer architecture and DenseFormer. The number of blocks in each architecture is reported next to the data-point. All DenseFormer models on this plot use a dilation factor of 4. We show results using a DWA period of 1 and 5. **Comparing perplexities:** Considering only the perplexity (y-axis), a 48 block DenseFormer performs similarly as a much deeper 72 block Transformer. **Comparing trade-offs:** A 48 block 4x5-DenseFormer matches the better perplexity of a 72 block Transformer while being 1 _._ 4 _×_ faster at inference.

frequently. In particular, we can consider only adding the DWA after every _p_ blocks (instead of after every block as in the standard DenseFormer). We refer to _p_ as the DWA period. A standard DenseFormer has period 1. A DenseFormer with dilation _k_ and DWA period _p_ —referred to as _k_ x _p_ -DenseFormer—can be described as:

**==> picture [235 x 89] intentionally omitted <==**

**==> picture [97 x 11] intentionally omitted <==**

A visual representation of the matrix of _α_ weights can be seen in Fig. 2. By increasing the periodicity _p_ , we further reduce the computational cost of DenseFormer. In Section 4 we evaluate the effect of increasing the period for a 4-dilated DenseFormer on both performance and speed. We observe that using small values larger than 1 for the period can provide a noticeable boost in speed without noticeable performance degradation. Using a period of _p_ reduces the computational overhead by a factor of 1 _/p_ . Hence, a _k_ x _p_ - DenseFormer only has 1 _/kp_ of the computational overhead of a regular DenseFormer. In Section 5 we also provide results with other sparsity patterns for the DWA weights

( _α_ ) but show that using dilation and periodicity works most favorably.

**Interplay between** _k_ **and** _p_ **.** The ideal value of _p_ can depend on the dilation _k_ used. For instance, using _k_ = _p_ = 4 implies the DWA module after block 4 will look at _{X_ 0 _, X_ 4 _}_ . Its output, _Y_ 4, will be sent through block 5 to yield _X_ 5. However, the next DWA module after block 8 will only look at _{X_ 0 _, X_ 4 _, X_ 8 _}_ (and not at _X_ 5). This means that _Y_ 4 will have to go through blocks 5 _,_ 6 _,_ 7 _,_ 8 before being accessible by later DWA modules. In contrast, using _k_ = 4 and _p_ = 5 allows the information to propagate much faster since DWA modules always have access to the processed output of the previous DWA module. This interplay can be visualized in Fig. 2 as well as in Appendix B.2.

## **4 Results**

We demonstrate the effectiveness of DenseFormer through experiments on language modeling tasks. We compare the performance of DenseFormer architectures with the standard Transformer architecture using model size, inference time, training time, and final perplexity (sometimes abbreviated as PPL) as metrics. For each metric, we consider a baseline that performs the same as DenseFormer on that metric. Concretely, we include the following baselines:

**Same Depth Baseline.** A standard architecture with the same depth as the DenseFormer. This baseline has roughly the same number of parameters as DenseFormer given the negligible number of DWA parameters.

**Same Inference Time Baseline.** A standard architecture that has the same inference time as the DenseFormer. Since adding DWAs to the architecture has a computational overhead, this baseline has more layers (i.e. more capacity) than DenseFormer.

**Same Perplexity Baseline.** A standard architecture that has roughly the same performance as the DenseFormer. This baseline is usually much deeper than the DenseFormer, showcasing the benefits of using DWAs.

**Same Training Time Baseline.** A standard architecture that has the same training time as the DenseFormer. Since adding DWAs to the architecture has a computational overhead, this baseline is trained for more iterations than DenseFormer.

**Skips with Gains.** It can be observed that DenseFormer, among other things, allows scaling the output of the previous layer, providing more control than the original skip connections. Therefore, we provide an additional baseline to show this is not the only benefit offered by DenseFormer and emphasize the importance of having direct access to the outputs of earlier layers. In particular, we consider a modified version of the standard architecture where each skip connection also contains a learned scaling factor which

is applied to the the values moving through the skip connection before being summed with the output from a different layer (e.g. self-attention).

**Experimental setup.** We use models with 8 heads, each having 64 dimensions, and train them using batches of 400 sequences of length 256. We use rotary positional encoding (Su et al., 2021). We optimize the model using AdamW (Loshchilov & Hutter, 2017) with _β_ 1 = 0 _._ 9 and _β_ 2 = 0 _._ 95 with weight decay factor 0 _._ 1.

We perform most of our experiments on the OpenWebText2 dataset (Gao et al., 2020), an enhanced version of OpenWebTextCorpus (Gokaslan & Cohen, 2019) with around 17B tokens. We train all models for 40 _k_ steps, thus keeping the number of data points used in training fixed. We use learning rate 0 _._ 001 with a cosine scheduler and do a warmup in the beginning 5% steps.

We present the result of training 48 block and 72 block DenseFormers along with baselines of various sizes in Tab. 1. We make the following observations based on these results:

**Better perplexity than same depth baseline.** When comparing with a baseline of the same depth, DenseFormer significantly outperforms the standard architecture. Moreover, as it can be seen in Fig. 3, the perplexity of a 48 block DenseFormer is only matched by a 72 block Transformer baseline.

**Faster than a baseline with the same perplexity.** The performance of a 48 block DenseFormer is on par with a 72 block standard architecture. Still, the 48 block DenseFormer is much faster at inference (measured in terms of batches per second) than the 72 block standard architecture. Moreover, the number of parameters and memory footprint of the 72 block baseline is 45% larger than the one of the 48 block DenseFormer.

**Better perplexity than a baseline with the same inference time.** Comparing the 48 block DenseFormer without dilation, with a 64 block standard architecture (which has the same inference speed), shows a wide gap between the higher performance of DenseFormer (17 _._ 84) and the standard Transformer architecture (17 _._ 94). Considering DenseFormer models with a dilation of 4 and/or a DWA period of 5 would increase this gap further.

**Weighted skip-connections are insufficient.** DenseFormer is changing the flow of information in the model. We can wonder whether it leverages the additional expressivity or whether the performance gains could be explained by making it easier to rescale the contribution of each block. When comparing the 48 block Transformer baseline to the 48 block skip-with-gains baseline, it seems adding tunable weights to each skip connection does not lead to a significant improvement. When compared with the 48 block DenseFormer, this showcases the importance of having direct access to all previous layers.

|Model|Dilation_×_DWA Period|Depth|Parameters # (M)|Perplexity (_↓_)|Inference BPS (_↑_)|
|---|---|---|---|---|---|
|Transformer|-|48|378.45|18.61 (0.02)|5.94 (0.00)|
|Skips With Gains|-|48|378.45|18.45 (0.03)|5.72 (0.01)|
||1_×_1|48|378.45|17.84 (0.00)|4.65 (0.00)|
|DenseFormer|4_×_1|48|378.45|17.86 (0.02)|5.31 (0.01)|
||4_×_5|48|378.45|17.87 (0.02)|5.72 (0.00)|
|Transformer|-<br>-|64<br>72|491.72<br>548.35|17.94 (0.01)<br>17.82 (0.04)|4.57 (0.00)<br>4.08 (0.00)|
||1_×_1|72|548.36|17.12 (0.02)|2.93 (0.00)|
|DenseFormer|4_×_1|72|548.35|17.17 (0.00)|3.60 (0.00)|
||4_×_5|72|548.35|17.21 (0.01)|3.90 (0.00)|
|Transformer|-<br>-|84<br>90|633.31<br>675.78|17.48 (0.01)<br>17.44 (0.01)|3.54 (0.00)<br>3.32 (0.00)|

_Table 1._ **Performance of DenseFormer and the standard architecture of different sizes on OpenWebText2 dataset.** The number of millions of parameters is reported as well as the final perplexity. Additionally the number of batches of size 64 that can be processed in one second is reported as a measure of inference speed. The results are based on three runs with different seeds. The mean value is reported with the standard error reported in parenthesis. In terms of perplexity, DenseFormer clearly outperforms a standard Transformer of the same depth as well as standard Transformers with a similar inference speed. While sometimes a deeper model with the standard architecture can match the performance of a shallower DenseFormer (e.g. 72 block standard architecture and 48 block DenseFormer), inference using the shallow DenseFormer remains much faster. The inference speed is significantly improved with negligible effect on perplexity when increasing the dilation factor and DWA period. Adding a scaling factor to all skip connections in the standard architecture (named Skips with Gains) does not yield the same performance boost as DenseFormer highlighting the importance of inter-block connectivity in DenseFormer.

**Faster with dilation and DWA period.** Finally, our Tab.1 results show that for small dilation factors _k_ and DWA period _p_ , _k_ x _p_ -DenseFormer perform comparably while significantly boosting speed. Indeed, as can also be seen in Fig. 3, using 4x1-DenseFormers or 4x5-DenseFormers allows pushing the Pareto frontier on speed and performance trade-off forward.

**More efficient during training.** We train a 48 block 4x5DenseFormer and compare it against a 48 block Transformer baseline trained _with the same time budget_ . The baseline is therefore trained for more iterations (41 _._ 5k vs. 40k) to compensate for the time overhead of DenseFormer. In Fig. 4c we visualize the perplexity (approximated on a small subset of the validation set) dropping as a function of the training time. The DenseFormer’s perplexity is dropping faster than that of the Transformer baseline. This shows the superior efficiency of DenseFormer during training. While the Transformer is trained for more steps, thus using more data points, it is still outperformed by the DenseFormer. The final perplexities on full validation set reached by the two models can be seen in Tab. 3.

## **4.1 Additional Experiments**

We perform additional experiments to show that our results are general and extend to different settings and larger scales. We also study the impact of the dilation factor _k_ and DWA period _p_ on the efficiency of our _k_ x _p_ -DenseFormer architecture.

|Model|Depth|Perplexity|
|Transformer|24|20.13|
|1x1-DenseFormer|24|19.60|
|Transformer|48|18.94|
|1x1-DenseFormer|48|**18.43**|
|Transformer|72|18.44|

_Table 2._ **Comparison on PG-19.** Comparing DenseFormers and Transformers on the PG19 dataset. The results show similar improvements as the ones observed on the OpenWebText2 dataset. This demonstrates the generality of our results. Those results were obtained using a batch size of 128.

**PG-19 experiments.** Tab. 2 shows the performance of DenseFormer against standard Transformer on PG-19, which consists of a large collection of full-length books from Project Gutenberg (Rae et al., 2020). We trained both architectures for 48 _k_ steps and used a batch size of 128 instead of 400. All the other training parameters are kept the same as for OWT2 experiments. On this dataset, we can clearly see the superior performance of DenseFormer.

**Experiments with longer sequences.** Due to computation limitations, we can not run all experiments at a large scale. We however repeat a limited set of experiments with longer sequences of 512 tokens using a smaller batch size of 128. A 48 block Transformer baseline reaches a final perplexity of 18 _._ 28 _±_ 0 _._ 03 against 17 _._ 73 _±_ 0 _._ 02 for the DenseFormer. This result shows that the gap between the

**==> picture [483 x 131] intentionally omitted <==**

18 . 6 k x p -DenseFormer 18 . 6 k x p -DenseFormer 24 Transformer (41.5k iters)<br>Transformer Transformer 4x5-DenseFormer (40k iters)<br>18 . 4 18 . 4<br>22<br>18 . 2 18 . 2<br>4x20 4x20<br>20<br>4x15 4x15<br>18 . 0 4x10 18 . 0 4x10<br>12x1 12x1<br>1x1 2x1 4x1 [8][x][1] 4x45x8 1x1 2x1 4x1 [8][x][1] 44x5x8 18<br>17 . 8 17 . 8<br>5 . 0 5 . 5 6 . 0 1 . 0 1 . 2 1 . 4 2 4 6 8<br>Batches Per Second (during inference) Batches Per Second (during training) Time (hours)<br>(a) PPL vs. Inference speed (BPS) (b) PPL vs. Training speed (BPS) (c) Comparison under same training time<br>Perplexity Perplexity Perplexity<br>**----- End of picture text -----**<br>

_Figure 4._ **Training and inference efficiency of** _k_ **x** _p_ **-DenseFormer vs. Transformer.** For 48 block models, we compare in **(a)** the different perplexity/inference speed trade-offs reached by a regular Transformer and _k_ x _p_ -DenseFormers. In the top right corner, the Transformer baseline is the model with the worst perplexity but the fastest at inference. In contrast, the 1x1-DenseFormer in the bottom left corner, is reaching the best perplexity but incurs a cost in inference speed. By varying the dilation _k_ and DWA period _p_ , some _k_ x _p_ -DenseFormer models (e.g. 4x5) provide most of the perplexity improvement of the original DenseFormer while significantly reducing the time overhead. A similar analysis holds when looking at the training speed in **(b)** . In **(c)** , we show the perplexity decreasing during training. The x-axis is time. To compensate for the computational overhead of DenseFormer, we train the Transformer baseline for more iterations, such that the two methods have the same training time budget. We observe how our 4x5-DenseFormer is reaching a better perplexity faster than the baseline. The perplexity in this figure is computed on a small subset of the validation set to avoid slowing down the training.

|Model|Steps|Train time (h)|Perplexity|
|---|---|---|---|
|Standard|41500|8.09|18.33 (0.00)|
|4x5-DenseFormer|40000|8.04|**17.87 (0.02)**|

_Table 3._ **Same training time comparison.** Comparison of 4x5DenseFormer’s performance against a standard Transformer trained for more iterations. The number of training steps of the standard architecture is chosen such that the training time is roughly the same (and always more than) that of the DenseFormer. Both architectures have 48 blocks and are trained with 2000 warmup steps. Even though the Transformer is trained with more steps, it is still outperformed by the DenseFormer.

two architectures persists for longer sequences.

**Effect of Dilation and DWA period.** Fig. 4a and Fig. 4b show the impact of using different combinations of dilation _k_ and DWA period _p_ on the final perplexity, training and inference speed. As can be seen, small values of the dilation factor (e.g. up to 4) have a negligible effect on the perplexity. However, increasing the dilation factor further affects the performance more adversely while the gain in both training and inference speed starts to plateau. Increasing the DWA period also provides a similar trade-off, with the perplexity being barely affected for _p ≤_ 5. From those figures, we conclude that a dilation of 4 and a DWA period of 5 seem to offer the best compromise between speed and perplexity. In Appendix B.2, we provide more detailed results, including showing how increasing dilation yields a more pronounced speed-up for deeper models, making larger dilation factors more effective in those scales.

## **5 Analyzing the Information Flow**

In this section, we investigate the learned DWA _α_ weights to gain more insight and intuition on the reason behind the superiority of DenseFormer.

**A stable weight pattern emerges.** We start by visualizing the learnt DWA weights for 48 and 72-block DenseFormer models (with a dilation and period of 1) in Fig. 5. Interestingly, the _α_ weight patterns learned at both depths are very similar:

- High weights are on the diagonal (corresponding to the normal information flow as in a standard Transformer) as well as on the immediate previous blocks.

- High weights are given to the initial embedding vectors. Those weights are positive in earlier layers while later layers assign a negative weight.

- An aggregation block is observed near the final layers where a high weight is given to all previous layers in the block (seen as a high-weight triangle near the diagonal in the lower right corner).

Finally, Fig. 5c shows similar patterns persist to some extent when using dilation factors higher than 1. Similar results hold for _k_ x _p_ -DenseFormers as can be seen in Appendix B.1.

**Small weights matter.** In the visualized weight matrix of Fig. 5, most of the weights on the inter-block connections are small. This observation raises the question as to whether it is possible to drop most of these connections or not. To

**==> picture [484 x 332] intentionally omitted <==**

0 0 0<br>1 . 000 1 . 000 1 . 000<br>10<br>10 0 . 368 0 . 368 10 0 . 368<br>0 . 135 20 0 . 135 0 . 135<br>20 0 . 050 30 0 . 050 20 0 . 050<br>0 . 000 0 . 000 0 . 000<br>− 0 . 050 40 − 0 . 050 − 0 . 050<br>30 30<br>− 0 . 135 50 − 0 . 135 − 0 . 135<br>40 − 0 . 368 60 − 0 . 368 40 − 0 . 368<br>− 1 . 000 − 1 . 000 − 1 . 000<br>70<br>0 10 20 30 40 0 20 40 60 0 10 20 30 40<br>Xi Xi Xi<br>(a) 48 block DenseFormer. (b) 72 block DenseFormer. (c) 48-block 4x1-DenseFormer.<br>Figure 5. Visualization of DWA Learned Weights. Each row shows the weights α learned by a DWA module at a given depth. While the<br>heatmaps are averaged across 3 runs with different seeds, those patterns are very consistent across seeds. In (a) and (b) , strikingly similar<br>patterns can be observed in both 48 and 72 layer DenseFormers. 48 and 72 layer DenseFormers. and 72 layer DenseFormers. 72 layer DenseFormers. layer DenseFormers. In (c) , we show the learned weights for a 48 block DenseFormer trained 48 block DenseFormer trained block DenseFormer trained<br>with a dilation of 4. 4.. Despite the sparsity, we still observe a very similar pattern to those learned by the non-dilated models.<br>20 . 0<br>0 . 20 DenseFormer<br>Transformer 19 . 5<br>0 . 15<br>19 . 0<br>0 . 10 18 . 5<br>DenseFormer<br>0 . 05 18 . 0 Transformer<br>17 . 5<br>0 . 00 0 10 20 30 40 50<br>1 6 12 18 24 30 36 42 48<br>Sparsity (% of α dropped)<br>Depth<br>scale) scale) scale)<br>Index (symlog Index (symlog Index (symlog<br>DWA weights DWA weights DWA weights<br>DWA DWA DWA<br>similarity<br>Perplexity<br>Cosine<br>**----- End of picture text -----**<br>

_Figure 5._ **Visualization of DWA Learned Weights.** Each row shows the weights _α_ learned by a DWA module at a given depth. While the heatmaps are averaged across 3 runs with different seeds, those patterns are very consistent across seeds. In **(a) and (b)** , strikingly similar patterns can be observed in both 48 and 72 layer DenseFormers. 48 and 72 layer DenseFormers. and 72 layer DenseFormers. 72 layer DenseFormers. layer DenseFormers. In **(c)** , we show the learned weights for a 48 block DenseFormer trained 48 block DenseFormer trained block DenseFormer trained with a dilation of 4. 4.. Despite the sparsity, we still observe a very similar pattern to those learned by the non-dilated models.

_Figure 7._ **Performance after dropping small DWA weights.** The figure shows the performance when the model is trained with no sparsity induced (dilation and period of 1) and the DWA weights are later sparsified based on their magnitude at inference. One can see that the perplexity quickly explodes after only sparsifying 15% of the weights. This observation suggests that even though many of the DWA weights are small (as can be seen in Fig. 5) they still play an important role in the output of the model.

_Figure 6._ **Cosine similarity between the output of each DWA module and the initial embedding vectors.** The results are averaged over three seeds, for DenseFormer models with 48 blocks and no dilation (corresponding to the weights in Fig. 1a). The model initially maintains a high correlation with the output of each DWA modules, but reduces that correlation towards later layers. Intuitively, we can hypothesize that this is the stage where the model is preparing to output the next token. A very similar plot can be observed for 72 block models in Appendix B.1.

sparsity pattern allowing DWA to access previous _k_ blocks as well as the input embedding vectors calling it “Last K”. Furthermore, given the large magnitude of weights on the last layer, we also experiment with only having a single DWA in the architecture which is placed after the last layer, calling it “Connect to Last”. Tab. 4 shows the perplexities when using these sparsity patterns and shows that they do not achieve the boost in performance obtained with DenseFormer. This observation further strengthens the importance of small DWA weights both during training and inference.

answer this question, we plot the perplexity after dropping a portion of the smallest DWA weights and report the results in Fig. 7. It can be seen that even though a large portion of weights in Fig. 5 are small, dropping beyond 15% of the weights leads to a significant increase in perplexity. Therefore, even though these weights are small, they seemingly play an important role in predicting the next token.

**Other sparsity patterns during training.** Alternative to pruning after training, we also consider imposing different sparsity patterns during training. We already have experimented with such sparsity patterns induced through dilation and DWA period. In Fig. 5, we observe that in many cases the largest weights are given to the few previous blocks as well as the first block. As such, we experiment with a

**Correlation with Input Embeddings.** Based on the special pattern of weights given to the embedding vectors— especially the negative weights given to the input by the last layers (see Fig. 5)—we hypothesize that the model tries to use the information in the input in earlier layers while removing the influence of the current token as it tries to predict

|Sparsity Pattern|Perplexity|Sparsity|
|Baseline Transformer|18.61(0.02)|-|
|Last K (_K_ = 4)|18.28 (0.01)|84%|
|Connect to Last|18.33(0.02)|96%|
|12x1-DenseFormer|17.96 (0.01)|92%|
|4x5-DenseFormer|**17.87 (0.02)**|95%|

_Table 4._ **Alternative DWA Sparsity Patterns.** We compare 48 block architectures with different sparsity patterns. In Last K, DWA can only access the output of the last _k_ blocks as well as the input embedding vectors. Connect to Last includes only a single DWA module in the architecture placed after the last layer, i.e. connecting every block to the last block. Neither of these patterns achieves the same perplexity boost as our proposed DenseFormer. 12x1 and 4x5-DenseFormers—with sparsities of respectively 92% and 95%—outperform other sparsity patterns, which signifies the importance of pairwise inter-block connections.

the next token. In order to test this hypothesis we plot the average cosine similarity of each token’s vector after each DWA block with its input embedding in Fig. 6. As expected based on the weight pattern, we observe that the similarity is high in the earlier layers. At the later stage, the model decreases this similarity significantly. We hypothesize this final decrease is due to the model moving from processing the current token to building the next token’s representation. In contrast, the similarity drops down very early for a standard transformer and remains low for the rest of the layers.

## **6 Future Work & Conclusion**

In this paper, we introduced the DenseFormer architecture. This architecture adds an averaging module called DWA after each block which allows it to directly access the outputs of previous blocks. We established the superiority of this architecture over Transformers in terms of perplexity/speed trade-off through experiments in a variety of settings. Additionally, we provided dilation and DWA periodicity as simple methods to improve speed without significantly hurting performance. Finally, we provided insights about the learned weights, revealing patterns persisting over different depths.

As the next steps, finding more efficient implementations of DenseFormer is grounds for future work. One possible direction is finding better sparsity patterns that also can be implemented efficiently. The weights visualization in Fig. 5 suggests such patterns might exist. Furthermore, finding efficient ways to shard DWA across multiple nodes is important to allow large-scale distributed training.

## **Impact Statement**

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## **Acknowledgement**

This project was supported by SNSF grant number 200020 ~~2~~ 00342.

## **References**

- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language Models are Few-Shot Learners, July 2020. URL http://arxiv.org/abs/2005. 14165. arXiv:2005.14165 [cs].

- Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. _CoRR_ , abs/2307.08691, 2023.

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C.´ Flashattention: Fast and memory-efficient exact attention with io-awareness. In _NeurIPS_ , 2022.

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.

- ElNokrashy, M., AlKhamissi, B., and Diab, M. Depth-wise attention (dwatt): A layer fusion method for data-efficient classification. _arXiv preprint arXiv:2209.15168_ , 2022.

- Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. _Journal of Machine Learning Research_ , 23(120):1–39, 2022.

- Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. OpenWebText2 dataset, as part of ‘the Pile: An 800gb dataset of diverse text for language modeling‘. _arXiv preprint arXiv:2101.00027_ , 2020.

- Gokaslan, A. and Cohen, V. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.

- He, B. and Hofmann, T. Simplifying transformer blocks. _CoRR_ , abs/2311.01906, 2023.

- He, K. and Sun, J. Convolutional neural networks at constrained time cost. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 5353– 5360, 2015.

- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016.

- Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. In _CVPR_ , pp. 2261–2269. IEEE Computer Society, 2017.

- Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are RNNs: Fast autoregressive transformers with linear attention. In _Proceedings of the International Conference on Machine Learning (ICML)_ , pp. 5294–5303, 2020. URL https://fleuret.org/papers/ katharopoulos-et-al-icml2020.pdf.

- Kitaev, N., Kaiser, L., and Levskaya, A. Reformer: The efficient transformer. In _ICLR_ . OpenReview.net, 2020.

- Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., and Shi, S. Encouraging divergent thinking in large language models through multi-agent debate. _arXiv preprint arXiv:2305.19118_ , 2023.

- Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017.

- Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with self-feedback. _arXiv preprint arXiv:2303.17651_ , 2023.

- Mohtashami, A., Pagliardini, M., and Jaggi, M. Cotformer: More tokens with attention make up for less depth. In _Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ NeurIPS 2023)_ , 2023.

- OpenAI. GPT-4 Technical Report, March 2023. URL http://arxiv.org/abs/2303.08774. arXiv:2303.08774 [cs].

- Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., and Lerer, A. Automatic differentiation in pytorch. 2017.

- Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, R., Cappelli, A., Alobeidli, H., Pannier, B., Almazrouei, E., and Launay, J. The refinedweb dataset for falcon LLM: outperforming curated corpora with web data, and web data only. _CoRR_ , abs/2306.01116, 2023.

- Petty, J., van Steenkiste, S., Dasgupta, I., Sha, F., Garrette, D., and Linzen, T. The impact of depth and width on transformer language model generalization. _CoRR_ , abs/2310.19956, 2023a.

- Petty, J., van Steenkiste, S., Dasgupta, I., Sha, F., Garrette, D., and Linzen, T. The impact of depth and width on transformer language model generalization. _arXiv preprint arXiv:2310.19956_ , 2023b.

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners, 2019.

- Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., and Lillicrap, T. P. Compressive transformers for long-range sequence modelling. In _ICLR_ . OpenReview.net, 2020.

- Shazeer, N. Fast transformer decoding: One write-head is all you need. _CoRR_ , abs/1911.02150, 2019.

- Shazeer, N. GLU variants improve transformer. _CoRR_ , abs/2002.05202, 2020.

- Srivastava, R. K., Greff, K., and Schmidhuber, J. Highway networks. _arXiv preprint arXiv:1505.00387_ , 2015.

- Su, J., Lu, Y., Pan, S., Wen, B., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. _CoRR_ , abs/2104.09864, 2021.

- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,` Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models, 2023a.

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models, 2023b.

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. _CoRR_ , abs/1706.03762, 2017. URL http://arxiv.org/abs/1706.03762.

- Villalobos, P., Sevilla, J., Heim, L., Besiroglu, T., Hobbhahn, M., and Ho, A. Will we run out of data? an analysis of the limits of scaling datasets in machine learning. _arXiv preprint arXiv:2211.04325_ , 2022.

Wang, B. and Komatsuzaki, A. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/ mesh-transformer-jax, May 2021.

- Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. _CoRR_ , abs/2006.04768, 2020.

## **A Implementation details**

In this section, we provide several implementations in Pytorch (Paszke et al., 2017). The first implementations we propose are very simple and rely on looping over the past representations to compute each DWA output. As a result, those naive implementations are slow when dilation and DWA periodicity are not used. In a second time, we propose a more optimized and faster implementation. As a comparison, for a 4x5-DenseFormer with 48 blocks, the naive implementation takes 674ms per training iteration against 657ms for the later implementation. In Appendix A.2, we provide a denseformer python package to help turn Transformers into DenseFormers in only 3 simple steps.

**TL;DR:** If you want to implement your own DenseFormer, simply check the denseformer python package: https: //github.com/epfml/DenseFormer

## **A.1 Naive Pytorch implementation**

**Naive** 1 **x** 1 **-DenseFormer implementation.** A naive implementation in Pytorch would consist of storing the output of each block during the forward, and feeding those representations to DWA modules after each block. A pseudocode would look like the following:

_Listing 1._ Naive 1x1-DenseFormer implementation

1 import torch 2 3 4 class DWA(torch.nn.Module): 5 6 def __init__(self, n_alphas): 7 super().__init__() 8 self.n_alphas = n_alphas 9 alphas = torch.zeros((n_alphas,)) 10 alphas[-1] = 1.0 11 self.alphas = torch.nn.Parameter(alphas) 12 13 def forward(self, all_previous_x): 14 weighted_avg = all_previous_x[0] * self.alphas[0] 15 for i in range(1, self.n_alphas): 16 weighted_avg += self.alphas[i] * all_previous_x[i] 17 return weighted_avg 18 19 20 class GPTBase(torch.nn.Module): 21 22 def __init__(self, config): 23 super().__init__() 24 self.config = config 25 self.dwa_modules = torch.nn.ModuleList([DWA(n_alphas=i+2) for i in range(config.n_blocks)]) 26 self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd) 27 self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.n_blocks)]) 28 self.ln_f = LayerNorm(config.n_embd, bias=config.bias) 29 self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False) 30 self.transformer.wte.weight = self.lm_head.weight # weight tying 31 32 def forward(self, idx): 33 x = self.wte(idx) 34 all_previous_x = [x] # This stores all the intermediary representations 35 for i in range(self.config.n_blocks): 36 x = self.blocks[i](x) 37 all_previous_x.append(x) 38 x = self.dwa_modules[i](all_previous_x) # Computing the weighted average 39 x = self.ln_f(x) 40 logits = self.lm_head(x) 41 return logits

**Naive** _k_ **x** _p_ **-DenseFormer implementation.** Here we introduce a naive implementation with dilation and DWA-frequency. The DWA module in the following implementation would be the same as for the 1x1-DenseFormer naive implementation.

_Listing 2._ Naive _k_ x _p_ -DenseFormer implementation

1 class GPTBase(torch.nn.Module): 2 3 def __init__(self, config): 4 super().__init__() 5 self.config = config 6 self.dwa_modules = torch.nn.ModuleList([DWA(n_alphas=(i+1+config.dilation)//config.dilation) for i in range(config.n_blocks)]) 7 self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd) 8 self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.n_blocks)]) 9 self.ln_f = LayerNorm(config.n_embd, bias=config.bias) 10 self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False) 11 self.transformer.wte.weight = self.lm_head.weight # weight tying 12 13 def forward(self, idx): 14 x = self.wte(idx) 15 all_previous_x = [x] # This stores all the intermediary representations 16 for i in range(self.config.n_blocks): 17 x = self.blocks[i](x)

18 all_previous_x.append(x) 19 if (i+1) % 20 all_previous_x_dilated = [x_ for j, x_ in enumerate(all_previous_x) if (len(all_previous_x)-1-j) % 21 x = self.dwa_modules[i](all_previous_x_dilated) # Computing the weighted average with dilation 22 x = self.ln_f(x) 23 logits = self.lm_head(x) 24 return logits

## **A.2 More Optimized Pytorch implementation**

The point of this more optimized implementation is to remove the for loops of the naive implementation, and instead use tensor operations. To achieve this, we need to create a tensor containing the previous representations instead of a list. Given the structure of this tensor which will simply accumulate all the past representation during the forward pass, we would want to pre-allocate an accumulator and update this accumulator with a new representation after each block. However, this implies using in-place operations that conflict with Pytorch’s autograd. We find a workaround by defining the InPlaceSetSlice class. We implement a helper module DWAModules built on top of this new class:

_Listing 3._ Content of the denseformer python package

1 import torch 2 3 4 class InPlaceSetSlice(torch.autograd.Function): 5 6 @staticmethod 7 def forward(ctx, full_tensor, last_slice, x_idx, x_val): 8 full_tensor[x_idx] = x_val 9 ctx.x_idx = x_idx 10 ret = torch.Tensor().to(full_tensor.device) 11 ret.set_(full_tensor[:x_idx + 1]) 12 return ret 13 14 @staticmethod 15 def backward(ctx, grad_out): 16 if ctx.x_idx == 0: 17 return None, None, None, grad_out[ctx.x_idx] 18 else: 19 return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx] 20 21 22 def apply_inplace_set(x_acc, x_idx, x_val): 23 full_tensor, last_slice = x_acc 24 new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val) 25 return full_tensor, new_slice 26 27 28 class DWAModules(torch.nn.Module): 29 30 def __init__(self, n_blocks, dilation=1, period=1): 31 super().__init__() 32 self.n_blocks = n_blocks 33 self.dilation = dilation 34 self.period = period 35 self.alphas = torch.nn.ModuleList([torch.nn.Linear((i+1+dilation)//dilation, 1, bias=False) if (i+1)% 36 self.accumulators = None 37 self._init_weights() 38 39 def _init_weights(self): 40 for module in self.alphas: 41 if module is not None: 42 module.weight.data.zero_() 43 module.weight.data[0, -1] = 1. 44 45 def init_accumulators(self, x): 46 x_accs = [] 47 for i in range(self.dilation): 48 current_group_size = (self.n_blocks + 1) // self.dilation 49 if i < (self.n_blocks + 1) % 50 current_group_size += 1 51 x_accs.append((torch.zeros((current_group_size, *x.shape), device=x.device, dtype=x.dtype), None)) 52 x_accs[0] = apply_inplace_set(x_accs[0], 0, x) 53 self.accumulators = x_accs 54 55 def forward(self, x, block_idx): 56 assert self.accumulators is not None, "‘init_accumulators(x)‘ needs to be called first" 57 self.accumulators[(block_idx+1) % 58 self.accumulators[(block_idx+1) % 59 (block_idx+1)//self.dilation, 60 x 61 ) 62 if (block_idx+1) % 63 x = torch.tensordot(self.alphas[block_idx].weight.view(-1), self.accumulators[(block_idx+1)% 64 return x

**denseformer python package.** The above module is available in the denseformer package which can be installed through the following link: https://github.com/epfml/DenseFormer. It provides the DWAModules class

which orchestrates all the DWA logic given a number of blocks, a dilation factor, and a DWA period. After installing the package, a Transformer can be turned into a DenseFormer in three simple steps:

_Listing 4._ Faster _k_ x _p_ -DenseFormer implementation using the denseformer package.

1 import torch 2 from denseformer import DWAModules 3 4 class DenseFormer(torch.nn.Module): 5 6 def __init__(self, config): 7 super().__init__() 8 self.config = config 9 self.dwa_modules = DWAModules(config.n_blocks, config.dilation, config.dwa_period) # Step 1 10 self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd) 11 self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.n_blocks)]) 12 self.ln_f = LayerNorm(config.n_embd, bias=config.bias) 13 self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False) 14 self.transformer.wte.weight = self.lm_head.weight 15 16 def forward(self, idx): 17 x = self.wte(idx) 18 self.dwa_modules.init_accumulators(x) # Step 2 19 for i in range(self.config.n_blocks): 20 x = self.blocks[i](x) 21 x = self.dwa_modules(x, block_idx=i) # Step 3 22 x = self.ln_f(x) 23 logits = self.lm_head(x) 24 return logits

## **B Additional Results**

## **B.1 Information Flow**

**Visualizing** _α_ **s with periodicity.** Similarly to Fig. 5, we show in Fig. 8 the DWA weights for 4x3, 4x4, and 4x5DenseFormers. We observe patterns similar to the 1x1-DenseFormer in Fig. 5 but at a lower resolution.

**==> picture [484 x 136] intentionally omitted <==**

0 1 . 000 0 1 . 000 0 1 . 000<br>0 . 368 0 . 368 0 . 368<br>10 10 10<br>0 . 135 0 . 135 0 . 135<br>20 0 . 050 20 0 . 050 20 0 . 050<br>0 . 000 0 . 000 0 . 000<br>− 0 . 050 − 0 . 050 − 0 . 050<br>30 30 30<br>− 0 . 135 − 0 . 135 − 0 . 135<br>40 − 0 . 368 40 − 0 . 368 40 − 0 . 368<br>− 1 . 000 − 1 . 000 − 1 . 000<br>0 10 20 30 40 0 10 20 30 40 0 10 20 30 40<br>Xi Xi Xi<br>(a) 48-block 4x3-DenseFormer. (b) 48-block 4x4-DenseFormer. (c) 48-block 4x5-DenseFormer.<br>scale) scale) scale)<br>Index (symlog Index (symlog Index (symlog<br>DWA weights DWA weights DWA weights<br>DWA DWA DWA<br>**----- End of picture text -----**<br>

_Figure 8._ **Visualization of DWA Learned Weights.** Each row shows the weights _α_ learned by a DWA module at a given depth. Those patterns are very consistent with the ones learned by a 1x1-DenseFormer, as seen in Fig. 5.

**Correlation with input embeddings at** 72 **blocks.** As in Fig. 6, we analyze the cosine similarity between the output of each DWA module and the initial embedding vectors for models of 72 blocks. The results in Fig. 9 are very consistent with those obtained with shallower models (Fig. 6).

**==> picture [147 x 90] intentionally omitted <==**

0 . 25 DenseFormer<br>0 . 20 Transformer<br>0 . 15<br>0 . 10<br>0 . 05<br>0 . 00<br>1 6 12 18 24 30 36 42 48 54 60 66 72<br>Depth<br>similarity<br>Cosine<br>**----- End of picture text -----**<br>

_Figure 9._ **Cosine similarity between the output of each DWA module and the initial embedding vectors.** The results are averaged over three seeds, for DenseFormer models with 72 blocks and no dilation. The model initially maintains a high correlation with the output of each DWA modules, but reduces that correlation towards later layers. Intuitively, we can hypothesize that this is the stage where the model is preparing to output the next token.

**Evolution of DWA weights during training.** In Fig. 10, we plot the DWA weights of a 48 block DenseFormer during training. We observe how the pattern is learned relatively fast, within the first 5000 iterations.

**==> picture [458 x 95] intentionally omitted <==**

0 0 0 0 0 0 0<br>10 10 10 10 10 10 10<br>20 20 20 20 20 20 20<br>30 30 30 30 30 30 30<br>40 40 40 40 40 40 40<br>0 10 20 30 40 0 10 20 30 40 0 10 20 30 40 0 10 20 30 40 0 10 20 30 40 0 10 20 30 40 0 10 20 30 40<br>Xi Xi Xi Xi Xi Xi Xi<br>(a) Step 0 (b) Step 1000 (c) Step 2000 (d) Step 3000 (e) Step 4000 (f) Step 5000 (g) Step 6000<br>Index Index Index Index Index Index Index<br>DWA DWA DWA DWA DWA DWA DWA<br>**----- End of picture text -----**<br>

_Figure 10._ **Rapid convergence of DWA weights during training.** The DWA weights are rapidly converging to their final pattern. After 5000 iterations, the weight pattern already looks very similar to the one in Fig. 5.

## **B.2 Analysis of Dilation and DWA Period**

**More detailed analysis of dilation.** For 48 block models, we study the impact of varying the dilation factor _k_ , we do not vary the DWA period which is set to 1. The results of this experiment are in Fig. 11. We observe how small dilation coefficients do not significantly deteriorate the perplexity yet increase the inference speed.

**==> picture [373 x 98] intentionally omitted <==**

18 . 6<br>18 . 4 1 . 2<br>DenseFormer<br>18 . 2 Transformer<br>1 . 1<br>18 . 0 DenseFormer (48 Blocks)<br>DenseFormer (72 Blocks)<br>17 . 8 1 . 0<br>1 2 4 6 8 12 1 2 4 6 8 12<br>Dilation Factor Dilation Factor<br>Improvement<br>Perplexity<br>BPS<br>Rel.<br>**----- End of picture text -----**<br>

**==> picture [411 x 9] intentionally omitted <==**

(a) Perplexity of 48 block k x1-DenseFormer on OWT2 (b) Relative speed improvement over dilation factor 1<br>**----- End of picture text -----**<br>

_Figure 11._ **Effect of the Dilation Factor** _k_ **on Speed and Performance.** Part **(a)** shows the degradation in perplexity as we increase the dilation factor of _k_ x1-DenseFormer models. A noticeable drop in performance occurs for larger dilation factors, e.g. after _k_ = 4. However, surprisingly, 12-Dilated DenseFormer still outperforms the Transformer baseline. As shown in **(b)** , while the perplexity is not so impacted by dilation, the inference speed is significantly improved. Interestingly, the speed gain also plateaus for larger values of _k_ , e.g. roughly _k_ = 4 for 48 blocks. The gain increases with the depth of the DenseFormer, and the plateau threshold occurs later for deeper models.

**More detailed analysis of the DWA period.** For 48 block models, we study the impact of varying the DWA period _p_ . We do not vary the dilation which is set to 4. In Fig. 12, we observe the impact of increasing _p_ on the perplexity. Interestingly, the perplexity profile is non-monotonic in _p_ , which exposes the interplay between _k_ , _p_ , and the depth of the model. Moreover, increasing the DWA period further increases the inference speed over increasing the dilation.

**==> picture [373 x 99] intentionally omitted <==**

18 . 6<br>1 . 2<br>18 . 4<br>DenseFormer 4 × p<br>18 . 2 Transformer 1 . 1<br>18 . 0<br>17 . 8 1 . 0<br>1 3 4 5 6 7 8 9 10 15 20 1 3 4 5 6 7 8 9 10 15 20<br>DWA period p DWA period p<br>Improvement<br>Perplexity<br>BPS<br>Rel.<br>**----- End of picture text -----**<br>

- (a) Perplexity of 48 block 4x _p_ -DenseFormer on OWT2 (b) Relative speed improvement over 1x1-DenseFormer

_Figure 12._ **Effect of the DWA period** _p_ **on Speed and Performance.** Part **(a)** shows the degradation in perplexity as we increase the DWA period of 4x _p_ -DenseFormer models. Surprisingly, a 4x20-DenseFormer still outperforms the Transformer baseline. As shown in **(b)** , while the perplexity is not so impacted, the inference speed is significantly improved.

## **B.3 Delaying the Training of DWA Weights**

In this section, we study what would happen if we started training the DWA weights at different training iterations. As seen in Fig. 10, the DWA weights are rapidly converging to their final values within the first 5000 iterations. Moreover, the initialization of the DWA weights corresponds to the same flow of information as in a normal transformer. This raises the question of whether training the DWA weights during the first training iterations is important, or whether a pre-trained model would still gain from adding the DWA weights later. To answer this question we experiment with training the DWA-weights after _N_ iterations. We do not modify the learning rate scheduler or any hyperparameter besides _N_ . Results in Tab. 5 show a diminishing return as _N_ increases. It seems important to tune the DWA weights from the beginning. A possible hypothesis could be that the iterates commit to a valley in the loss landscape relatively early during training. Once deciding to go to the valley where DWA weights are not used, it is difficult to recover and ultimately benefit from newly added DWA weights. We believe this phenomenon could be mitigated using a better learning rate scheduler. We leave this investigation as future work.

|Model<br>N|Perplexity|
|---|---|
|Baseline Transformer<br>-|18.61(0.02)|
|4x5-DenseFormer<br>0<br>4x5-DenseFormer<br>1k<br>4x5-DenseFormer<br>2k<br>4x5-DenseFormer<br>4k<br>4x5-DenseFormer<br>6k<br>4x5-DenseFormer<br>10k<br>4x5-DenseFormer<br>20k<br>4x5-DenseFormer<br>30k|**17.87 (0.02)**<br>17.99<br>18.07<br>18.13<br>18.17<br>18.23<br>18.33<br>18.40|

_Table 5._ **Start training the DWA weights after** _N_ **iterations.** At initialization, a DenseFormer is the same as a Transformer. We experiment with tuning the DWA weights only after _N_ iterations. This means the model is trained as a Transformer for _N_ iterations, and as a DenseFormer from _N_ to 40k iterations.

## **B.4 Rank Analysis**

In this section, we compare the ranks of matrices learned using DenseFormer and Transformer architectures. Our main result in Fig. 13 is that there is no significant difference in rank between the two approaches.

**==> picture [385 x 216] intentionally omitted <==**

15<br>10 Transformer 4 Transformer Transformer<br>DenseFormer DenseFormer 10 DenseFormer<br>5 2<br>5<br>0 0 0<br>0 250 500 750 0 200 400 600 800 0 250 500 750<br>Ranked singular values Ranked singular values Ranked singular values<br>(a) QKV attn matrix (b) Output attn matrix (c) MLP up proj.<br>40<br>Transformer Transformer<br>7 . 5 DenseFormer DenseFormer<br>5 . 0 20<br>2 . 5<br>0 . 0 0<br>0 200 400 600 800 0 250 500 750<br>Ranked singular values Ranked singular values<br>(d) MLP down proj. (e) Embeddings<br>Intensity Intensity Intensity<br>Intensity Intensity<br>**----- End of picture text -----**<br>

_Figure 13._ **Ranked singular values averaged across blocks.** For 48 block models, we average the singular values for each matrix across blocks (except for the embedding matrix). We observe no significant differences between Transformers and DenseFormers. Results are averaged over 3 seeds.

## **B.5 Experiments with a batch size of** 128

In this section, we revisit experiments from the main paper but use a small batch size of 128 instead of 400 during training.

**Speed and performance trade-off.** In Fig. 14 we show the trade-off between inference speed and perplexity for different numbers of blocks. Similarly to Fig. 3, DenseFormers reach a better perplexity than much deeper Transformer models. Interestingly, the perplexity gap is larger than when using larger batches (compared to a batch size of 400 used in Fig. 3). A 48 block DenseFormer is performing on par with a 90 block Transformer. This might indicate that the DenseFormer is more robust to large gradient noise compared to Transformers. DenseFormers reach better trade-offs in terms of inference speed and perplexity. Those results are expected to improve if we were to train a 4x5-DenseFormer instead of a 4x1-DenseFormer. Detailed results can be seen in Tab. 6.

**Results with other sparse patterns.** In Tab. 7 we reproduce the experiments of Tab. 4 but with a batch size of 128. Similar conclusions follow.

**==> picture [171 x 171] intentionally omitted <==**

DenseFormer<br>23 . 5 Transformer 48<br>72<br>23 . 0 84<br>90<br>48<br>22 . 5<br>72<br>22 . 0<br>84<br>90<br>3 4 5 6<br>Batches Per Second<br>Perplexity<br>**----- End of picture text -----**<br>

_Figure 14._ **Speed and performance trade-off.** Comparison of speed and performance trade-off between the standard Transformer architecture and 4x1-DenseFormer. The number of blocks in each architecture is reported next to the data-point. All DenseFormer models on this plot use a dilation factor of 4. **Comparing perplexities:** Considering only the perplexity (y-axis), a 48 layer DenseFormer strikingly outperforms much deeper Transformer baselines. **Comparing trade-offs:** A 48 layer 4-Dilated DenseFormer matches the better perplexity of a 90 layer Transformer while being 1 _._ 6 _×_ faster at inference.

|Model|Dilation|Depth|Parameters # (M)|Perplexity|Inference BPS|
|---|---|---|---|---|---|
|Transformer|-|48|378.45|23.67 (0.09)|5.98 (0.00)|
|Skips With Gains|-|48|378.45|23.78 (0.19)|5.72 (0.01)|
||1|48|378.45|22.61 (0.05)|4.67 (0.00)|
|DenseFormer|2|48|378.45|**22.60 (0.04)**|5.15 (0.01)|
||4|48|378.45|22.68 (0.06)|**5.36 (0.00)**|
|Transformer|-<br>-|64<br>72|491.72<br>548.35|23.21 (0.07)<br>23.10 (0.02)|4.59 (0.00)<br>4.12 (0.00)|
||1|72|548.36|**21.81 (0.00)**|2.93 (0.00)|
|DenseFormer|2|72|548.35|21.92 (0.04)|3.39 (0.00)|
||4|72|548.35|22.03 (0.04)|**3.62 (0.00)**|
|Transformer|-<br>-|84<br>90|633.31<br>675.78|22.84 (0.07)<br>22.67 (0.04)|3.56 (0.00)<br>3.35 (0.00)|

_Table 6._ **Performance of DenseFormer and the standard architecture of different sizes on OpenWebText2 dataset.** Using a batch size of 128, and a DWA period of 1. DenseFormer clearly outperforms a standard architecture of the same depth as well as standard architecture with the same inference speed. While sometimes a deeper model with the standard architecture can match the performance of a shallower DenseFormer, inference using the shallow DenseFormer remains much faster.

|Sparsity Pattern|Perplexity|Sparsity|
|Baseline Transformer|23.67 (0.09)|-|
|12x1-DenseFormer|**22.91 (0.06)**|92%|
|Last K (_K_ = 4)|23.23 (0.07)|84%|
|Connect to Last|23.45 (0.05)|96%|

_Table 7._ **Alternative DWA Sparsity Patterns.** We compare 48 block architectures with different sparsity patterns. In Last K, DWA can only access the output of the last _k_ blocks as well as the embedding vectors. Connect to Last includes only a single DWA module in the architecture placed after the last layer, i.e. connecting every block to the last block. Neither of these patterns allows achieving the same perplexity boost as the original DenseFormer. Even with a dilation of 12, which implies a sparsity of 92%, the Denseformer models outperform other sparsity patterns, which signifies the importance of pairwise inter-block connections.

**Visualizing the DWA weights (trained with a batch size of** 128 **).** In Fig. 15 we plot the DWA weights obtained when training with a batch size of 128. The learned patterns are very consistent with the ones of Fig. 5 obtained with a larger

## batch size.

**==> picture [484 x 137] intentionally omitted <==**

0 0 0<br>1 . 000 1 . 000 1 . 000<br>10<br>10 0 . 368 0 . 368 10 0 . 368<br>0 . 135 20 0 . 135 0 . 135<br>20 0 . 050 30 0 . 050 20 0 . 050<br>0 . 000 0 . 000 0 . 000<br>− 0 . 050 40 − 0 . 050 − 0 . 050<br>30 30<br>− 0 . 135 50 − 0 . 135 − 0 . 135<br>40 − 0 . 368 60 − 0 . 368 40 − 0 . 368<br>− 1 . 000 − 1 . 000 − 1 . 000<br>70<br>0 10 20 30 40 0 20 40 60 0 10 20 30 40<br>Xi Xi Xi<br>(a) 48 block DenseFormer. (b) 72 block DenseFormer. (c) 48-block 4x1-DenseFormer.<br>scale) scale) scale)<br>Index (symlog Index (symlog Index (symlog<br>DWA weights DWA weights DWA weights<br>DWA DWA DWA<br>**----- End of picture text -----**<br>

_Figure 15._ **Visualization of DWA Learned Weights.** Each row shows the weights _α_ learned by a DWA module at a given depth. While the heatmaps are averaged across 3 runs with different seeds, those patterns are very consistent across seeds. In **(a) and (b)** , strikingly similar patterns can be observed in both 48 and 72 layer DenseFormers. In **(c)** , we show the learned weights for a 48 block DenseFormer trained with a dilation of 4. Despite the sparsity, we still observe a very similar pattern to those learned by the non-dilated models.
