# Iterative Reverse Image Search (IRIS)

Recent advancements in computer vision underscore the need for large, labeled datasets to develop robust and efficient deep learning-based vision models. However, creating such extensive datasets through manual annotation is both cost-prohibitive and impractical due to the highly imbalanced nature of real-world datasets, where negative samples, which are of minimal interest, predominate. In this work, we introduce the Iterative Reverse Image Search (IRIS) algorithm, an iterative approach designed to alleviate the data annotation burden significantly. We propose a search algorithm that combines Nearest Neighbor search and a modified Diversity Sampler on the embeddings of pre-trained self-supervised models to discover samples similar to a given query sample. These components collaboratively guide the annotator to label only those samples identified as likely positive, thereby reducing the labeling of negative samples and accelerating the discovery of new samples while minimizing human effort. Although the idea can be easily extended to any domain, we strictly confine ourselves to computer vision here.

# Definitions
- **Unlabelled Dataset**: $\mathbb{D}_{\text{images}}$
  The complete set of all unlabelled images available to the algorithm.

- **Seed Samples**:
  A small set of initial samples selected from $\mathbb{D}_{\text{images}}$, which are used as a reference. The algorithm attempts to find samples similar to these.

- **Positive Samples**:
  Samples from $\mathbb{D}_{\text{images}}$ that are labelled (by the annotator) to belong to the same class as the seed samples. Any set of positive samples is denoted by a ```+``` in the superscript eg $A^+$

- **Negative Samples**:
  Samples from $\mathbb{D}_{\text{images}}$ that are labelled (by the annotator) to belong to a different class than the seed samples. Any set of negative sample is denoted by a ```-``` in the superscript eg $A^-$

- **Feature Extractor**:  
  The feature extractor is the pre-trained backbone that converts images into feature vectors. It is denoted by $f_{\text{enc}}$.

- **$f_c$ (MLP Head)**:  
  A single-layer MLP head that maps the latent space to $\mathbb{R}$, typically used to assign confidence scores or make predictions based on the feature representations.

- **Latent Dataset** ($\mathcal{D}$):  
  The full dataset is represented in the latent space:  

$$
  \mathcal{D} := \{ f_{\text{enc}}(d) \mid d \in \mathbb{D}_{\text{images}} \}
$$

- **Labelled Set** ($\mathcal{L}$):  
  The set of all labelled samples (both positive and negative) grows over successive annotation rounds.

- **Discovered Positives** ($\Lambda$):  
  The subset of $\mathcal{D}^+$ identified and labelled as positive by the annotator.

- **Discovery Rate (Recall Rate)** ($d_r$):
    The discovery rate is the proportion of positive samples discovered:

$$
d_r = \frac{|\Lambda|}{|\mathcal{D}^+|}
$$

- **Labelling Efficiency (Precision)** ($L_e$):
    The ratio of discovered positive samples to the total labelled samples:

$$
L_e = \frac{|\Lambda|}{|\mathcal{L}|}
$$

- **Composite Metric:**
    A geometric mean of the above two metrics (precision and recall):

$$
\text{Composite Metric} = \sqrt{d_r \times L_e}
$$

## Embedding and Self-Supervised Learning

Before beginning our discussion on the search algorithm and its associated technicalities, it is essential first to convert our images into embeddings. These embeddings serve several crucial purposes:

1. **Dimensionality Reduction**  
   Images are inherently high-dimensional objects with a large number of degrees of freedom. Converting them into compact, dense representations allows us to address the curse of dimensionality while significantly reducing computational overhead. The resulting embeddings form a compressed yet information-rich representation of the original image space, making the subsequent search both efficient and accurate.

2. **Metric Definition in Embedding Space**  
   Each embedding corresponds to a point in $\mathbb{R}^n$, and the relative orientation and magnitude of these vectors capture intrinsic similarities between the original samples. This establishes a well-defined metric space, where distances between points correlate with semantic similarity in the original image domain. Thus, embeddings not only encode semantic information but also provide a geometrical interpretation of similarity, which is essential for constructing an effective search algorithm.

3. **Formation of Clusters**  
   The embeddings tend to form clusters, where samples with similar features are mapped close to each other. This clustering property implies that once a single point within a cluster is identified, one can infer the approximate nature of other points in that region. The concept of a metric and direction in the embedding space, therefore, allows for efficient exploration and retrieval of similar samples.

To convert our image to embeddings, we primarily use VICReg. The working of VICReg is discussed in the following section.
### VICReg

[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906) work by training on the surrogate task of generating embedding spaces that are invariant to specific augmentation spaces. This is achieved by using a Siamese-like setup, where two views of the same event are used, but both have been augmented to different degrees. The Euclidean distance in the embedding space between these two views is then minimised using gradient descent.

![](https://codimd.web.cern.ch/uploads/upload_b4cf6c75203f0b7c827e481f2518c359.png)
VICReg Training Procedure [Bardes et al.](https://arxiv.org/pdf/2105.04906)

Along with the MSE loss that acts across the two projection vectors Z, to ensure stability against collapse and the non-triviality of the embedding space, we have the Covariance and Variance regularisation terms. The covariance component prevents the embedding dimensions from capturing the same information. In contrast, the variance component prevents all embedding vectors in a batch from collapsing to a single point in the representation space.




# Intuition

## Embedding and Clustering
Given that the image dataset has now been converted into embeddings, we next discuss the nature of clustering within the embedding space. For a given seed sample $S$, our objective is to identify as many semantically similar samples as possible while minimizing manual annotation.

To elaborate further, consider an example scenario using the ImageNet dataset, denoted as $\mathbb{D}_{\text{Images}}$. For example, ImageNet-1K comprises 1000 distinct classes. We may therefore express the dataset as:  

$$
\mathbb{D}_{\text{ImageNet}} \equiv \bigcup\limits_{i=1}^{1000} C_i,
$$

where each class $C_i = \{ c_i^0, c_i^1, c_i^2, \dots, c_i^n \in \mathbb{D}_{\text{ImageNet}} \}$ corresponds to samples belonging to a single distinct class.In the said notation, the subscript refers to the class index, while the superscript refers to the sample index within that class. Now, given a few seed samples from $C_i$, our objective is to discover and label as many additional samples belonging to the same class as possible, while simultaneously minimizing the number of samples that do not belong to $C_i$. In essence, the task can be formulated as an optimization problem, wherein the goal is to maximize intra-class discovery under the constraint of minimal inter-class contamination.

Following the discussion so far, one might quickly realize that using a K-nearest neighbor search on the embedding space might be the solution to this optimization problem. Although the former is partially true, one cannot discover all the samples belonging to a single class solely by using a nearest neighbor type search algorithm. The argument can be further understood by examining the clusters more closely within the embedding space.


- **Sparse Clusters:**  
  Embedding algorithms such as VICReg are known for their ability to cluster semantically similar samples closely while simultaneously pushing dissimilar ones farther apart. In the ideal limit of perfect optimization, one would expect all samples belonging to the same class to collapse into a compact cluster, with inter-cluster centers maximally separated in the latent space. The compactness, or “tightness,” of a cluster can thus be defined by the relative distances between any two samples within the same class.  

  Conversely, under suboptimal training conditions, both intra-cluster compactness and inter-cluster separation deteriorate, eventually approaching complete randomness as the model becomes less ideal. Practically, most embedding algorithms operate in an intermediate regime, where clusters are not perfectly compact and their centers are not well-separated. This results in partial overlap and contamination between clusters due to the presence of negative samples (belonging to different classes) within the neighborhood of positive ones.  

  Such overlap directly impacts labeling efficiency, especially in nearest–neighbor–based search algorithms, where the number of samples within a given search radius increases exponentially with the radius. 



- **Multiple Disconnected Clusters:**  
The samples belonging to a desired class may often be distributed across multiple, potentially disjoint, clusters within the embedding space. Continuing with the ImageNet example consisting of classes ${ C_0, C_1, \dots, C_n} $, it is possible that the search objective is to identify all samples corresponding to a composite class such that $C^* = C_a \cup C_b \cup C_c$. A representative case would be the discovery of all images of mammals within the ImageNet dataset. In such scenarios, the nearest-neighbour search remains constrained to a local region of the embedding space and, consequently, fails to retrieve samples lying in spatially disconnected clusters. This inherent locality limits the overall effectiveness of the search process in capturing the complete semantic diversity of the target class.

- **Boundary Queries:**  

  Seed samples originating near the boundary of a cluster, within a latent space of dimension $N$, inherently require a large search radius due to the scaling of hypervolume with radius as $r^N$. As a result, moving away from the cluster centroid leads to an exponential decline in search efficiency.

- **Out-of-Cluster Queries:**  
  If a query sample lies outside any cluster, the nearest neighbour search mechanism completely fails. Furthermore, as the query moves farther from the cluster centre, the computational effort required to locate similar samples increases exponentially. 


To maximize the efficiency of the search, we need two components within the search algorithm. We need a nearest neighbour-like search to explore the proximity of a given positive sample to find more positive samples from the surroundings, which we call the **Spread Sampler (SpreadS)**. We need a stochastic sampler or a diversity-type search that tries to sample new points that are far away from this cluster. The Stochastic or Random Like Sampler, which we call **Cluster Search (ClusterS)**, attempts to find new Clusters that the Spread Sampler can explore in later iterations.
 
## Spread Search (SpreadS)
We present the following modified variant of the nearest neighbor search algorithm, which samples positive samples from the surroundings.
### Algorithm: Spread Search

**Input:**  
$S$ (set of queries), $a$ (number of nearest neighbours to be returned)

**Output:**  
$\delta$ (set of $a$-nearest neighbours for a given set of queries)

---

1. $\mathcal{D}' \gets \mathcal{D} \setminus (\Lambda \cup S)$  
2. Set $\mathcal{M}[i,j] = \text{MSE}(\mathcal{D}'[i], S[j])$  
3. **For each** $i$:  
   &emsp; $\mathcal{R}[i] = \min \left( \bigcup_j \mathcal{M}[i,j] \right)$  
4. $\mathcal{R}^* = \text{arg sort}_{\text{ascending}}(\mathcal{R})$  
5. $\delta \gets$ first $a$ samples of $\mathcal{R}^*$  
6. **Return** $\mathcal{D}'[\delta]$

---

The above implementation of the nearest neighbour sampling works slightly differently from a vanilla KNN algorithm. In the said algorithm, we have a hyperparameter $a$ which we refer to as the search radius. The reference points whose nearest neighbors are to be discovered are denoted by $S$ as the query sample. The nearest neighbour search is performed on the entire dataset, excluding the already discovered and labelled samples, as well as the reference samples. For each such sample in the embedding space, we compute its distance with respect to the reference/seed/query points and return the first $a$ samples that have the smallest pairwise distance to the reference points.

## Cluster Search (ClusterS)

The task of **ClusterS**, or the diversity sampler, is to identify new data points belonging to previously undiscovered clusters associated with the same class as the seed samples. To estimate the likelihood of a point belonging to the seed class, a Multi-Layer Perceptron (MLP) head is trained on the already identified positive samples. The objective of ClusterS is to discover samples that exhibit a high probability of being positive—belonging to the same class—while simultaneously residing sufficiently far away on the embedding space from the already discovered cluster centers. The *Cluster Search* method, therefore, attempts to optimise the following criteria to achieve the most effective and diverse sampling of positive points.

- **$M_1$**: The likelihood of being positive. This component of the optimisation criterion ensures that the diversity sampler returns only those samples that have a high probability of being a positive sample, and not random samples.

$$
M_1 = \sum\limits_{i=1}^n{f_c(x_i)} \text{where } f_c \text{ is the MLP score.}
$$

- **$M_2$**: The distance between each drawn sample. This component ensures that, in a single sampling step, the diversity sampler returns samples from as many different sources as possible.

$$
M_2 = \sum\limits_{i=1}^n\sum\limits_{j=1}^n{\| x_i-x_j \|}
$$
  
- **$M_3$**: The distance between each drawn sample with the positive labelled samples $\Lambda$. This component ensures that the diversity sampler indeed returns samples from clusters far away from those already discovered.

$$
M_3 = \sum\limits_{i=1}^N\sum\limits_{j=1}^n \| \Lambda_i-x_j \|
$$

In the above equations, $N$ is the number of positive samples we have currently labelled, and n$ is the number of Cluster Search samples to draw in this run.


The total cost that needs to be maximized is $$M = \alpha M_1 + \beta M_2 + \gamma M_3.$$

When $\alpha = 1, \beta = 0, \gamma = 0$, this becomes strictly a nearest neighbour search algorithm.
When $\alpha = 0, \beta,\gamma \neq 0$, this strictly becomes a stochastic sampler.

In this work, we explore how to solve this optimisation problem to define an optimal search algorithm.

### Grid Search (GridS)
In this section, we develop the GridS algorithm, which can serve as a suitable candidate for the diversity sampler or ClusterS. Later, we also discuss another parallel algorithm called RandS, which can be used as a diversity sampler too. Here, we limit our discussion to GridS.

The Cluster Search algorithm must maximize the three components of the cost function that we described in previous sections, coupled with the Spread search, to work optimally. Formulating analytic solutions to the above optimization is challenging. We try to find an approximate algorithmic solution to the above problem. 


We begin by dividing the entire embedding space into grids by segmenting the first few principal components of the embedding space. To address the second component of the equation, we strictly select only one sample from each grid. Now we have the first and the last component of the problem remaining. We scrutinize it and realise that the two are inversely linked to one another, given that the confidence of the MLP head is bound to decrease as the distance of the selected point from the already labelled samples on which the MLP head was trained increases. This makes it possible for us to rewrite the joint optimisation criterion, excluding the second component, as:

$$M = \zeta \sum\limits_{i=1}^n{f_c(x_i)} + (1-\zeta)* \sum\limits_{i=1}^N\sum\limits_{j=1}^n \| \Lambda_i-x_j \|$$

By fine-tuning and empirically finding the value of $\zeta$, we can exactly solve our optimization problem. In the original cost, we also had the parameter $\beta$ that controlled how different the two drawn samples are from one another. This parameter is realized through the number of partitions/grids into which the entire embedding space is broken.

#### GridS Algorithm

**Input:**  
$\mathcal{D}$ (dataset), $m$ (dimensionality of PCA), $N$ (number of segment per PCA dimension), $\zeta$ (hyperparameters)

**Output:**  
$S$ (sampled set)

---

1. **Step 1: Dimensionality Reduction Using PCA**  
   $\mathcal{D}_{red} \gets \text{PCA}(\mathcal{D}, m)$
   Let $\mathcal{D}_{red} = \{(x_{1i}, x_{2i},..., x_{mi}) \mid i = 1, 2, \dots, n\}$  
   i.e. $n$ points with coordinates $(x_1, x_2,..., x_m)$  

2. **Step 2: Partition the Dataset along Each Dimension**  
   **For each** $j \in \{1, 2, \dots, m\}$:  
   &emsp;a. $\mathcal{R} \gets \mathcal{D}_{red}$  
   &emsp;b. $\mathcal{R} \gets \text{argsort}_{i=1}^n \mathcal{R}[i][j]$  
   &emsp;c. **For each** $k \in \{1, 2, \dots, N\}$:  
   &emsp;&emsp;i. $D_{kj} \gets \mathcal{R}\Big[\dfrac{(k-1)n}{N} : \dfrac{kn}{N}\Big]$

3. **Step 3: Define Blocks**  
   $L \gets [\ ]$  
   **For each** $k \in \{1, 2, \dots, N^m\}$:  
   &emsp;a. $B_k = \Big\{\bigcap\limits_{l=1}^n \bigcap\limits_{j=1}^m \mathcal{D}_{k[l]j} \ \big| \ k \text{ is an m-string on } \{1, 2, \dots, N\}\Big\}$  
   &emsp;b. $f_c(v) \gets \zeta f_{MLP}(v) + (1 - \zeta)\big(\min\limits_{l \in \Lambda}\{\|l - v\|\}\big)$, where $\Lambda$ is the set of discovered positive samples.  
   &emsp;c. $f_c(B_k) = \max\{f_c(v) \mid v \in B_k\}$  
   &emsp;d. $L.\text{append}(B_k, f_c(B_k))$

4. $L = \text{sort}(L)$ w.r.t. $f_c(B)$ for $B \in L$

5. **Output:** $L[N^m - s : -1]$ # $s$ samples

---
### Random Search (RandS)

We return to our discussion of the complete three-component cost function. This time, we aim to develop a simpler algorithm by further relaxing the boundaries of our assumption. We return to the second component of our cost and claim that if we randomly draw samples from the dataset, then we can automatically ensure that all the samples are well spread apart, given a large number of samples are selected. This is equivalent to setting the value of $N^m$ (total number of grids) in the GridS algorithm, equivalent to the size of the entire dataset. To weakly satisfy the first and third components while still embedding the diversity condition, we simply set a threshold for the confidence value of the MLP and randomly select our sample from a set that has a higher confidence value compared to this threshold. We refer to this as the RandS variant of the Cluster Search Algorithm.

### RandS Algorithm

**Input:**  
$\mathcal{D}$ (dataset), $\alpha$(hyperparameter, controls diversity), $b$ (Number of samples to return)

**Output:**  
$S$ (sampled set)

---

1. **For each** $\mathcal{D}[i]$ in $\mathcal{D}$:  
   &emsp;a. $H_i \gets \text{Head}(\mathcal{D}[i])$  

2. $T \gets \text{Mean}(H) + \alpha \times \text{Stdev}(H)$  

3. $S \gets \{\mathcal{D}[i] \mid H_i > T\}$  

4. $S \gets \text{Select } b \text{ random samples from } S$  

5. **Return** $S$

---

## Algorithm: IRIS

Now, given that we have developed all the associated components of our search algorithm, we define the complete algorithm as follows:

**Input:**  
$\Lambda$ (Initial seeds/queries), epochs, $a$, $b$, $l$, $\alpha$

**Output:**  
$\Lambda^*$ (Updated values)

---

1. $\delta^0 \gets \Lambda$  
2. **For** $i = 1$ to epochs:  
   &emsp;a. $K \gets \text{SpreadS}(\delta^0, a)$  
   &emsp;b. $\delta \gets \text{ClusterS}(\mathcal{D}, b)$  
   &emsp;c. $\gamma \gets K \cup \delta$  
   &emsp;d. $\text{Oracle}(\gamma) \to (\gamma^+, \gamma^-)$  
   &emsp;e. $\Lambda \gets \Lambda \cup \gamma^+$  
   &emsp;f. **Train MLP head on** $(\Lambda \cup \gamma^-)$  
   &emsp;g. $i \gets i + 1$  
3. **For each** $\mathcal{D}[i]$ in $\mathcal{D}$:  
   &emsp;a. $H_i \gets \text{Head}(\mathcal{D}[i])$  
4. $T \gets \text{Mean}(H) + \alpha \times \text{Std}(H)$  
5. $z \gets \{\mathcal{D}[i] \mid H_i > T\}$  
6. $\text{Oracle}(z) \to (z^+, z^-)$  
7. $\Lambda^* \gets \Lambda \cup z^+$  
8. **Return** $\Lambda^*$  

In the above algorithm, we begin with a few seed samples that we denote by $\delta^0$. Given the iterative nature of our algorithm, we iterate through the entire dataset $\text{epoch}$ number of times. For each iteration, we sample the dataset through both the Spread Sampler and GridSampler. We accept $a$ samples (and denote it as $K$) from the grid sampler and $b$ samples (and denote it as $\delta$) from the Cluster Sampler. We combine the two sets as $\gamma \equiv K \cup \delta$. Now, given the sampling step is complete, we annotate the samples and separate them into $\gamma^+$ (samples that belong to the same class as $\delta^0$) and $\gamma^-$ (samples belonging to a different class). The newly discovered samples are now stored as $\Lambda \gets \Lambda \cup \gamma^+$. Next, we update the weights of our MLP confidence estimator by training it on the set $(\Lambda \cup \gamma^-)$. Subsequently, once all iterations are completed, we estimate the confidence of the entire dataset for the last time and select the highest-confidence samples (with confidence higher than a set threshold) from the dataset that are not yet annotated. We then annotate these samples and store the positive samples. 

---
## Experiments and Results

To assess the strengths and limitations of our algorithm, we evaluate it on the ImageNet dataset. We first compute embeddings for all images using a pretrained VICReg encoder. For each evaluation setting, we uniformly sample 50 classes at random and attempt to recover all samples belonging to those classes, given a fixed budget of initial query examples. We sweep the query budget over {1, 2, 5, 10, 50}. For each budget, we run five independent random seeds and repeat the entire procedure five times to address statistical noise.

### Spread Search (Nearest Neighbor Search) only

We have discussed in the previous sections how the nearest neighbor algorithm alone is not sufficient for discovering all the samples associated with our search query. Here, we present our findings after evaluating just the nearest neighbour search component over two different radii of searches.

| No. of seeds | LE (a=50) | D (a=50) | GM (a=50) | LE (a=200) | D (a=200) | GM (a=200) |
|-------------:|----------:|---------:|----------:|-----------:|----------:|-----------:|
|            1 |     0.15  |  0.0323  |   0.0696  |    0.1233  |   0.0369  |    0.06745 |
|            2 |     0.15  |  0.0331  |   0.0704  |    0.1461  |   0.0573  |    0.09149 |
|            5 |     0.24  |  0.0761  |   0.1351  |    0.1707  |   0.0723  |    0.11109 |
|           10 |     0.26  |  0.0661  |   0.1310  |    0.1786  |   0.1046  |    0.13668 |
|           50 |     0.41  |  0.1173  |   0.2193  |    0.2866  |   0.1176  |    0.18358 |

**Note:** LE = labelling efficiency; D = discovery rate; GM = geometric mean.


Spread search, being a nearest-neighbour exploration, has a relatively low discovery rate but a high labelling efficiency. We evaluate our algorithm for two different search \(a = 50\) and \(a = 200\). Given that there exist two different evaluation components to maximize, we try to combine them into a single composite metric by calculating the Geometric Mean between the two. We will compare this composite score across all different algorithms.

### Spread Search and Random Search

To supplement our SpreadS algorithm, we add a random search algorithm to it. We begin by just adding a simple random sampler to the search algorithm. If we refer back to our original cost function:

$$M = \alpha \sum\limits_{i=1}^n{f_c(x_i)}+ \beta \sum\limits_{i=1}^n\sum\limits_{j=1}^n{\| x_i-x_j \|} + \gamma \sum\limits_{i=1}^N\sum\limits_{j=1}^n \| \Lambda_i-x_j \|$$

By simply having a random sampler, we are effectively eliminating the first term, or setting the value of $\alpha \to 0$. Moreover, as we discussed in the previous sections, for a sufficiently large set of randomly sampled samples, $\beta \to 0$. This makes our optimisation reward that we want to maximise as:
$$M = \sum\limits_{i=1}^N\sum\limits_{j=1}^n \| \Lambda_i-x_j \|$$
This is automatically maximised again when we randomly select samples from the distribution, given that $\|\Lambda^+\| << \|D\|$ or the total number of positive samples discovered is much less in number compared to the total number of samples in the dataset

| **No. of seeds** | **LE**  | **D**   | **GM**  |
| ----------------:| ------- | ------- | ------- |
|                1 | 0.03408 | 0.85384 | 0.17059 |
|                2 | 0.03520 | 0.88153 | 0.17617 |
|                5 | 0.03381 | 0.88423 | 0.17290 |
|               10 | 0.03395 | 0.89153 | 0.17398 |
|               50 | 0.02829 | 0.93538 | 0.16268 |


Compared to a simple nearest neighbor search, we can already see that the composite score has increased for most cases, except for the case with 50 different seeds. This can be explained by the fact that the nearest neighbour search improves as it has more initial cluster centers to begin with, given that it gets access to almost all the clusters, making the need for a diversity sampler redundant.

Next, we attempt to evaluate the complete RandS algorithm, which was discussed in detail in the previous section.

| **No. of seeds** | **LE**   | **D**     | **GM**   |
|------------------:|:--------:|:---------:|:--------:|
| 1  | 0.1204 | 0.8861 | 0.3266 |
| 2  | 0.1205 | 0.8819 | 0.3260 |
| 5  | 0.1159 | 0.8659 | 0.3169 |
| 10 | 0.1141 | 0.8713 | 0.3153 |
| 50 | 0.1058 | 0.8969 | 0.3082 |

We can clearly see that this combination of algorithms outperforms all the previous ones. Even for a vast number of initial seed examples, the algorithm outperforms the simple nearest neighbour search. The reason behind this can be explained by the fact that with a large number of initial high-quality seed/query samples, the MLP head learns the correct parameters relatively quickly, making the confidence prediction more accurate.

### Spread Search and Grid Search
Finally, we present our findings on the complete algorithm using the GridS algorithm, which promises to maximise all three components of the cost function explicitly. The complete cost function for the optimisation problem is mainly dependent on two independent hyperparameters, which are:
 - **N**: The number of grids that we break the embedding space into. N is inversely proportional to the $\beta$ parameter in the original equation.
 - **$\zeta$**: The parameters $\alpha$ and $\gamma$ are dependent on each other, and the overall cost can be simplified into a two-component form, which is controlled by the parameter $\zeta$.

Determining the above two parameters accurately will ensure that our algorithm works perfectly. Given the excessive cost of running the experiment for multiple variants of $N$ and $\zeta$, we limit ourselves to selected values of $\zeta$ and a single value of $N$, which is 10 for our experiments. The performance for the different values of $\zeta$ is shown in the figure below:

![](https://codimd.web.cern.ch/uploads/upload_74873e8018dd853e9a53aec1ea16d402.png)

The above plot shows that the labelling efficiency monotonically increases as the value of $\zeta$ approaches 1. The discovery rate decreases, but the overall change in discovery rate is minimal compared to the change in labelling efficiency. To get a better understanding, we plot the variation of the composite score over the different values of $\zeta$

![](https://codimd.web.cern.ch/uploads/upload_16d5032ad56708ff38150f8483545fc9.png)
From the above plot, it is clear that the maximum performance is achieved by setting the value of $\ zeta$ to 1.

Now, we compare the performance of all the different algorithmic combinations that we have discussed so far.

![](https://codimd.web.cern.ch/uploads/upload_c2981ec956152d9e12376aa995dad4f5.png)
![](https://codimd.web.cern.ch/uploads/upload_3cd0d6f74c27d4128a68c93884fb9f74.png)

As shown in the above plots, the RandS component, when used as a diversity sampler, performs slightly better than grid-like sampling. This is evident from the fact that the GridS had the optimal value of the $\zeta$ parameter as one, making it more similar to the RandS sampler. Moreover, RandS automatically assumes the value of $N^m \to \|D\|$, making it absolutely possible to have a value of $N^m \in [0, \| D \| ]$ where the performance of the GridS algorithm exceeds that of the RandS. Moreover, the performance of the GridS diversity sampler closely resembles that of the RandS diversity sampler, indicating that our optimisation assumptions were indeed accurate. Additionally, the GridS is difficult to tune and more expensive to use compared to the RandS algorithm. In comparison to the nearest neighbor search and manual labeling annotation, we see a significant performance improvement.

# Conclusion and Future Direction
In this work, we developed the IRIS framework to alleviate the task of manual labelling by combining nearest neighbour type search with a suitable diversity sampler. We discussed two different types of diversity samplers, both of which demonstrate superior performance compared to the existing nearest neighbor type or manual labeling baselines. IRIS can be used to expedite the curation of new datasets and citizen science-type projects where manual labeling is essential. The values of the hyperparameters that have been extensively discussed throughout this work can be tuned using existing labels in a large, mostly unlabelled dataset. In our future work, we aim to explore how this approach plays out by using ImageNet 1k to tune the parameters and utilizing the same dataset to label the entirety of ImageNet 22k. Datasets, such as iNaturalist, can be used to determine the parameters of the search algorithm, which can later be used to curate datasets (or identify) of endangered/rare species. 

