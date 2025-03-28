# CS 886: Graph Neural Networks

## Logistics
+ **Instructor:** [Kimon Fountoulakis](https://opallab.ca/team/)
+ **Seminar Time:** Tu/Th 1pm to 2:20pm
+ **Office hours:** Monday 1pm to 2pm.

## Overview

Learning from multi-modal datasets is currently one of the most prominent topics in artificial intelligence. The reason behind this trend is that many applications, such as recommendation systems and fraud detection, require the combination of different types of data. In addition, it is often the case that data exhibit relations that need to be captured for downstream applications. In this course, we are interested in multi-modal data that combine a graph—i.e., a set of nodes and edges—with attributes for each node and/or edge. The attributes of the nodes and edges capture information about the nodes and edges themselves, while the edges between the nodes capture relations among them. Capturing relations is particularly helpful for applications where we are trying to make predictions for nodes given neighborhood data.

One of the most prominent and principled ways of handling such multi-modal data for downstream tasks such as node classification is graph neural networks. Graph neural network models can mix hand-crafted or automatically learned attributes of the nodes while taking into account relational information among the nodes. Therefore, the output vector representation of the graph neural network contains global and local information about the nodes. This contrasts with neural networks that only learn from the attributes of entities.

This seminar-based course will cover seminal work in the space of graph neural networks. Below I provide the topics and architectures which we will study during the course. 

**Topics** 
1. Generalization performance of graph neural networks
2. Expressive power of graph neural networks
4. Large language models and graphs
5. Neural algorithmic reasoning
6. Generative graph neural networks
7. Self-supervised learning in graphs
9. Oversmoothing
10. Scalability

**Architectures**:
1. Spectral and spatial convolutional graph neural networks
2. Graph attention networks
3. Invariant and equivariant graph neural networks
4. General message passing graph neural networks
5. Higher-order graph neural networks
6. Graph neural networks for heterogeneous graphs

We will focus on both practical and theoretical aspects of machine learning on graphs. Practical aspects include, scalability and performance on real data. Examples of theoretical questions include: what does convolution do to the input data? Does convolution improve generalization compared to not using a graph? How do multiple convolutions change the data and how do they affect generalization?

**Course structure:** The seminar is based on weekly student presentations, discussions, a midterm and a final project. 

## (Tentative) Schedule
The schedule below is subject to change:
| Week | Date | Topic | Readings | Slides |
|:-----|:-----|:------|:------------|:-----|
| 1 | 1/7 | Introduction, problems and applications (Kimon lecturing) | - [Geometric Deep Learning](https://arxiv.org/abs/2104.13478) (Chapter 1) <br/> - [Geometric foundations of Deep Learning](https://towardsdatascience.com/towards-geometric-deep-learning-iv-chemical-precursors-of-gnns-11273d74125) <br/>  - [Towards Geometric Deep Learning I: On the Shoulders of Giants](https://towardsdatascience.com/towards-geometric-deep-learning-i-on-the-shoulders-of-giants-726c205860f5) <br/> - [Towards Geometric Deep Learning II: The Perceptron Affair](https://towardsdatascience.com/towards-geometric-deep-learning-ii-the-perceptron-affair-fafa61b5c40a) <br/> - [Towards Geometric Deep Learning III: First Geometric Architectures](https://towardsdatascience.com/towards-geometric-deep-learning-iii-first-geometric-architectures-d1578f4ade1f) <br/> - [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) <br/> - [Intro to graph neural networks (ML Tech Talks)](https://www.youtube.com/watch?v=8owQBFAHw7E) <br/> - [Foundations of Graph Neural Networks](https://www.youtube.com/watch?v=uF53xsT7mjc) <img width=2000/> | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/lecture_1.key)  |
| 1 | 1/9 | Spatial graph convolution and its theoretical performance on simple random data (Kimon lecturing) | - [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) <br/> - [Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization](https://proceedings.mlr.press/v139/baranwal21a.html) <br/> - [Code for reproducing the experiments](https://github.com/opallab/Graph-Convolution-for-Semi-Supervised-Classification-Improved-Linear-Separability-and-OoD-Gen.) <br/> - [Effects of Graph Convolutions in Multi-layer Networks](https://openreview.net/pdf?id=P-73JPgRs0R) <br/> - [Code for reproducing the experiments](https://github.com/opallab/Effects-of-Graph-Convs-in-Deep-Nets) <br/> - [Theory of Graph Neural Networks: Representation and Learning](https://arxiv.org/abs/2204.07697) <br/> - [PyTorch code for GCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv) <br/> - [Example code](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/lecture_2.key), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/lecture_3.key)  |
| 2 | 1/14 | Graph Attention Network, Graph Attention Retrospective, and Optimality of Message Passing (Kimon lecturing) | - [Graph Attention Networks](https://arxiv.org/abs/1710.10903) <br/> - [PyTorch Code](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv) <br/> - [Graph Attention Retrospective](https://jmlr.org/papers/v24/22-125.html) <br/> - [Code for reproducing the experiments](https://github.com/opallab/Graph-Attention-Retrospective/) <br/> - [Video lecture](https://youtu.be/duWVNO8_sDM) <br/> - [Theory of Graph Neural Networks: Representation and Learning](https://arxiv.org/abs/2204.07697) <br/> - [Optimality of Message-Passing Architectures for Sparse Graphs](https://openreview.net/forum?id=d1knqWjmNt)| [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/Lecture_4.key), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/Lecture_5.key) | 
| 2 | 1/16 | A popular spectral graph convolution model, message passing, symmetries and reasoning (Kimon lecturing) | - [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) <br/> - [What Can Neural Networks Reason About?](https://arxiv.org/abs/1905.13211)  | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/cnns_on_graphs.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/mp_symmetries_alignment.key) |
| 3 | 1/21 | Expressive power, Part 1| - [The Expressive Power of Graph Neural Networks (Chapter 5.3)](https://graph-neural-networks.github.io/static/file/chapter5.pdf) <br/> - [The Expressive Power of Graph Neural Networks (Chapter 5.4, up to 5.4.2.3), included)](https://graph-neural-networks.github.io/static/file/chapter5.pdf) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/expressive_power_gnns.key) |
| 3 | 1/23 | Expressive power, Part 2 | - [The Expressive Power of Graph Neural Networks (Chapter 5.4.2.4, up to 5.4.3.2), included)](https://graph-neural-networks.github.io/static/file/chapter5.pdf)  <br/> - [What graph neural networks cannot learn: depth vs width](https://openreview.net/pdf?id=B1l2bp4YwS)| [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/The_Expressive_Power_of_Graph_Neural_Networks.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/what_gnns_cannot_learn.pptm) | 
| 4 | 1/28 | Higher-order graph neural networks | - [k-hop Graph Neural Networks](https://arxiv.org/abs/1907.06051) <br/> - [Multi-Hop Attention Graph Neural Network](https://arxiv.org/abs/2009.14332) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/k-hop_gnns.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/MAGNA.pdf) |
| 4 | 1/30 | Higher-order graph neural networks and their expressive power | - [Provably Powerful Graph Networks](https://arxiv.org/abs/1905.11136) <br/> - [How Powerful are K-hop Message Passing Graph Neural Networks](https://arxiv.org/pdf/2205.13328.pdf)| [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/Lecture_provably_powerful_gnns.key), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/HowpowerfularekhopMPGNNs.pdf) |
| 5 | 2/4 | Invariant and Equivariant Graph Neural Networks, Part 1 | - [An Introduction To Invariant Graph Networks, Part 1](http://irregulardeep.org/An-introduction-to-Invariant-Graph-Networks-(1-2)/) and [An Introduction To Invariant Graph Networks, Part 2](https://irregulardeep.org/How-expressive-are-Invariant-Graph-Networks-(2-2)/) <br/> - [Invariant and Equivariant Graph Networks](https://arxiv.org/pdf/1812.09902.pdf) |[Slides](https://github.com/opallab/cs886-winter-2025/blob/main/invariant_graph_networks.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/INVARIANT-AND-EQUIVARIANT-GRAPH-NETWORKS.pdf) |
| 5 | 2/6 | Invariant and Equivariant Graph Neural Networks, Part 2 | - [Building powerful and equivariant graph neural networks with structural message-passing](https://proceedings.neurips.cc/paper/2020/file/a32d7eeaae19821fd9ce317f3ce952a7-Paper.pdf) <br/> - [Universal Invariant and Equivariant Graph Neural Networks](https://papers.nips.cc/paper/2019/file/ea9268cb43f55d1d12380fb6ea5bf572-Paper.pdf)  | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/building_powerful_and_equivariant.pptx), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/UniversalIn_Equi_varientGNN.pdf) |
| 6 | 2/11 | GNNs for heterogeneous graphs | - [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) <br/> - [MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding](https://arxiv.org/abs/2002.01680) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/R-GCN.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/MAGNN.pdf) |
| 6 | 2/13 | Oversmoothing Part 1 | - [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751) <br/> - [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/2007.02133) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/deepGCNs.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/simple_and_deep_gnns.pptx) |
| 7 | 2/18 | Reading Week |  | | |
| 7 | 2/20 | Reading Week |  |  |
| 8 | 2/25 | Oversmoothing Part 2 | - [Not too little, not too much: a theoretical analysis of graph (over)smoothing](https://arxiv.org/abs/2205.12156) <br/> - [Analysis of Corrected Graph Convolutions](https://arxiv.org/abs/2405.13987) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/not_too_little_not_too_much_2025.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/analysis_of_graph_corrected.pdf)  |
| 8 | 2/27 | Scalable GNNs, Part 1 | - [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf) <br/> - [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/cluster_gcn.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/sign.pdf) |
| 9 | 3/4 | Scalable GNNs, Part 2 | - [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) <br/> - [Training Graph Neural Networks with 1000 Layers](https://arxiv.org/pdf/2106.07476.pdf) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/graphsaint.pptx), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/1k_layers.pdf) | 
| 9 | 3/6 | Self-supervised learning in graphs | - [Graph Self-Supervised Learning: A Survey](https://arxiv.org/abs/2103.00111) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/self_supervised_learning_part_1.key) |
| 10 | 3/11 | Link prediction | - [Link Prediction Based on Graph Neural Networks](https://arxiv.org/pdf/1802.09691.pdf) <br/> - [Line Graph Neural Networks for Link Prediction](https://arxiv.org/pdf/2010.10046.pdf) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/line_graph.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/link_prediction_gnn.pptx) |
| 10 | 3/13 | GNNs for combinatorial optimization, Part 1 | - [Erdos Goes Neural: an Unsupervised Learning Framework for Combinatorial Optimization on Graphs](https://proceedings.neurips.cc/paper/2020/file/49f85a9ed090b20c8bed85a5923c669f-Paper.pdf) <br/> - [Simulation of Graph Algorithms with Looped Transformers](https://arxiv.org/abs/2402.01107) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/simulation_for_graphs.pdf) |
| 11 | 3/18 | GNNs for combinatorial optimization, Part 2 | - [Attention, Learn to Solve Routing Problems!](https://arxiv.org/pdf/1803.08475.pdf) <br/> - [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/file/d14c2267d848abeb81fd590f371d39bd-Paper.pdf) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/attention_routing.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/exact_combinatorial_optimization.key) |
| 11 | 3/20 | GNNs + LLMs, Part 1 | - [Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560) <br/> - [Can Language Models Solve Graph Problems in Natural Language?](https://arxiv.org/abs/2305.10037) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/talk_like_a_graph.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/CanLanguageModelsSolveGraphProblems.pdf) |
| 12 | 3/25 | GNNs + LLMs, Part 2 | - [Graph Neural Prompting with Large Language Models](https://arxiv.org/abs/2309.15427) <br/> - [G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering](https://arxiv.org/abs/2402.07630) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/graph_neural_network_prompting_with_large_language_models.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/g_retriever.pdf) |
| 12 | 3/27 | Algorithmic Reasoning, Part 1 | - [Neural Algorithmic Reasoning](https://arxiv.org/abs/2105.02761) and [A Generalist Neural Algorithmic Learner](https://arxiv.org/abs/2209.11142) <br/> - [Deep Equilibrium Algorithmic Reasoning](https://arxiv.org/abs/2410.15059) | [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/NeuralAlgorithmicReasoningAGeneralistNeuralAlgorithmicLearner.pdf), [Slides](https://github.com/opallab/cs886-winter-2025/blob/main/deep_equilibrium_algorithmic_reasoning.pptx) |
| 13 | 4/1 | Algorithmic Reasoning, Part 2 | - [Pointer Graph Networks](https://arxiv.org/abs/2006.06380) <br/> - [Relational Attention: Generalizing Transformers for Graph-Structured Tasks](https://arxiv.org/abs/2210.05062) | |
| 13 | 4/3 | GNNs + LLMs, Part 3 | - [A Survey of Large Language Models for Graphs](https://arxiv.org/abs/2405.08011) | |





## Readings

+ [Geometric Deep Learning](https://geometricdeeplearning.com), Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković, 2021
+ [Theory of Graph Neural Networks: Representation and Learning](https://arxiv.org/abs/2204.07697), Stefanie Jegelka, 2022
+ [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/), William L. Hamilton, 2020
+ [Graph Neural Networks](https://graph-neural-networks.github.io), Lingfei Wu, Peng Cui, Jian Pei, Liang Zhao, (2022)

## Other courses online related to machine learning on graphs

+ [Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/), Jure Leskovec, Stanford
+ [Graph Representation Learning](https://cs.mcgill.ca/~wlh/comp766/), William L. Hamilton, McGill
+ [Introduction to Graph Neural Networks](https://www.youtube.com/watch?v=Iiv9R6BjxHM), Xavier Bresson, Nanyang Techinical University and NYU
+ [Recent Developments in Graph Network Architectures](https://www.youtube.com/watch?v=M60huxIvKbE), Xavier Bresson, Nanyang Techinical University
+ [Benchmarking GNNs](https://www.youtube.com/watch?v=tuChBSo8_eg), Xavier Bresson, Nanyang Techinical University
+ [Foundations of Graph Neural Networks](https://www.youtube.com/watch?v=uF53xsT7mjc), Petar Veličković, DeepMind
+ [Geometric Deep Learning Course](https://geometricdeeplearning.com/lectures/)
+ [Machine Learning for the Working Mathematician: Geometric Deep Learning](https://www.youtube.com/watch?v=7pRIjJ_u2_c), Geordie Williamson, The University of Syndney
+ [Advanced lectures on community detection](https://indico.ictp.it/event/9797/other-view?view=ictptimetable), Laurent Massoulie, INRIA Paris

## Online reading seminars

+ [LoGaG: Learning on Graphs and Geometry Reading Group](https://hannes-stark.com/logag-reading-group)

## Code

+ [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
+ [Deep Graph Library](https://www.dgl.ai)

## Competitions

+ [Open Graph Benchmark](https://ogb.stanford.edu/docs/leader_overview/)
+ [CLRS Algorithmic Reasoning Benchmark](https://arxiv.org/abs/2205.15659)
+ [The CLRS-Text Algorithmic Reasoning Language Benchmark](https://arxiv.org/abs/2406.04229)

## Datasets

+ [PyTorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
+ [Open Graph Benchmark](https://ogb.stanford.edu)
+ [HyperGraphs](https://www.cs.cornell.edu/~arb/data/)
+ [TUDatasets](https://chrsmrrs.github.io/datasets/)
+ [Non Homophily Benchmarks](https://github.com/CUAI/Non-Homophily-Benchmarks)
+ [Graph Learning Benchmarks](https://graph-learning-benchmarks.github.io/glb2022)
+ [Hetionet](https://het.io)
+ [Heterogeneous graph benchmarks](https://www.biendata.xyz/hgb/)
+ [Long Range Graph Benchmark](https://towardsdatascience.com/lrgb-long-range-graph-benchmark-909a6818f02c)
+ [IGB-Datasets](https://github.com/IllinoisGraphBenchmark/IGB-Datasets)


## Workload Breakdown
+ Class Participation: 15%
+ Midterm Project: 20%
+ Presentations: 25%
+ Final Project: 40%

## Midterm Project

This is a 3-page paper, along with (if relevant) the source code of your project, including instructions on how to run it. 
You may use your midterm project as a foundation for your final project, which will be 6 pages. Please see the next section below for details.

Options for the midterm project:

+ Option A (Empirical evaluation):
Pick a problem that interests you.
Implement and experiment with several graph neural network methods to tackle this problem.
+ Option B (Method design):
Identify a problem for which there are no satisfying approaches.
Develop a new graph neural network architecture to tackle this problem.
Analyze theoretically and/or empirically the performance of your technique.
+ Option C (Theoretical analysis):
Identify a problem or a graph neural network architecture for which theoretical performance (e.g., complexity, performance on random data, expressivity) is not well understood. Analyze the properties of this problem or technique.

Information about the project template and the source code is given below.
+ Project Paper: The project papers will be 3 pages. You can have extra pages for the references and the appendix.
They will be written in the two-column [ICML](https://icml.cc) format, using the ICML template which you can find in the corresponding website.
+ Project Source Code: Please put your source code into github and include a link in your project writeup. 
On the github page, please document exactly how to run your source code.

## Final Project
There is one main deliverable of your project, a 6-page paper and (if relevant) the source code of your project 
with instructions to run your code. Note that you are allowed to use the midterm project (3 pages) as a foundation
for this project.

Options for the final project:

+ Option A (Empirical evaluation):
Pick a problem that interests you.
Implement and experiment with several graph neural network methods to tackle this problem.
+ Option B (Method design):
Identify a problem for which there are no satisfying approaches.
Develop a new graph neural network architecture to tackle this problem.
Analyze theoretically and/or empirically the performance of your technique.
+ Option C (Theoretical analysis):
Identify a problem or a graph neural network architecture for which theoretical performance (e.g., complexity, performance on random data, expressivity) is not well understood. Analyze the properties of this problem or technique.

Information about the project template and the source code is given below.
+ Project Paper: The project papers will be 6 pages. You can have extra pages for the references and the appendix.
They will be written in the two-column [ICML](https://icml.cc) format, using the ICML template which you can find in the corresponding website.
+ Project Source Code: Please put your source code into github and include a link in your project writeup. 
On the github page, please document exactly how to run your source code.

## Publish Your Project
Although not required for the course, keep in mind that I am more than happy to help you publish your final project.
For example, in CS886 2024, Robert Wang published his final project at [NeurIPS 2024](https://nips.cc/virtual/2024/poster/95519).

## Presentations
Each student will be doing 2 presentations (estimated number, based on previous years) in the term. Each presentation will be about 40 to 50 minutes long plus questions.
Here are the important points summarizing what you have to do for your presentations.

+ You must present with slides. The content in your slides should be your own but you can use others' materials, e.g., 
figures from the paper we are reading, when necessary and by crediting your source on your slide.
+ Please have a separate slide, or set of slides, for each of the 4 questions below:  
  + What is the problem?
  + Why is it important?
  + Why don't previous methods work on that problem?
  + What is the solution to the problem the authors propose?
  + What interesting research questions does the paper raise?
+ It is very helpful to demonstrate the ideas in the paper through examples. So try to have examples in your presentation.

## University of Waterloo Academic Integrity Policy
The University of Waterloo Senate Undergraduate Council has also approved the following message outlining University of Waterloo policy on academic integrity and associated policies.

## Academic Integrity
In order to maintain a culture of academic integrity, members of the University of Waterloo community are expected to promote honesty, trust, fairness, respect and responsibility. Check the Office of Academic Integrity's [website](https://uwaterloo.ca/academic-integrity) for more information. All members of the UW community are expected to hold to the highest standard of academic integrity in their studies, teaching, and research. This site explains why academic integrity is important and how students can avoid academic misconduct. It also identifies resources available on campus for students and faculty to help achieve academic integrity in, and our, of the classroom.

## Grievance
A student who believes that a decision affecting some aspect of his/her university life has been unfair or unreasonable may have grounds for initiating a grievance. Read [Policy 70 - Student Petitions and Grievances, Section 4](https://uwaterloo.ca/secretariat/policies-procedures-guidelines/policy-70). When in doubt please be certain to contact the department's administrative assistant who will provide further assistance.

## Discipline
A student is expected to know what constitutes academic integrity, to avoid committing academic offenses, and to take responsibility for his/her actions. A student who is unsure whether an action constitutes an offense, or who needs help in learning how to avoid offenses (e.g., plagiarism, cheating) or about “rules” for group work/collaboration should seek guidance from the course professor, academic advisor, or the Undergraduate Associate Dean. For information on categories of offenses and types of penalties, students should refer to [Policy 71-Student Discipline](https://uwaterloo.ca/secretariat/policies-procedures-guidelines/policy-71). For typical penalties check [Guidelines for the Assessment of Penalties](https://uwaterloo.ca/secretariat/guidelines/guidelines-assessment-penalties).

## Avoiding Academic Offenses
Most students are unaware of the line between acceptable and unacceptable academic behaviour, especially when discussing assignments with classmates and using the work of other students. For information on commonly misunderstood academic offenses and how to avoid them, students should refer to the Faculty of Mathematics Cheating and Student Academic Discipline Policy.

## Appeals
A decision made or a penalty imposed under Policy 70, Student Petitions and Grievances (other than a petition) or Policy 71, Student Discipline may be appealed if there is a ground. A student who believes he/she has a ground for an appeal should refer to [Policy 72 - Student Appeals](https://uwaterloo.ca/secretariat/policies-procedures-guidelines/policy-72).

## Note for students with disabilities
The AccessAbility Services Office (AAS), located in Needles Hall, Room 1401, collaborates with all academic departments to arrange appropriate accommodations for students with disabilities without compromising the academic integrity of the curriculum. If you require academic accommodations to lessen the impact of your disability, please register with the AAS at the beginning of each academic term.
