# WDGTC- Weakly-Dependent Graph Tensor Completion

## Title
Tensor Completion for Weakly-dependent Data on Graph for Metro Passenger Flow Prediction

This paper is accepted in AAAI-2020, and available at [arxiv](https://arxiv.org/abs/1912.05693)

## Abstract
Low-rank tensor decomposition and completion have attracted significant interest from academia given the ubiquity
of tensor data. However, low-rank structure is a global property, which will not be fulfilled when the data presents complex
and weak dependencies given specific graph structures. One particular application that motivates this study is the spatiotemporal
data analysis. As shown in the preliminary study, weakly dependencies can worsen the low-rank tensor completion performance. In this paper, we propose a novel lowrank CANDECOMP / PARAFAC (CP) tensor decomposition and completion framework by introducing the L1-norm penalty and Graph Laplacian penalty to model the weakly dependency on graph. We further propose an efficient optimization algorithm based on the Block Coordinate Descent for efficient estimation. A case study based on the metro passenger flow data in Hong Kong is conducted to demonstrate an improved performance over the regular tensor completion methods.

## To Run the Code

----------------------------------------------
| Checklist | Comments |
|----------------|----------------------|
| Data | In the "processed" Folder |
| Code | In the "source" Folder |
| Some Results for Reference | In the "result" Folder|
| Pre-Requisite| pip install tensorly==0.4.3|

Code Structure:

.
+-- _CP_Version.py
|   +-- POI.py
|   +-- tdvm_cp_vector_form.py
|   |   +-- tensorly
|   |   +-- Telegram_chatbot.py
+-- _Parameters Searching, Networks and post-estimation analysis.py
|   +-- Telegram_chatbot.py
|   +-- pca_plus_hierarchical_6PC.py

.
├── _config.yml
├── _drafts
|   ├── begin-with-the-crazy-ideas.textile
|   └── on-simplicity-in-technology.markdown
├── _includes
|   ├── footer.html
|   └── header.html
├── _layouts
|   ├── default.html
|   └── post.html
├── _posts
|   ├── 2007-10-29-why-every-programmer-should-play-nethack.textile
|   └── 2009-04-26-barcamp-boston-4-roundup.textile
├── _data
|   └── members.yml
├── _site
└── index.html
