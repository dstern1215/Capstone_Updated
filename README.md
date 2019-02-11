
# Project: Classifying Quora Posts
_Author: Daniel Stern_

## Problem Statement

The internet can be a fantastic resource for information, or for connecting with people across the world. Its relatively anonymous, inpersonal nature, however can also lead to a quite toxic environment. While some are drawn to the internet to exchange ideas or create new connections, there are also those who also use the medium to indulge their anti-social impulses. 

Racism, misogyny and content intended to provoke or "trigger" those who are different than them is unfortunately widespread, and one of the more unfortunate aspects of the internet.

The website Quora exemplifies both the positive and negative aspects of the internet, describing itself as a "platform to ask questions and connect with people who contribute unique insights and quality answers." Although Quora often features excellent questions and responses by subject matter experts, there are "trolls" who seek to take advantage of potential creduilty and create content that is at best "insincere", and at worst hateful and disgusting. 

For a platform like Quora (or any other discussion site), it is important to cultivate a safe trusted environment, both for the moral sake of the well-being of its users, and also to maintain and grow their base of active users.

Using a dataset provided from Kaggle.com, this report will take an extensive look into the semantic differences between "sincere" and "insincere" posts and establish a model that can either flag or completely delete these negative types of messages.

Since discussion forums (and especially comment sections) can become indundiated with toxic/insincere language, this is an especially important use case for websites that receive such user-generated-content, and who want to maintain an engaging yet safe environment 


## Executive Summary

The enclosed technical report examines a dataset from Kaggle.com, with around 1.5 million questions, categorized as "Sincere" or "Insincere". The balance of post labels was around 94% "Sincere" posts and around 6% "Insincere" posts. Although this "class inbalance" makes the actual modeling problem challenging (since there is less "insincere" data for the model to train on).

Although accuracy score is often used as the primary metric for measuring classification algorithms, the class imbalance means that this metric is appropriate for this use case, as a classifier could achieve a 94% accuracy score by simply choosing the majority class every time. Based on the inital performance of baseline models, it makes sense for lower probability thresholds to be used to deem posts as being part of the positive class, due to the fact that it would minimize false positives (which seems in line with Quora's goal).

EDA revealed some definite trends in the language used between the two classes, with the "insincere" class routinely using language that was steeped in racism, misogyny and an attempt to provoke by bring up taboo subjects like incest. Techniques such as word embeddings, T-SNE and Latent Dirichlet Allocation were used to extract potential meaning from the words in the corpus as opposed to the more simple Count or Term Frequency. However, these techniques were more beneficial for EDA purposes rather than modeling, due to the mature of the data.

A wide variety of models were fit on the data, including popular "white box" models such as logistic regression, ensemble methods like Random Forest and Extra Trees, and deep learning methods like Neural Networks. Ultimately, it is a very exciting time in the NLP, with a lot of new emerging tools, and it will ultimately require a deeper look into the effectiveness of wordembeddings or topic modeling to get a full grasp of the tools available to solve the problem at hand.

### Contents:
- [Notebook 1: EDA Pre-Cleaning](./code/Notebook01_EDAPreCleaning.ipynb)
- [Notebook 2 Pre-Processing Text](./code/Notebook02_PreProcessingSteps.ipynb)
- [Notebook 3 EDA After Text Cleaning](./code/Notebook03_PostPreprocessingEDA.ipynb)
- [Notebook 4 Training Word2Vec Model](./code/Notebook04_TrainingWord2VecModel.ipynb)
- [Notebook 5 TSNE and Doc2Vec Visualization](./code/Notebook05_TSNEDoc2Vec_Visualization.ipynb)
- [Notebook 6 LDA Insincere Topic Visualization](./code/Notebook06_LDATopicModelingVisualizations.ipynb)
- [Notebook 7 Training a Doc2Vec Model](./code/Notebook07_TrainingADoc2VecModel.ipynbb)
- [Notebook 8 Fitting a Neural Network](./code/Notebook08_FittingANeuralNetwork.ipynb)
- [Notebook 9 Classification Model Comparison](./code/Notebook09ClassificationModelComparison.ipynb)


## Conclusions & Recommendations

After analyzing the data, while there do appear to be clear differences in the language used between the "sincere" and "insincere" posts, the class imbalance problem makes it a challenging use case. 

EDA revealed a strong prevalance of "hate speech" within the most used words for the "insincere" category, with many posts within that class demonstrating antagonistic behavior. There were, however, also posts that were more sarcastic rather than explicitly mean, which the models had a tough time picking up. 

There are a lot of exciting new tools to use for NLP (such as LDA, and Word2Vec models), and to do them justice would require a much deeper dive. Many of them have a lot of potential (and the enclosed report provides examples of how to fit models with on a text corpus), but their effectiveness remains inconclusive. While some of these tools have power capabilities for document similarity, unfortunately this use case's severe class inbalance likely means that these tools are more effective for EDA than for modeling, in this specific case. Also, the stochastic nature of the Doc2Vec tool (especially its unseen sentence similarity mechanism) likely makes it too unreliable a model for this use case as well. I was however able to fit a deterministic neural network using pytorch, which yielded promising results. The computationally expensive nature of the model prevented too deep a dive, though its predicted probabilities were compared to the those of the logistic regression model.

Although this report does not thoroughly explore all of the possible combinations of models and techniques, it does seem like an ensemble or "blended" approach could work in this situation. When certain posts existing in the "grey" area of relatively low probability (.1 to .25) were analyzed, it seemed like the neural network was classifying some posts that the logistic regression missed, and vice versa.

For the time being, I would recommend a production model of a logistic regression and TFiDF, based on initial solid/fast performance, easy interpretability, and a relatively linear nature to its predictions. The thresholds could also be adjusted to account for either more false positives or false negatives, with the latter making more sense in this context. I would also continue to research the effeciveness of both Neural Networks and Word Embeddings, to see there could be some useful combinations between one and a simpler model.
