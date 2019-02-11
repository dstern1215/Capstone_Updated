
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

Although not a completely exhaustive list, a variety of different classifiers and word representations were used, in an attempt to find an ideal combination. An ensembling method, combing a Logistic Regression, Extra Trees Classifier and Naive Bayes showed the most promise, although at present the model has not been sufficiently hyperparametized due to time and compute issues. Traditional latent word vectorization methods like CountVectorizer and TFIDF were used to format the text, as well as both pre-trained word embeddings, and ones trained on the corpus being modeled. 

A workflow has been set up to continute to test and bench different combinations of models, word vectors and hyperperameters, and this repo will continue to be updated in the future.

### Contents:
- [Notebook 1: EDA Pre-Cleaning](./code/Notebook01_EDAPreCleaning.ipynb)
- [Notebook 2 Pre-Processing Text](./code/Notebook02_PreProcessingSteps.ipynb)
- [Notebook 3 EDA After Text Cleaning](./code/Notebook03_PostPreprocessingEDA.ipynb)
- [Notebook 4 LDA Insincere Topic Visualization](./code/Notebook04_LDATopicModelingVisualizations.ipynb)
- [Notebook 5 Classifier Training Process](./code/Notebook05_ClassifierProcess.ipynb)
- [Notebook 6 Other Classifiers With Countvectorizer](./code/Notebook06_OtherClassifiersWithCountVectorizer.ipynb)
- [Notebook 7 Classifiers With TFIDF](./code/Notebook07_ClassifersWithTFIDF.ipynb	)
- [Notebook 8 Classifiers On Trained Word Embedding (TFIDF)](./code/Notebook08_ClassifersWithTFIDFEmbeddingsFromOurCorpus.ipynb)
- [Notebook 9 Classifiers On Trained Word Embedding (Mean)](./code/Notebook09_ClassifersWithMeanEmbeddingsFromOurCorpus.ipynb)
- [Notebook 10 Classifiers On Glove (Mean)](./code/Notebook10_ClassifersWithMeanEmbeddingsFromGlove.ipynb)
- [Notebook 11 Classifiers On Glove (Mean)](./code/Notebook11_ClassifersWithTFIDFEmbeddingsFromGlove.ipynb)


## Conclusions & Recommendations

After analyzing the data, while there do appear to be clear differences in the language used between the "sincere" and "insincere" posts, the class imbalance problem makes it a challenging use case. 

EDA revealed a strong prevalance of "hate speech" within the most used words for the "insincere" category, with many posts within that class demonstrating antagonistic behavior. There were, however, also posts that were more sarcastic rather than explicitly mean, which the models had a tough time picking up. 

There are a lot of exciting new tools to use for NLP (such as LDA, pre-trained embeddings and Word2Vec models, among others), and to do them justice would require a much deeper dive. A scalable workflow has been set up that will allow more models to be tested against more combinations of word embeddings, and it is a great opportunity to benchmark classification models against each other in a highly imbalanced setting.

Although this report does not thoroughly explore all of the possible combinations of models and techniques, an ensemble voting classifier was created, which showed promising results. Further hyperparameter tuning is needed, however, and the SVC model should be attempted, as it has a good reputation for problems like this,

For the time being, I would recommend a production model of an ensemble classifier. For the time being, an ensemble of Logistic Regression, Extra Trees and Naive Bayes (bernouli) seems like a good baseline, but further iteration and experimentation is likely to deliver improvements on this initial performance.
