# Knowledge Base (sample data for RAG)

## What is Machine Learning?
Machine Learning (ML) is a branch of Artificial Intelligence where systems learn
patterns from data instead of being explicitly programmed with rules. The three
main paradigms are supervised learning, unsupervised learning, and reinforcement
learning.

## Supervised Learning
In supervised learning the model is trained on labelled data, meaning every input
example comes with the correct output. Common tasks are classification (predicting
a category, e.g. spam vs not-spam) and regression (predicting a number, e.g. house
price). Popular algorithms include Linear Regression, Logistic Regression, Decision
Trees, and Neural Networks.

## Unsupervised Learning
Unsupervised learning finds structure in data that has NO labels. The two classic
tasks are clustering (grouping similar points together, e.g. K-Means) and
dimensionality reduction (compressing features while keeping information, e.g. PCA).

## Reinforcement Learning
Reinforcement Learning (RL) trains an agent to take actions in an environment to
maximise a cumulative reward. The agent learns by trial and error using feedback
signals. Famous examples are AlphaGo and robots learning to walk.

## What is a Neural Network?
A neural network is a model made of layers of connected "neurons". Each neuron
computes a weighted sum of its inputs, adds a bias, and passes the result through a
non-linear activation function such as ReLU or Sigmoid. Deep learning refers to
neural networks with many hidden layers.

## What is a Transformer?
The Transformer is a neural network architecture introduced in the 2017 paper
"Attention Is All You Need". It relies on the self-attention mechanism, which lets
every token look at every other token in the sequence. Transformers are the
foundation of modern Large Language Models (LLMs) like GPT and Llama.

## What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that gives a language model
access to an external knowledge base. Instead of relying only on what it memorised
during training, the model first RETRIEVES relevant documents and then GENERATES an
answer grounded in those documents. This reduces hallucinations and lets the model
use up-to-date, private, or domain-specific information.

## What are Embeddings?
An embedding is a list of numbers (a vector) that represents the meaning of a piece
of text. Texts with similar meaning have vectors that are close together. Embeddings
are what let a vector database find "semantically similar" documents for a query.

## What is Groq?
Groq is a company that builds specialised hardware (the LPU, Language Processing
Unit) to run LLMs extremely fast. The Groq API offers a free tier and is compatible
with popular open models such as Llama 3, making it great for learning and building.
