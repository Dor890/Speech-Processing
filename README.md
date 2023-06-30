# Speech Recognition and Processing
Written by: Dor Messica, Roi Zehavi.

This repository contains a collection of exercises and a final project focused on speech processing and recognition. The exercises cover a range of topics, including Fourier transforms, MFCC (Mel-frequency cepstral coefficients), Mel spectrograms, audio and music classification, DTW, DNN-HMM, CTC, Full ASR model and more.

## Exercise 1
In Exercise 1, we have explored Fourier transforms, MFCC, Mel spectrograms, and basic audio handling and manipulation techniques. This exercise serves as a foundation for understanding speech processing and recognition. It covers fundamental concepts and techniques that are widely used in the field.

<img width="449" alt="log_mel" src="https://github.com/Dor890/Speech-Processing/assets/64433958/dcabab37-316f-492f-bfb2-a91dd3de01ac">


## Exercise 2
Exercise 2 focuses on music classification using different features. We implemented there logistic regression from scratch to build a music classifier. This exercise provides hands-on experience with implementing a machine learning algorithm and applying it to the task of music classification. By completing this exercise, we have gained insights into feature selection and classification techniques.


## Exercise 3
In Exercise 3, We delved into digit audio classification. This exercise specifically uses MFCC and DTW (Dynamic Time Warping) to build a classifier for recognizing spoken digits. By working on this exercise, We have gained practical knowledge of applying MFCC and DTW in the context of speech recognition.

![image](https://github.com/Dor890/Speech-Processing/assets/64433958/436f4767-37c2-4b75-96ce-19d11acbbb92)


## Exercise 4
In this exercise we implement the CTC loss in Python, which calculates the probability of a specific labeling given the model’s output distribution over phonemes. We assume to be given with a sequence of phonemes p and the network’s output y. In words, y is a matrix with the shape of T × K where T is the number of time steps, and K is the amount of phonemes, where each column i of y is a distribution over K phonemes at time i. Our goal is to implement the CTC function to calculate P(p|x) using the following equations:

![image](https://github.com/Dor890/Speech-Processing/assets/64433958/e1bfab5b-637d-49c6-8698-642774d01d36)

Where the final probability is given by the following:

![image](https://github.com/Dor890/Speech-Processing/assets/64433958/6cdf7af0-005c-4d75-b016-9d85b57a87bf)



## Final Project
For the final project we builded a full Automatic Speech Recognition (ASR) pipeline, when we used here everything we have learned in the
course including: Dynamic Time Warping (DTW), Hidden Markov Models (HMMs), DNN-HMM, End-to-End models: CTC, ASG, RNN Transducers, LAS, etc., Language modeling, Different acoustic feature, Ensemble methods, Different search methods e.t.c. 

Throughout our project we documented evaluated, analyzed, and document the performance of each of the tested configurations, that way we make smarter decisions when picking the final model, which is considered as a full-blown research project (on a small scale).

We used here AN4 dataset. This is a small dataset recorded and distributed by Carnegie Mellon University (CMU). It consists of recordings of people spelling out addresses, names, etc. Moreover, we evaluate out model using Word Error Rate (WER) and Character Error Rate (CER).

The structure of the model is as folowing:
![image](https://github.com/Dor890/Speech-Processing/assets/64433958/788a3f41-b831-4d53-9ea0-928642dc47f0)


## Conclusion
We hope this repository provides you with a valuable learning experience in speech processing and recognition. The exercises are designed to cover key concepts and techniques, allowing you to gain hands-on experience with various aspects of speech-related tasks. Feel free to explore the exercises, experiment with the code, and expand your knowledge in this exciting field!

Happy coding!
