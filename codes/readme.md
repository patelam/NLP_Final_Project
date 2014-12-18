We have stored our code in 3 folders:

1. Reranker
2. Decoder
3. aligner

In the reranker folder, we have five files:

1. score-reranker.py: Computes the BLEU scores for test data
2. uneven.py: Uses ordinal regression algorithm with eight features
3. rerank_five.py: Uses ordinal regression algorithm with five features 
4. bleu.py: Computes the BLEU scores for a translated sentence with respect to reference sentence
5. baseline.py: Uses perceptron algorithm with five features


In the decoder, we have three files:

1. beam7.py: Uses count of untranslated words, count of number of words along with the five given features, and the beam search decoder.  
2. models.py: Computes the TM-LM probabilities from the input.      
3. beam8.py: Uses alignment features, count of untranslated words, count of number of words along with the five given features, and the beam search decoder.

In the aligner code, we have one file:
1. ibm1.py-Generates the IBM model1 alginment weight file


All these codes were run from the root directory(project/).  

To run the codes for our project, we use the set of similar instructions as mentioned in test.py
