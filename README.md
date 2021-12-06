# NLPARG

**N**atural *L*anguage **P**rocessing for **A**ntimicrobial **R**esistance **G**enes (**NLPARG**) software was developed to predict antimicrobil resistance genes. Three neural embedding models, **Global vectors (GloVe), skip-gram (Sg) and Continuous-bag-of-words (CBOW)** were used as embedding layersin recurrent neural networks. Users have the option of using any of the three three neural layers used in the models for prediction.


Users have to first download this repository using the command below:
```
git clone https://github.com/danielobiri/NLPARG.git

```
work in the NLPARG directory
```
cd NLPARG
```

Choice of wordvector

```
Users have the option of choosing any f the three embedding layers:
1. Global vectors
2. Skip-gram
3. continuous-bag-of-words
```

The template command for NLPARG is shown below

```
python3 NLPARG.py -choice_wordvector [1,2 or 3] -in your_input_file.fasta  -out file_results
```
Running with default sample
choice of word vector: Global vector (1)
input_file: 
```
python3 NLPARG.py -choice_wordvector 1 -in Positive_training_set.fasta  -out Positive_training_set.fasta_results
```
The output file
```
The following are contained in the output file
The date and time of execution
The type of embedding model use
Each sequence has the annotation **(an antimicrobial resistance gene sequence or non-antimicrobial resistance gene sequence)** with the corresponding **probability**
At the end of the file, the number of antimicrobial resistance gene sequences identied are presented
```
