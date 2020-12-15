Introduction
====
A Smart Model for Epigenetics in Plants (SMEP) was constructed to identify various epigenomic sites using deep neural networks (DNNs) approach. The SMEP prediction leverages associations with DNA 5-methylcytosine (5mC) and N6-methyladenosine (6mA) methylation, RNA N6-methyladenosine (m6A) methylation, histone H3 lysine-4 trimethylation (H3K4me3), H3K27me3, and H3 lysine-9 acetylation (H3K9ac). You can use this code to train your own prediction models, or employed the constructed model to predict the modification of the input sequence in Rice.

System requirement
=====
1. Python 2.7 or Python 3.5
2. tensorflow 2.0.0
3. keras 2.3.1
4. theano 1.0.4

Quick Start to install the required program
====
1. Install the python 2.7 or 3.5 from Anaconda https://www.anaconda.com/
2. pip install tensorflow==2.0.0
3. pip install keras==2.3.1
4. pip install theano==1.0.4
5. git clone https://github.com/BRITian/smep

Training models
====
The program smep_train_py2.7.py or smep_train_py3.5.py was used to train the prediction model, in the python environment 2.7 or 3.5, respectively. There are four parameters that should be provided with the following order, training filename, test filename, sequence length and class number of the model. In the coding file, the first coloum is the label of the sequence, which is the modified or unmodified state. The followings in the line is the coding data, and each nucleotide is encoded as a number, which the A is encoded as the 0, T is encoded as 1, C is encoded as 2 and G is encoed as 3. 
The followings are the examples to construct the predicting models in the python environment 2.7.  
1.	The 5mC predicting model  
python smep_train_py2.7.py example_Test_file_5mC example_Train_file_5mC 41 4  
2.	The 6mA predicting model  
python smep_train_py2.7.py example_Test_file_6mA example_Train_file_6mA 41 2  
3.	The m6A predicting model  
python smep_train_py2.7.py example_Test_file_m6A example_Train_file_m6A 800 2  
4.	The histone H3 lysine-4 trimethylation (H3K4me3) predicting model  
python smep_train_py2.7.py example_Test_file_H3K4me3 example_Train_file_H3K4me3 800 2  
5.	The histone H3 lysine-27 trimethylation (H3K27me3) predicting model  
python smep_train_py2.7.py example_Test_file_H3K27me3 example_Train_file_H3K27me3 800 2  
6.	The histone H3 lysine-9 acetylation (H3K9ac) predicting model  
python smep_train_py2.7.py example_Test_file_H3K9ac example_Train_file_H3K9ac 800 2  
	The model file will be constructed in the current directory.  

Prediction the modifications in the sequence
====
The main program smep_prediction.pl could be used to predict the modification in the sequence. There are three parameters (-I -T -O) that should be provided.  

perl smep_prediction.pl -I input_fasta_sequence -T modification_type -O output_file  

-I, The input sequence with fasta format  
-T, The epigenetic modification type. There are six pre-constructed models (5mC, 6mA, m6A, H3K27me3, H3K4me3 or H3K9ac)     
-O, The output file  

The followings are some command examples.  
perl smep_prediction.pl -I test_5mC.fasta -O test_5mC.out -T 5mC  
perl smep_prediction.pl -I test_6mA.fasta -O test_6mA.out -T 6mA  
perl smep_prediction.pl -I test_m6A.fasta -O test_m6A.out -T m6A  
perl smep_prediction.pl -I test_ H3K27me3.fasta -O test_ H3K27me3.out -T H3K27me3  
perl smep_prediction.pl -I test_ H3K4me3.fasta -O test_ H3K4me3.out -T H3K4me3  
perl smep_prediction.pl -I test_ H3K9ac.fasta -O test_ H3K9ac.out -T H3K9ac  
  
  
The predicted results were saved in the output file. In the predicted file, the first column is the fragment number. The second and third column are the sequence ID and the location of the first nucleic acid in the fragment. The fourth and fifth columns are the predicted flag for the modification marker and the probability. The sixth column is the sequence of the fragment. The flag and its corresponding modification were shown as the followings.   
1.	5mC, 0 (No modification), 1 (), 2, 3.  
2.	For the other modifications (6mA, m6A, H3K4me3, H3K27me3 and H3K9ac), the number 0 and 1 represented the non-modification and modification, respectively.   

