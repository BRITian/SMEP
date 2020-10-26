Introduction
====
A Smart Model for Epigenetics in Plants (SMEP) was constructed to identify various epigenomic sites using deep neural networks (DNNs) approach. The SMEP prediction leverages associations with DNA 5-methylcytosine (5mC) and N6-methyladenosine (6mA) methylation, RNA N6-methyladenosine (m6A) methylation, histone H3 lysine-4 trimethylation (H3K4me3), H3K27me3, and H3 lysine-9 acetylation (H3K9ac). 

System requirement
=====
1. Python 2.7 or Python 3.5
2. tensorflow 2.0.0
3. keras 2.3.1
4. theano 1.0.4

Quick Start
====
1. Install the python 2.7 or 3.5 from Anaconda https://www.anaconda.com/
2. pip install tensorflow==2.0.0
3. pip install keras==2.3.1
4. pip install theano==1.0.4
5. git clone https://github.com/BRITian/smep

Training
====

The program smep_train_py2.7.py or smep_train_py3.5.py was used to train the prediction model, in the python environment 2.7 or 3.5, respectively. There are four parameters that should be provided with the following order, training filename, test filename, sequence length and class number of the model.  
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

Prediction
====
The program smep_prediction_py2.7.py or smep_prediction_py3.5.py was used to train the prediction model, in the python environment 2.7 or 3.5, respectively. There are three parameters that should be provided with the following order, input filename, model name and the sequence length.  
The followings are the examples to predict the modification sites in the python environment 2.7.  
1.	Predicting the 5mC modification sites  
python smep_prediction_py2.7.py example_Test_file_5mC example_Train_file_5mC 41 4  
2.	Predicting the 6mA modification sites  
python smep_prediction_py2.7.py example_Test_file_6mA example_Train_file_6mA 41 2  
3.	Predicting the m6A modification sites  
python smep_prediction_py2.7.py example_Test_file_m6A example_Train_file_m6A 800 2  
4.	Predicting the histone H3 lysine-4 trimethylation (H3K4me3) modification sites  
python smep_prediction_py2.7.py example_Test_file_H3K4me3 example_Train_file_H3K4me3 800 2  
5.	Predicting the histone H3 lysine-27 trimethylation (H3K27me3) modification sites  
python smep_prediction_py2.7.py example_Test_file_H3K27me3 example_Train_file_H3K27me3 800 2  
6.	Predicting the histone H3 lysine-9 acetylation (H3K9ac) modification sites  
python smep_prediction_py2.7.py example_Test_file_H3K9ac example_Train_file_H3K9ac 800 2  
  
  The predicted results were saved in the file at the directory of ‘pres’. In the predicted file, the first and second columns are the sequence number and the predicted flag for the modification marker. The flag and its corresponding modification were shown as the followings. 
1.	5mC, 0 (Non-modification), 1 (CG), 2(CHG), 3(CHH).
2.	For the other modifications (6mA, m6A, H3K4me3, H3K27me3 and H3K9ac), the number 0 and 1 represented the non-modification and modification, respectively. 

