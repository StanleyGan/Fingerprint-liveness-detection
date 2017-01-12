# Author: Stanley Gan, glgan@sfu.ca
# Fingerprint-liveness-detection using Neural Network models, built using Keras ontop of Tensorflow

Fingerprint images are based on the LivDet2015 competition, in which our group requested personally from the organization. Please refer to the organization if you would like to access the images.

I worked my models on Digital Persona scanned images, feel free to try others

1) My models utilized features which are extracted by statistical methodologies Binarized Statistical Image Features(BSIF), 
Local Phase Quantization(LPQ) and Weber Local Descriptor(WLD). Here are the references:
- BSIF: http://www.ee.oulu.fi/~jkannala/bsif/bsif.pdf
- LPQ: http://www.cse.oulu.fi/CMV/Downloads/LPQMatlab
- WLD: http://www.ee.oulu.fi/~jiechen/paper/TPAMI2009-WLD.pdf

MATLAB codes for each method are available from the author's website.

2) Once you run the MATLAB codes in extracting features from fingerprint images, you will have a large matrix of numbers. 
Replace the extracted features file destination in my code files to your own file destination.

3) There are 4 models here, in which each based on BSIF features, LPQ features, WLD features and MixFeat(BSIF,LPQ,WLD). 
Each model has respective cross validate and test files. Cross validate files are to finetune only on the layers, dropout rate etc. 
of the models. You can tweak these settings to achieve better performances if you like to. After cross validating, transfer the 
settings you had to the corresponding test files and run them to get the models' performances on testing. After end of testing,
the test files will print an ACE score, which is one of the performance metric used by LivDet2015.
