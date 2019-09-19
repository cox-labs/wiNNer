# wiNNer
Sliding-window-based regression model for prediction of fragment intensities using Python and Keras 

## Overview

wiNNer(window-based neural network being easily retrainable)uses classical sliding window-based machine learning algorithm 
to predict peptide fragment intensity.
wiNNer is build using conventional neural network using  Keras (https://keras.io) v.2.0.8, a high-level neural network application programming interface. 
TensorFlow v.1.3.0 was used as backend in Keras.
Here, we provide a set of instructions for how to use the model for the intensity prediction.

Citation
--------

If you use `wiNNer` in your projects, please cite

Tiwary, S. et al.,*High-quality MS/MS spectrum prediction for data-dependent and data-independent acquisition data analysis* [doi:10.1038/s41592-019-0427-6](https://doi.org/10.1038/s41592-019-0427-6).

You can predict fragment intensities using wiNNer in two ways:

*   using wiNNerprediction.exe.
*   using wiNNerprediction python script.

## Prerequisites

*   Clone this repository to your local machine, using the following command: \
    `https://github.com/cox-labs/wiNNer.git`
*   There are no system-specific requirements.

## To run python script Install
 
*   Python 
*   Numpy (`pip install numpy`)
*   Pandas (`pip install pandas`)
*   keras (`pip install keras`)
*   Tensorflow v1.7 (`pip install tensorflow==1.7.0`)

## To run with command prompt 

For tryptic peptide fragment intensity prediction use wiNNer_model as modelname.
For ancient sample peptides fragment intensity predictions use ancient_model as modelname.

```
Python wiNNerprediction.py -i <inputfile> -o <outputfile> -d <modelname> -s <Sequence>
Python wiNNerprediction.py -i <inputfile> -o <outputfile> -d <modelname> -s <ModifiedSequence>

```

## To run executable 

No installation required.
Model and executable should be in one folder.
Run the executable using command prompt.
Select appropriate input file, model name directory and the sequence column 
Sequence column can be Sequence or ModifiedSequence  

```
wiNNerprediction.exe -i <inputfile> -o <outputfile> -d<modelname> -s<Sequence>
wiNNerprediction.exe -i <inputfile> -o <outputfile> -d<modelname> -s<ModifiedSequence>

```
## Data format

Input data table should be written in TXT (tab-separated) file format and
contain at least the following columns:

*   Peptide sequence:
    *   Amino-acid modifications should be given in “(modification)” format as in MaxQuant output tables -
        for example, “ACDM(ox)FK” is a valid format,
*   Charge:
    *   Our model can handle charges up to 7.
*   Fragmentation type:
    *   Our model currently supports HCD and CID fragmentations.
	* 	For ancient_model we only support HCD fragmentations.

The input table can contain any number of columns, and the 3 required columns.
Here’s an example input:

```
ModifiedSequence,Charge,Fragmentation
AKM(ox)LIVR,3,HCD
ILFWYK,2,CID
or
_AKM(ox)LIVR_,3,HCD
_ILFWYK_,2,CID
```


## Contact
shivani@biochem.mpg.de.