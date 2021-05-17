SOURCE CODE FOR "LEARNING-BASED VIEW SYNTHESIS FOR LIGHT FIELD CAMERAS"

This package is a MATLAB implementation of the learning-based view synthesis
algorithm described in:

N. K. Kalantari, T. Wang, R. Ramamoorthi
"Learning-Based View Synthesis for Light Field Cameras", 
ACM Transaction on Graphics 35, 6, December 2016. 

More information can also be found on the authors' project webpage:
http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/

Initial release implementation by Nima K. Kalantari, 2016.

This material is based upon work supported by the Office of Naval Research
under Grant No. N00014152013 and National Science Foundation under Grants 
No. 1451830 and 1617234.

-------------------------------------------------------------------------
I. OVERVIEW

This algorithm takes in a decoded Lytro light field image (in .png format) 
and produces the full 8 by 8 light field from only the four corner
sub-aperture images. The output for each location will be written in the
"Results" directory as an individual image. The code also outputs the
numerical quality evaluation for location (5, 5) in a text file.

The code was written in MATLAB 2015b, and tested on Windows 10.

-------------------------------------------------------------------------
II. RUNNING TEST CODE

1. Download MatConvNet package from the following link:
http://www.vlfeat.org/matconvnet/

unzip it and copy it to the "Libraries" folder. Then follow the MatConvNet
installation instruction to set it up properly. Please compile MatConvNet
with GPU and Cudnn support to have the best performance. Otherwise, please
set the appropriate flags in the "InitParam.m" file accordingly.

2. Download the "Test set" from the project page, unzip it and copy the
desired png files into the scene folder. The results shown in the paper
can be found in the "PAPER" folder of the test set.

3. The code is now ready to be executed. Simply open "Test.m" in MATLAB and
run it. The code will read all the scene files in the "Scenes" folder and
reconstructs the full light field from just the four corner sub-aperture images.

Note 1: to run the code on your own light fields captured with a Lytro Illum camera,
simply use the Lytro Power Tools (https://www.lytro.com/imaging/power-tools)
to extract the raw light field in png format. Then copy the png file to the
"Scenes" folder and run the main file.

Note 2: if you are not able to compile the MatConvNet package on Mac, please
download one of the older versions and set opts.enableImreadJpeg = false.


-------------------------------------------------------------------------
III. RUNNING TRAINING CODE

1. Follow step 1 of section III to setup the MatConvNet

2. Download the "Training set" and "Test set" from the project page, unzip
them and copy the png files in the "TrainingData/Training" and 
"TrainingData/Test" folders respectively.

3. Run "PrepareData.m" to process the training and test sets. It takes a long 
time for the first training image to be processed since a huge h5 file needs 
to be created first.

4. Run "Train.m" to start the training. Convergence happens after roughly
180,000 iterations. This takes around 2 days on an Nvidia Geforce GTX 1080.

The code writes the network in the "TrainingData" folder. Once converged network
is obtained, you can copy the network from "TrainingData" to "TrainedNetwork"
folder to test the system using the new network.



-------------------------------------------------------------------------
IV. VERSION HISTORY

v1.0 - Initial release   (Sep., 2016)

v2.0 - Training code added (Feb. 2017)

-------------------------------------------------------------------------

If you find any bugs or have comments/questions, please contact 
Nima K. Kalantari at nkhademi@ucsd.edu.

San Diego, California
February, 2017