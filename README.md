# ArrayConvNet
Using convolutional neural networks (CNNs), we propose a new technique for automatic earthquake detection and 4D localization. This repository contains the code and trained models to write our paper. The full insights and report should be referenced at our paper submission. 

## Data
This study is based on the earthquake information and waveform data from the Hawaiian Volcano Observatory (HVO), run by the USGS. The USGS earthquake catalog is obtained from the [USGS](https://earthquake.usgs.gov/earthquakes/search/), last accessed March 23, 2020.  The waveform data is available from the IRIS DMC.

Special thanks to researchers and staff at HVO for collecting the seismic data and providing the earthquake catalog used in this study and Quoqing Lin for providing the relocated earthquakes that are used to assess the effects of location accuracy in the training data set.

## Preprocessing the data
To process the raw trace data, we use two scripts:
1. `generate_detect_data_3cRand.py`
2. `generate_event_array3c4d.py`

The first script processes all detection data given earthquake and noise events. It creates the files for the training and test data for training the detection model.

Similarly, the second script processes all earthquake event data and their labels. It creates the files for the training and test data for training the localization model.   

## Training the models
Once all data is processed, we train and test the detection and localization model with the scripts `detect_3c.py` and `predict_location3c4d.py`, respectively. The output of these scripts are the trained models with their accuracies on test datasets. 

### Trained models
For ease of use, we've included trained detection and localization models. The directory `models` contains:

- `SeisConvNetDetect_sortedAbs50s.pth`: trained model for detection
- `SeisConvNetLoc_NotAbs2017Mcut50s.pth`: trained model for 4d localization

Together, they create ArrayConvNet.

## Validation on continuous data
Earthquake catalogs usually represent only a subset of earthquakes that occurred, with detection and localization limited by signal-to-noise ratios in seismic records, number of detected stations, and other factors. Our training data from the USGS catalog for Hawaii is no exception. So while our ArrayConvNet performs well for the validation data set, we tested further on continuous data to evaluate its true efficacy. In `validate_consecution.py`, we pass in continous seismic readings and evaluate the results.
