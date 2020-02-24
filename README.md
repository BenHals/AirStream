# AirStream
A data stream system to make predictions in non-stationary conditions and adapt to changes in unknown/unobserved features.
AirStream was specifically designed for inferring air quality level from recent readings from surrounding sensors.
This process is affected by changing conditions such as wind direction, wind speed, pollution source behavior changes and seasonality.
These changes must be adapted to in some way, even if no monitoring is availiable.

AirStream uses state-of-the-art data stream methods to detect and adapt to change in conditions it cannot observe.
Based on a foundation of detecting change and dynamically switching classifier, AirStream can also create a model of these changing conditions.
AirStream detects _concept drift_, a change in distribution on incoming data to identify changes in causal conditions.
By matching current stream conditions to past classifiers, AirStream selects the best classifier to use or builds a new one.
Under the hypothesis that a classifier will perform similarly given similar conditions, this selection provides information on current conditions.

![Air Pollution](Poll_overlay.pdf)

## Instructions To Run
0. Install 
 - numpy
 - scikit-multiflow (We depend on the HoeffdingTree implementation).
 - tqdm (For progress bars).
 - utm (For working with latitude and longitude locations).

### Data Formatting
1. Each data set should be in a folder inside `RawData`, with the name of the folder being the unique ID of the data set, `$dataname$`.
 - This folder should contain a csv file named `$dataname$_full.csv`. The first column in this file should be a datetime, then a column for each sensor (feature) then a column for each auxiliary feature (used for evaluating relationship to weather, not for testing/training or deployment).
 - This folder should also contain a `$dataname$_sensors.json` file giving the x,y positions for each sensor in meters.
 - An example is shown in the `TestData` folder, and a jupyter notebook used to construct the files from raw data is given as `process_data_template.ipynb`.

### Running Evaluation
2. The main evaluation entry point is `conduct_experiment.py`. This script runs all baselines and AirStream on a specified data set with all combinations of passed settings. The script also preprocesses the data file, setting the target sensor, removing auxiliary features and applying masking.
 - Important command line arguments availiable are:
  - `-rwf`: The `$dataname$` of the data set, must match the folder and files in `RawData`.
  - `-hd`: A flag denoting if the data files have headers. Usually should be set.
  - `-ti`: The index of the sensor to designate the target, i.e. the sensor which will be predicted. Can specify multiple to run multiple times using different targets e.g. `-ti 0 1 2` will run 3 times using sensor 0, 1 and 2 as targets.
  - `-bp` & `-bl`: The break percentage and break length respectively (in terms of observations seen). Used for masking. Defaults to bp of 0.02 and bl of 75.
  - `-d`: changing the directory containing input. Should be set to a parent directory containing a `RawData` directory.
  - `-o`: changing the directory containing output. Should be set to a parent directory containing a `experiments` directory.
  - `-bcs`: Select which baselines to run from lin, temp, normSCG, OK, tree and arf. Defaults to all.
  - Arguments for changing AirStream settings. Refer to code, defaults to those used in the paper.
  - Output is written to `experiments` with a file structure: `experiments/$dataname$/$target index$/$seed$`
  - All classifier output is written to this folder, with AirStream results denoted `sys...`.
  - The `..._results.json` files provide all results for a given classifier, including accuracy and relationship to auxiliary features.
  - An example command to run is: `python conduct_experiment.py -rwf TestData -hd`

### Running AirStream
3. Code for the AirStream classifier is in `ds_classifier.py`. To use:
 - Initialize the classifier using `AirStream = DSClassifier(learner = HoeffdingTree)` (or pass in settings).
 - Incrementally train using `AirStream.partial_fit(X, y)` where X is a list of observations (use shape [1, -1] for a single observation) and y is a list of labels.
 - Classify using `AirStream.predict(X)` where X is a list of observations (use shape [1, -1] for a single observation).
 - AirStream active state at any point is given by `AirStream.active_state`. The value immediately prior to any prediction is the ID of the classifier used to make that prediction.


