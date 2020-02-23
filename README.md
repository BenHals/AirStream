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

## Instructions To Run

### Data Formatting
1. Each data set should be in a folder inside `RawData`, with the name of the folder being the unique ID of the data set, `dataname`.
 - This folder should contain a csv file named `dataname_full.csv`. The first column in this file should be a datetime, then a column for each sensor (feature) then a column for each auxiliary feature (used for evaluating relationship to weather, not for testing/training or deployment).
 - An example is shown in the `TestData` folder, and a jupyter notebook used to construct the file is given as `process_data_template.ipynb`.

2. Run