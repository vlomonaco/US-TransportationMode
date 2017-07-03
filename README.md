# US-TransportationMode

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()
[![built with Python2.7](https://camo.githubusercontent.com/65bf37ffbdcaef2a2cb000f66e2f395b32243357/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6275696c64253230776974682d707974686f6e322e372d7265642e737667)](https://www.python.org/)

**US-Transporation** is the name of our dataset that contains sensor data from over 13 users. In light of the lack in the literature of a common benchmark for TMD, we have collected a large set of measurements belonging to different subjects and through a simple Android Application. We openly release the dataset, so that other researchers can benefit from it for further improvements and research reproducibility.<br>
Our dataset is built from people of different gender, age and occupation. Moreover, we do not impose any restriction on the use of the application, hence every user records the data performing the action as she/he is used to, in order to assess real world conditions. <br>
In this page in addition to downloadable datasets, you can find Python's code for extracting features,and building machine learning models to make predictions. <br>
You can find more information about the dataset and our work at: http://cs.unibo.it/projects/us-tm2017/index.html.

# Menù

* **[Dependecies](https://github.com/vlomonaco/US-TransportationMode/blob/master/README.md#dependecies)**
* **[Documentation](https://github.com/vlomonaco/US-TransportationMode/blob/master/README.md#documentation)**
  * **[Code](https://github.com/vlomonaco/US-TransportationMode/blob/master/README.md#code)** 
  * **[Get started](https://github.com/vlomonaco/US-TransportationMode/blob/master/README.md#get-started)**
  * **[Project Structure](https://github.com/vlomonaco/US-TransportationMode/blob/master/README.md#project-structure)**
* **[License](https://github.com/vlomonaco/US-TransportationMode/blob/master/README.md#license)**
* **[Team of collaborators](https://github.com/vlomonaco/US-TransportationMode/blob/master/README.md#team-of-collaborators)**

## Dependecies
In order to extecute the code in the repository you'll need to install the following dependencies:
* [Python 2.7](https://www.python.org/)
* [Scikit-learn](http://scikit-learn.org/stable/)
* [Pandas](http://pandas.pydata.org/)

## Documentation
### Code
In this section we show the functionalities developed in our work and the relative parameters used.
#### TMDataset.py
<table>
<thead>
<th>Function name</th>
<th>Parameter</th>
<th>Description</th>
</thead>
<tbody>
<tr>
<td>clean_file()</td>
<td></td>
<td>
Fix original raw files problems: 
<ul>
<li>delete measure from  <strong>sensor_to_exclude</strong></li>
<li>if <strong>sound</strong> or <strong>speed</strong> measure rows have negative time, use module</li>
<li>if **time** have incorrect values ("/", ">", "<", "-", "_"...), delete file</li>
<li>if file is empty, delete file</li> 
</ul>
</td>
</tr>

<tr>
<td>transform_raw_data()</td>
<td></td>
<td>Transform sensor raw data in orientation independent data (with magnitude metric)</td>
</tr>

<tr>
<td>__fill_data_structure</td>
<td></td>
<td>Fill tm, users, sensors data structures with the relative data from dataset</td>
</tr>

<tr>
<td>__range_position_in_header_with_features(sensor_name)</td>
<td>sensor_name: name of the sensor</td>
<td>Return position of input sensor in header with features</td>
</tr>

<tr>
<td>create_header_files()</td>
<td></td>
<td>Fill directory with all file consistent with the header without features</td>
</tr>

<tr>
<td>__create_time_files()</td>
<td></td>
<td>Fill directory with all file consistent with the featured header divided in time window</td>
</tr>

<tr>
<td>__create_dataset()</td>
<td></td>
<td>Create dataset file</td>
</tr>

<tr>
<td>__split_dataset()</td>
<td></td>
<td>Split passed dataframe into test and train</td>
</tr>

<tr>
<td>preprocessing_files()</td>
<td></td>
<td>Clean files and transform in orientation independent</td>
</tr>

<tr>
<td>analyze_sensors_support()</td>
<td></td>
<td>For each sensors analyze user support, put support result in sensor_support.csv [sensor,nr_user,list_users,list_classes]</td>
</tr>

<tr>
<td>create_balanced_dataset(sintetic)</td>
<td>sintetic: set if data are sintentic or not. Default the value is <strong>False</strong>.</td>
<td>Analyze dataset composition in term of class and user contribution fill balance_time with minimum number of window for transportation mode</td>
</tr>

<tr>
<td>get_excluded_sensors(sensor_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data.</td>
<td>Return list of excluded sensor based on the correspondent classification level</td>
</tr>

<tr>
<td>get_remained_sensors(sensor_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data.</td>
<td>Return list of considered sensors based on the correspondent classification level</td>
</tr>

<tr>
<td>get_sensors_set_features()</td>
<td></td>
<td>Return list of the sensors set with their features</td>
</tr>

<tr>
<td>get_sensor_features(sensor)</td>
<td>sensor: data of a specific sensor</td>
<td>Return the features of a specific sensor</td>
</tr>

</tbody>
</table>

#### TMDetection.py

<table>
<thead>
<th>Function name</th>
<th>Parameter</th>
<th>Description</th>
</thead>
<tbody>

<tr>
<td>decision_tree(sensors_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data</td>
<td>Decision tree algorithm training on training al train set and test on all test set</td>
</tr>

<tr>
<td>random_forest(sensors_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data</td>
<td>Random forest algorithm training on training al train set and test on all test set</td>
</tr>

<tr>
<td>neural_network(sensors_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data</td>
<td>Neural network algorithm training on training al train set and test on all test set</td>
</tr>

<tr>
<td>support_vector_machine(sensors_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data</td>
<td>Support vector machine algorithm training on training al train set and test on all test set</td>
</tr>

<tr>
<td>classes_combination(sensors_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data</td>
<td>Use different algorithms changing target classes, try all combination of two target classes</td>
</tr>
<tr>
<td>leave_one_subject_out(sensors_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data</td>
<td></td>
</tr>
<tr>
<td>support_vector_machine(sensors_set)</td>
<td>sensor_set: type of sensor dataset used with different sensor data</td>
<td>Use different algorithms leaving one subject out from training and testing only on this subject considering all classes in dataset and only user classes</td>
</tr>

<tr>
<td>single_sensor_accuracy()</td>
<td></td>
<td>Use feature relative to one sensor to build model and evaluate</td>
</tr>

</tbody>
</table>

### Get started
Before starting, you must first download the data:
```shell
python download_dataset.py
```
Then you have to clean the raw data and extract the feature:

```shell
python TMDataset.py
```
Finally you can build models: 
```shell
python TMDetection.py
```
For further and detail information about our code, see our [tutorial section](http://cs.unibo.it/projects/us-tm2017/tutorial.html)

### Project Structure
Up to now the projects is structured as follows:
```unicode
.
├── TransportationData
|   ├── datasetBalanced
|         └── ...
|   └── _RawDataOriginal
|         └── ...
├── README.md
├── LICENSE
├── const.py
├── function.py
├── TMDataset.py
├── TMDetection.py
├── util.py
├── sintetic_dataset_generator.py
├── sintetic_dataset_config.json
├── download_dataset.py
└── cleanLog.log
```
## License
This work is licensed under a MIT License.

## Team of collaborators
This project has been developed at the University of Bologna with the effort of different people:

* <a href="http://www.cs.unibo.it/~difelice/">Marco Di Felice</a>, Associate Professor - email: marco.difelice3@unibo.it
* <a href="http://www.cs.unibo.it/bononi/" target="_blank">Luciano Bononi</a>, Associate Professor - email : luciano.bononi@unibo.it
* <a href="https://www.unibo.it/sitoweb/luca.bedogni4">Luca Bedogni</a>, Research Associate - email: luca.bedogni4@unibo.it
* <a href="http://vincenzolomonaco.com/">Vincenzo Lomonaco</a>, PhD Student - email: vincenzo.lomonaco@unibo.it
* <a href="https://www.linkedin.com/in/matteo-cappella-30005b111/" target="_blank">Matteo Cappella</a> • Master student - email: matteo.cappella@studio.unibo.it
* <a href="https://www.linkedin.com/in/simone-passaretti-397b23110/" target="_blank">Simone Passaretti</a> • Master student - email: simone.passaretti@studio.unibo.it
### Past collaborators
* <a href="https://www.linkedin.com/in/claudiacarpineti/" target="_blank">Claudia Carpineti</a> • Master graduate student - email: claudia.carpineti@studio.unibo.it
