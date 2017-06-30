# US-TransportationMode

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()
<a href="https://www.python.org/"><img src="https://camo.githubusercontent.com/65bf37ffbdcaef2a2cb000f66e2f395b32243357/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6275696c64253230776974682d707974686f6e322e372d7265642e737667" alt="built with Python2.7" data-canonical-src="https://img.shields.io/badge/build%20with-python2.7-red.svg" style="max-width:100%;"></a>

**US-Transporation** is the name of our dataset that contains sensor data from over 13 users. In light of the lack in the literature of a common benchmark for TMD, we have collected a large set of measurements belonging to different subjects and through a simple Android Application. We openly release the dataset, so that other researchers can benefit from it for further improvements and research reproducibility.<br>
Our dataset is built from people of different gender, age and occupation. Moreover, we do not impose any restriction on the use of the application, hence every user records the data performing the action as she/he is used to, in order to assess real world conditions. <br>
In this page in addition to downloadable datasets, you can find Python's code for extracting features,and building machine learning models to make predictions. <br>
You can find more information about the dataset and our work at: http://cs.unibo.it/projects/us-tm2017/index.html.

## Dependecies
In order to extecute the code in the repository you'll need to install the following dependencies:
* <a href="https://www.python.org/">Python 2.7</a>
* <a href="http://scikit-learn.org/stable/">Scikit-learn</a>
* <a href="http://pandas.pydata.org/">Pandas</a>

## Documentation
### Code
In this section we show the functionalities developed in our work and the relative parameters used.
#### Function
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
<li>delete measure from  **sensor_to_exclude</li>
<li>if **sound** or **speed** measure rows have negative time, use module</li>
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
<td>sensor_set:</td>
<td>Return list of excluded sensor based on the correspondent classification level</td>
</tr>

<tr>
<td>get_remained_sensors(sensor_set)</td>
<td>sensor_set:</td>
<td>Return list of considered sensors based on the correspondent classification level</td>
</tr>

<tr>
<td>get_sensors_set_features()</td>
<td></td>
<td></td>
</tr>

<tr>
<td>get_sensor_features()</td>
<td></td>
<td></td>
</tr>

</tbody>
</table>


### Run
Before starting with detection, you have to clean the raw data and extract the feature:
```shell
python TMDataset.py
```
Next you can build models: 
```shell
python TMDetection.py
```
### Project Structure
Up to now the projects is structured as follows:

## License
This work is licensed under a MIT License.

## Author
This project has been developed at the University of Bologna with the effort of different people:

* <a href="http://www.cs.unibo.it/~difelice/">Marco Di Felice</a>, Professor - email: marco.difelice3@unibo.it
* <a href="http://vincenzolomonaco.com/">Vincenzo Lomonaco</a>, PhD Student - email: vincenzo.lomonaco@unibo.it
* <a href="https://www.unibo.it/sitoweb/luca.bedogni4">Luca Bedogni</a>, Research Associate - email: luca.bedogni4@unibo.it
