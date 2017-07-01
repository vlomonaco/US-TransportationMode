import csv
import logging
import math
import os
import re
import shutil
import json
from os import listdir
import pandas as pd
import const
import util
import sys
import math


class TMDataset:
    tm = []
    users = []
    sensors = []
    n_files = 0
    header = {}
    header_with_features = {}
    balance_time = 0  # in seconds
    train = pd.DataFrame()
    test = pd.DataFrame()
    cv = pd.DataFrame()
    sintetic = False

    @property
    def get_users(self):
        if len(self.users) == 0:
            self.__fill_data_structure()
        return self.users

    @property
    def get_tm(self):
        if len(self.tm) == 0:
            self.__fill_data_structure()
        return self.tm

    @property
    def get_sensors(self):
        if len(self.sensors) == 0:
            self.__fill_data_structure()
        return self.sensors

    @property
    def get_header(self):
        if len(self.header_with_features) == 0:
            self.__fill_data_structure()
        return self.header_with_features

    # Fix original raw files problems:
    # (1)delete measure from  **sensor_to_exclude**
    # (2)if **sound** or **speed** measure rows have negative time --> use module
    # (3)if **time** have incorrect values ("/", ">", "<", "-", "_"...) --> delete file
    # (4)if file is empty --> delete file
    def clean_files(self):
        if os.path.exists(const.CLEAN_LOG):
            os.remove(const.CLEAN_LOG)

        patternNegative = re.compile("-[0-9]+")
        patternNumber = re.compile("[0-9]+")

        # create directory for correct files
        if not os.path.exists(const.DIR_RAW_DATA_CORRECT):
            os.makedirs(const.DIR_RAW_DATA_CORRECT)
        else:
            shutil.rmtree(const.DIR_RAW_DATA_CORRECT)
            os.makedirs(const.DIR_RAW_DATA_CORRECT)

        # create log file
        logging.basicConfig(filename=const.CLEAN_LOG, level=logging.INFO)
        logging.info("CLEANING FILES...")
        print("CLEAN FILES...")
        filenames = listdir(const.DIR_RAW_DATA_ORIGINAL)
        # iterate on files in raw data directory - delete files with incorrect rows
        nFiles = 0
        deletedFiles = 0
        for file in filenames:

            if file.endswith(".csv"):
                nFiles += 1
                # to_delete be 1 if the file have to be excluded from the dataset
                to_delete = 0
                with open(os.path.join(const.DIR_RAW_DATA_ORIGINAL, file)) as current_file:
                    res_file_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
                    with open(res_file_path, "w") as file_result:
                        for line in current_file:
                            line_data = line.split(",")

                            first_line = True
                            if first_line:
                                first_line = False
                                if line_data[1] == "activityrecognition":
                                    line_data[0] = "0"

                            endLine = ",".join(line_data[2:])
                            # check if time data is correct, if is negative, make modulo
                            if re.match(patternNegative, line_data[0]):
                                current_time = line_data[0][1:]
                            else:
                                # if is not a number the file must be deleted
                                if re.match(patternNumber, line_data[0]) is None:
                                    to_delete = 1
                                current_time = line_data[0]
                            # check sensor, if is in sensors_to_exclude don't consider
                            if line_data[1] not in const.SENSORS_TO_EXCLUDE_FROM_FILES:
                                current_sensor = line_data[1]
                                line_result = current_time + "," + current_sensor + "," + endLine
                                file_result.write(line_result)

                # remove files with incorrect values for time
                if to_delete == 1:
                    logging.info("  Delete: " + file + " --- Time with incorrect values")
                    deletedFiles += 1
                    os.remove(res_file_path)

        # delete empty files
        file_empty = []
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        for file in filenames:
            full_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
            # check if file is empty
            if (os.path.getsize(full_path)) == 0:
                deletedFiles += 1
                file_empty.append(file)
                logging.info("  Delete: " + file + " --- is Empty")
                os.remove(full_path)

        pattern = re.compile("^[0-9]+,[a-z,A-Z._]+,[-,0-9a-zA-Z.]+$", re.VERBOSE)
        # pattern = re.compile("^[0-9]+,[a-z,A-Z,\.,_]+,[-,0-9,a-z,A-Z,\.]+$", re.VERBOSE)
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        for file in filenames:
            n_error = 0
            full_path = os.path.join(const.DIR_RAW_DATA_CORRECT, file)
            # check if all row respect regular expression
            with open(full_path) as f:
                for line in f:
                    match = re.match(pattern, line)
                    if match is None:
                        n_error += 1
            if n_error > 0:
                deletedFiles += 1
                os.remove(full_path)

        logging.info("  Tot files in Dataset : " + str(nFiles))
        logging.info("  Tot deleted files : " + str(deletedFiles))
        logging.info("  Remaining files : " + str(len(listdir(const.DIR_RAW_DATA_CORRECT))))

        self.n_files = len(listdir(const.DIR_RAW_DATA_CORRECT))
        logging.info("END CLEAN FILES")
        print("END CLEAN.... results on log file")

    # transform sensor raw data in orientation independent data (with magnitude metric)
    def transform_raw_data(self):
        if not self.sintetic:
            dir_src = const.DIR_RAW_DATA_CORRECT
            dir_dst = const.DIR_RAW_DATA_TRANSFORM
        else:
            dir_src = const.DIR_SINTETIC_RAW_DATASET
            dir_dst = const.DIR_SINTETIC_RAW_DATA_TRANSFORM

        if not os.path.exists(dir_src) and not self.sintetic:
            self.clean_files()

        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        if os.path.exists(dir_src):
            filenames = listdir(dir_src)
        else:
            shutil.rmtree(dir_dst)
            sys.exit("THERE ARE NO SYNTHETIC DATA TO BE PROCESSED")

        logging.info("TRANSFORMING RAW DATA...")
        print("TRANSFORMING RAW DATA...")
        for file in filenames:
            if file.endswith(".csv"):
                with open(os.path.join(dir_src, file)) as current_file:
                    with open(os.path.join(dir_dst, file), "w") as file_result:
                        for line in current_file:
                            line_data = line.split(",")
                            endLine = ",".join(line_data[2:])
                            current_time = line_data[0]
                            sensor = line_data[1]
                            user = "," + line_data[(len(line_data) - 2)] if self.sintetic else ""
                            target = "," + line_data[(len(line_data) - 1)] if self.sintetic else ""
                            target = target.replace("\n","")
                            # check sensors
                            if line_data[1] not in const.SENSORS_TO_EXCLUDE_FROM_DATASET:  # the sensor is not to exlude
                                if line_data[1] not in const.SENSOR_TO_TRANSFORM_MAGNITUDE:  # not to transofrom
                                    if line_data[1] not in const.SENSOR_TO_TRANSFROM_4ROTATION:  # not to trasform (4 rotation)
                                        if line_data[1] not in const.SENSOR_TO_TAKE_FIRST:  # not to take only first data
                                            # report the line as it is
                                            current_sensor = line_data[1]
                                            line_result = current_time + "," + current_sensor + "," + endLine
                                        else:
                                            current_sensor = line_data[1]
                                            vector_data = line_data[2:] if not self.sintetic else line_data[2:(len(line_data) - 2)]
                                            vector_data = [float(i) for i in vector_data]
                                            line_result = current_time + "," + current_sensor + "," + str(vector_data[0]) + user + target + "\n"
                                    else:  # the sensor is to transform 4 rotation
                                        current_sensor = line_data[1]
                                        vector_data = line_data[2:] if not self.sintetic else line_data[2:(len(line_data) - 2)]
                                        vector_data = [float(i) for i in vector_data]
                                        magnitude = math.sin(math.acos(vector_data[3]))
                                        line_result = current_time + "," + current_sensor + "," + str(magnitude) + user + target + "\n"
                                else:  # the sensor is to transform
                                    current_sensor = line_data[1]
                                    vector_data = line_data[2:] if not self.sintetic else line_data[2:(len(line_data)-2)]
                                    vector_data = [float(i) for i in vector_data]
                                    magnitude = math.sqrt(sum(((math.pow(vector_data[0], 2)),
                                                               (math.pow(vector_data[1], 2)),
                                                               (math.pow(vector_data[2], 2)))))
                                    line_result = current_time + "," + current_sensor + "," + str(magnitude) + user + target + "\n"
                                file_result.write(line_result)
            elif file.endswith(".json"):
                shutil.copyfile(os.path.join(dir_src,file),os.path.join(dir_dst,file))
        logging.info("END TRANSFORMING RAW DATA...")
        print("END TRANSFORMING RAW DATA...")

    # fill tm, users, sensors data structures
    def __fill_data_structure(self):
        if not self.sintetic:
            dir_src = const.DIR_RAW_DATA_TRANSFORM
            if not os.path.exists(dir_src):
                print("You should clean files first!")
                return -1
        else:
            dir_src = const.DIR_SINTETIC_RAW_DATA_TRANSFORM

        filenames = listdir(dir_src)

        for file in filenames:
            if file.endswith(".csv"):
                if not self.sintetic:
                    data = file.split("_")
                    if data[2] not in self.tm:
                        self.tm.append(data[2])
                    if data[1] not in self.users:
                        self.users.append(data[1])
                else:
                    json_name = os.path.splitext(file)[0] + '.json'
                    with open(os.path.join(dir_src,json_name)) as data_file:
                        data = json.load(data_file)
                        for activity in data["activities"]:
                            for key in activity:
                                if key not in self.tm:
                                    self.tm.append(key)
                        if data["samples"][0] == "all":
                            samples_list = data["samples_list"]
                        else:
                            samples_list = data["sample"]

                        self.users = samples_list


                f = open(os.path.join(dir_src, file))
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    if row[1] not in self.sensors and not row[1] == "":
                        self.sensors.append(row[1])
                f.close()

        self.header_with_features = {0: "time", 1: "activityrecognition#0", 2: "activityrecognition#1"}
        header_index = 3
        for s in self.sensors:
            if s != "activityrecognition":
                self.header_with_features[header_index] = s + "#mean"
                self.header_with_features[header_index + 1] = s + "#min"
                self.header_with_features[header_index + 2] = s + "#max"
                self.header_with_features[header_index + 3] = s + "#std"
                header_index += 4

        self.header = {0: "time", 1: "activityrecognition#0", 2: "activityrecognition#1"}
        header_index = 3
        for s in self.sensors:
            if s != "activityrecognition":
                self.header[header_index] = s + "#0"
                header_index += 1

    # return position of input sensor in header with features
    def __range_position_in_header_with_features(self, sensor_name):
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.__fill_data_structure()
        range_position = []
        start_pos = end_pos = -1
        i = 0
        found = False
        while True and i < len(self.header_with_features):
            compare = (str(self.header_with_features[i])).split("#")[0]
            if compare == sensor_name:
                found = True
                if start_pos == -1:
                    start_pos = i
                else:
                    end_pos = i
                i += 1
            else:
                i += 1
                if found:
                    if end_pos == -1:
                        end_pos = i - 2
                    break
        range_position.append(start_pos)
        range_position.append(end_pos)
        return range_position

    # return position of input sensor in header without features
    def __range_position_in_header(self, sensor_name):
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.__fill_data_structure()
        range_position = []
        start_pos = end_pos = -1
        i = 0
        found = False
        while True and i < len(self.header):
            compare = (str(self.header[i])).split("#")[0]
            if compare == sensor_name:
                found = True
                if start_pos == -1:
                    start_pos = i
                else:
                    end_pos = i
                i += 1
            else:
                i += 1
                if found:
                    if end_pos == -1:
                        end_pos = i - 2
                    break
        if end_pos == -1:
            end_pos = len(self.header) - 1
        range_position.append(start_pos)
        range_position.append(end_pos)
        return range_position

    # fill directory with all file consistent with the header without features
    def create_header_files(self):
        if self.sintetic:
            dir_src = const.DIR_SINTETIC_RAW_DATA_TRANSFORM
            dir_dst = const.DIR_SINTETIC_RAW_DATA_HEADER
        else:
            dir_src = const.DIR_RAW_DATA_TRANSFORM
            dir_dst = const.DIR_RAW_DATA_HEADER

        if not os.path.exists(dir_src):
            self.transform_raw_data()

        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.__fill_data_structure()

        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        print("CREATE HEADER FILES....")
        filenames = listdir(dir_src)

        for file in filenames:
            if file.endswith(".csv"):
                if not self.sintetic:
                    current_file_data = file.split("_")
                    target = current_file_data[2]
                    user = current_file_data[1]
                full_current_file_path = os.path.join(dir_src, file)
                with open(full_current_file_path) as current_file:
                    full_current_file_path = os.path.join(dir_dst, file)
                    with open(full_current_file_path, "w") as file_header:
                        # write first line of file
                        header_line = ""
                        for x in range(0, len(self.header)):
                            if x == 0:  # time
                                header_line = self.header[0]
                            else:
                                header_line = header_line + "," + self.header[x]
                        header_line = header_line + ",target,user" + "\n"
                        file_header.write(header_line)
                        # write all other lines
                        j = -1
                        for line in current_file:
                            j += 1
                            line_data = line.split(",")
                            # first element time
                            new_line_data = {0: line_data[0]}
                            sensor_c = line_data[1]
                            if self.sintetic:
                                user = line_data[3]
                                target = line_data[4]
                                target = target.replace("\n", "")
                            pos = self.__range_position_in_header(sensor_c)
                            # others elements all -1 except elements in range between pos[0] and pos[1]
                            curr_line_data = 2
                            for x in range(1, len(self.header)):  # x is the offset in list new_line_data
                                if x in range(pos[0], pos[1] + 1):
                                    if curr_line_data < len(line_data):
                                        if "\n" not in line_data[curr_line_data]:
                                            if "-Infinity" in line_data[curr_line_data]:
                                                new_line_data[x] = ""
                                            else:
                                                new_line_data[x] = line_data[curr_line_data]
                                        else:
                                            if "-Infinity" in line_data[curr_line_data]:
                                                new_line_data[x] = ""
                                            else:
                                                new_line_data[x] = line_data[curr_line_data].split("\n")[0]
                                        curr_line_data += 1
                                    else:
                                        new_line_data[x] = ""
                                else:
                                    new_line_data[x] = ""
                            new_line = ""
                            for x in range(0, len(new_line_data)):
                                if x == 0:
                                    new_line = new_line_data[0]
                                else:
                                    new_line = new_line + "," + new_line_data[x]
                            new_line = new_line + "," + str(target) + "," + str(user) + "\n"
                            file_header.write(new_line)
            elif file.endswith(".json"):
                shutil.copyfile(os.path.join(dir_src, file), os.path.join(dir_dst, file))
        print("END HEADER FILES....")

    # fill directory with all file consistent with the featured header divided in time window
    def __create_time_files(self):
        if self.sintetic:
            dir_src = const.DIR_SINTETIC_RAW_DATA_HEADER
            dir_dst = const.DIR_SINTETIC_RAW_DATA_FEATURES
        else:
            dir_src = const.DIR_RAW_DATA_HEADER
            dir_dst = const.DIR_RAW_DATA_FEATURES

        # create files with header if not exist
        if not os.path.exists(dir_src):
            self.create_header_files()
        if len(self.header) == 0 or len(self.header_with_features) == 0:
            self.__fill_data_structure()
        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        print("DIVIDE FILES IN TIME WINDOWS AND COMPUTE FEATURES....")
        # build string header
        header_string = ""
        for i in self.header_with_features:
            header_string = header_string + self.header_with_features[i] + ","
        header_string = header_string[:-1]
        header_string += ",target,user\n"

        # compute window dimension
        window_dim = int(const.SAMPLE_FOR_SECOND * const.WINDOW_DIMENSION)

        # loop on header files
        filenames = listdir(dir_src)
        for current_file in filenames:
            if current_file.endswith("csv"):
                if not self.sintetic:
                    current_tm = current_file.split("_")[2]
                    current_user = current_file.split("_")[1]

                source_file_path = os.path.join(dir_src, current_file)
                df_file = pd.read_csv(source_file_path, dtype=const.DATASET_DATA_TYPE)

                featureNames = [col for col in df_file.columns if
                                col != 'target' and col != 'user' and col != 'time' and col != 'activityrecognition#0' and col != 'activityrecognition#1']
                # max time in source file
                #print(current_file)

                end_time = df_file.loc[df_file['time'].idxmax()]['time']
                #print(end_time)
                destination_file_path = os.path.join(dir_dst, current_file)
                destination_file = open(destination_file_path, 'w')
                destination_file.write(header_string)

                start_current = 0
                i = 0

                # track previuos value, if no value are present for a windows use previous
                previous_mean = []
                previous_min = []
                previous_max = []
                previous_std = []
                previous_activityRec = ""
                previous_activityRecProba = ""

                # loop on windows in file
                while True:

                    # current value for features
                    current_mean = []
                    current_min = []
                    current_max = []
                    current_std = []

                    # define time range
                    end_current = start_current + window_dim
                    if end_time <= end_current:
                        range_current = list(range(start_current, end_time, 1))
                        start_current = end_time
                    else:
                        range_current = list(range(start_current, end_current, 1))
                        start_current = end_current
                    # df of the current time window
                    df_current = df_file.loc[df_file['time'].isin(range_current)]
                    nfeature = 0
                    if self.sintetic:
                        if df_current.loc[:, "target"].size > 0:
                            df_current_tm = df_current.loc[:, "target"]
                            current_user = df_current.loc[:, "user"].iloc[0]
                            equal = True
                            for tm in range(0,df_current_tm.size-1,1):
                                if not df_current_tm.iloc[tm] == df_current_tm.iloc[tm+1]:
                                    equal = False
                                    break

                            if equal == False:
                                continue
                            else:
                                current_tm = df_current_tm.iloc[0]

                    currentLine = ""
                    for feature in featureNames:
                        currentFeatureSerie = df_current[feature]
                        currentMean = currentFeatureSerie.mean(skipna=True)
                        currentMin = currentFeatureSerie.min(skipna=True)
                        currentMax = currentFeatureSerie.max(skipna=True)
                        currentStd = currentFeatureSerie.std(skipna=True)
                        if i == 0:
                            previous_mean.append(str(currentMean))
                            current_mean.append(str(currentMean))
                            previous_min.append(str(currentMin))
                            current_min.append(str(currentMin))
                            previous_max.append(str(currentMax))
                            current_max.append(str(currentMax))
                            previous_std.append(str(currentStd))
                            current_std.append(str(currentStd))
                        else:
                            if str(currentMean) == 'nan':
                                current_mean.append(str(previous_mean[nfeature]))
                            else:
                                current_mean.append(str(currentMean))
                            if str(currentMin) == 'nan':
                                current_min.append(str(previous_min[nfeature]))
                            else:
                                current_min.append(str(currentMin))
                            if str(currentMax) == 'nan':
                                current_max.append(str(previous_max[nfeature]))
                            else:
                                current_max.append(str(currentMax))
                            if str(currentStd) == 'nan':
                                current_std.append(str(previous_std[nfeature]))
                            else:
                                current_std.append(str(currentStd))
                            currentLine = currentLine + str(current_mean[nfeature]) + ","
                            currentLine = currentLine + str(current_min[nfeature]) + ","
                            currentLine = currentLine + str(current_max[nfeature]) + ","
                            currentLine = currentLine + str(current_std[nfeature]) + ","
                            nfeature += 1
                    if df_current.shape[0] > 0:
                        # select 'activityrecognition#0' and 'activityrecognition#1' from  df_current
                        df_current_google = df_current[['activityrecognition#0', 'activityrecognition#1']]
                        df_current_google = df_current_google[df_current_google['activityrecognition#1'] >= 0]
                        current_values = []
                        if df_current_google.shape[0] == 0:
                            current_values.append(previous_activityRec)
                            current_values.append(previous_activityRecProba)
                        else:
                            if df_current_google.shape[0] == 1:
                                df_row = df_current_google
                                current_values.append(df_row['activityrecognition#0'].item())
                                current_values.append(df_row['activityrecognition#1'].item())
                                previous_activityRec = ""
                                previous_activityRecProba = ""
                            else:
                                # pick prediction with max probability to be correct
                                activity0 = df_current_google.loc[df_current_google['activityrecognition#1'].idxmax()][
                                    'activityrecognition#0']
                                activity1 = df_current_google.loc[df_current_google['activityrecognition#1'].idxmax()][
                                    'activityrecognition#1']
                                current_values.append(activity0)
                                current_values.append(activity1)
                                previous_activityRec = activity0
                                previous_activityRecProba = activity1

                    previous_mean = list(current_mean)
                    previous_min = list(current_min)
                    previous_max = list(current_max)
                    previous_std = list(current_std)

                    if len(currentLine) > 2:
                        line = str(i) + "," + str(current_values[0]) + "," + str(current_values[1]) + "," + currentLine[
                                                                                                            :-1]
                        line = line + "," + str(current_tm) + "," + str(current_user) + "\n"
                        destination_file.write(line)
                    i += 1
                    if start_current == end_time:
                        break
        print("END DIVIDE FILES IN TIME WINDOWS AND COMPUTE FEATURES......")

    # create dataset file
    def __create_dataset(self):
        if self.sintetic:
            dir_src = const.DIR_SINTETIC_RAW_DATA_FEATURES
            dir_dst = const.DIR_SINTETIC_DATASET
            file_dst = const.SINTETIC_FILE_DATASET
        else:
            dir_src = const.DIR_RAW_DATA_FEATURES
            dir_dst = const.DIR_DATASET
            file_dst = const.FILE_DATASET

        # create files with time window if not exsist
        if not os.path.exists(dir_src):
            self.__create_time_files()

        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)
        else:
            shutil.rmtree(dir_dst)
            os.makedirs(dir_dst)

        filenames = listdir(dir_src)

        result_file_path = os.path.join(dir_dst, file_dst)
        with open(result_file_path, 'w') as result_file:
            j = 0
            for file in filenames:
                if file.endswith(".csv"):
                    current_file_path = os.path.join(dir_src, file)
                    with open(current_file_path) as current_file:  # tm file
                        i = 0
                        for line in current_file:
                            # if the current line is not the first, the header
                            if i != 0:
                                result_file.write(line)
                            else:
                                if j == 0:
                                    result_file.write(line)
                                i += 1
                        j += 1
                        # split data

    # splid passed dataframe into test, train and cv
    def __split_dataset(self, df):
        if self.sintetic:
            dir_src = const.DIR_SINTETIC_DATASET
            file_training_dst = const.SINTETIC_FILE_TRAINING
            file_test_dst = const.SINTETIC_FILE_TEST
            file_cv_dst = const.SINTETIC_FILE_CV
        else:
            dir_src = const.DIR_DATASET
            file_training_dst = const.FILE_TRAINING
            file_test_dst = const.FILE_TEST
            file_cv_dst = const.FILE_CV

        training, cv, test = util.split_data(df, train_perc=const.TRAINING_PERC, cv_perc=const.CV_PERC,
                                        test_perc=const.TEST_PERC)
        training.to_csv(dir_src + '/' + file_training_dst, index=False)
        test.to_csv(dir_src + '/' + file_test_dst, index=False)
        cv.to_csv(dir_src + '/' + file_cv_dst, index=False)

    # clean files and transform in orientation independent
    def preprocessing_files(self):
        print("START PREPROCESSING...")
        self.clean_files()
        self.transform_raw_data()

    # for each sensors analyze user support
    # put support result in sensor_support.csv [sensor,nr_user,list_users,list_classes]
    def analyze_sensors_support(self):
        if not os.path.exists(const.DIR_RAW_DATA_CORRECT):
            print("You should pre-processing files first!")
            return -1
        if len(self.users) == 0 or len(self.sensors) == 0 or len(self.tm) == 0:
            self.__fill_data_structure()
        # build data frame for user support
        columns = ['sensor', 'nr_user', 'list_users', 'list_classes']
        index = list(range(len(self.sensors)))
        df_sensor_analysis = pd.DataFrame(index=index, columns=columns)
        df_sensor_analysis['sensor'] = self.sensors
        filenames = listdir(const.DIR_RAW_DATA_CORRECT)
        n_users = []
        users_list = []
        classes_list = []
        for s in self.sensors:
            class_list = []
            user_list = []
            for file in filenames:
                if file.endswith(".csv"):
                    data = file.split("_")
                    f = open(os.path.join(const.DIR_RAW_DATA_CORRECT, file))
                    if data[2] not in class_list:
                        class_list.append(data[2])
                    reader = csv.reader(f, delimiter=",")
                    for row in reader:
                        if row[1] == s and data[1] not in user_list:
                            user_list.append(data[1])
                    f.close()
            n_users.append(len(user_list))
            index = df_sensor_analysis[df_sensor_analysis['sensor'] == s].index.tolist()
            df_sensor_analysis.ix[index, 'list_users'] = str(user_list)
            df_sensor_analysis.ix[index, 'list_classes'] = str(class_list)
            users_list.append(str(user_list))
            classes_list.append(str(class_list))
        df_sensor_analysis['nr_user'] = n_users
        df_sensor_analysis['list_users'] = users_list
        df_sensor_analysis['list_classes'] = classes_list

        df_sensor_analysis = df_sensor_analysis.sort_values(by=['nr_user'], ascending=[False])

        # remove result file if exists
        try:
            os.remove(const.FILE_SUPPORT)
        except OSError:
            pass
        df_sensor_analysis.to_csv(const.FILE_SUPPORT, index=False)

    # analyze dataset composition in term of class and user contribution fill balance_time
    # with minimum number of window for transportation mode
    def create_balanced_dataset(self, sintetic = False):
        self.sintetic = sintetic
        # create dataset from files
        self.__create_dataset()

        if self.sintetic:
            dir_src = const.DIR_SINTETIC_DATASET
            file_src = const.SINTETIC_FILE_DATASET
            file_dst = const.SINTETIC_FILE_DATASET_BALANCED
        else:
            dir_src = const.DIR_DATASET
            file_src = const.FILE_DATASET
            file_dst = const.FILE_DATASET_BALANCED

        # study dataset composition to balance
        if not os.path.exists(dir_src):
            self.__create_dataset()
        if len(self.users) == 0 or len(self.sensors) == 0 or len(self.tm) == 0:
            self.__fill_data_structure()
        print("START CREATE BALANCED DATASET....")

        df = pd.read_csv(dir_src + "/" + file_src)
        min_windows = df.shape[0]

        for t in self.tm:  # loop on transportation mode
            df_t = df.loc[df['target'] == t]
            # h, m, s = time_for_rows_number(df_t.shape[0])
            if df_t.shape[0] <= min_windows:
                min_windows = df_t.shape[0]

        target_df = df.groupby(['target', 'user']).agg({'target': 'count'})
        target_df['percent'] = target_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
        target_df.loc[:, 'num'] = target_df.apply(lambda row: util.to_num(row, min_windows), axis=1)

        self.balance_time = min_windows

        # create balanced dataset
        dataset_incremental = pd.DataFrame(columns=df.columns)
        for index, row in target_df.iterrows():
            current_df = df.loc[(df['user'] == index[1]) & (df['target'] == index[0])]
            if current_df.shape[0] == row['num']:
                # put in new dataset
                dataset_incremental = pd.concat([dataset_incremental, current_df])
            else:
                # select num rows to put in new dataset
                df_curr = current_df.sample(n=int(row['num']))
                dataset_incremental = pd.concat([dataset_incremental, df_curr])
        dataset_incremental.to_csv(dir_src + '/' + file_dst, index=False)
        self.__split_dataset(dataset_incremental)
        print("END CREATE BALANCED DATASET....")

    @property
    def get_train(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_TRAINING)

    @property
    def get_test(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_TEST)

    @property
    def get_cv(self):
        return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_CV)

    @property
    def get_dataset(self):
        if const.SINTETIC_LEARNING:
            return pd.read_csv(const.DIR_SINTETIC_DATASET + "/" + const.SINTETIC_FILE_DATASET_BALANCED)
        else:
            return pd.read_csv(const.DIR_DATASET + "/" + const.FILE_DATASET_BALANCED)

    # return list of excluded sensor based on the correspondent classification level
    def get_excluded_sensors(self, sensors_set):
        excluded_sensors = []
        if sensors_set == 1:
            excluded_sensors = const.sensor_to_exclude_first
        if sensors_set == 2:
            excluded_sensors = const.sensor_to_exclude_second
        if sensors_set == 3:
            excluded_sensors = const.sensors_to_exclude_third
        return excluded_sensors

    # return list of considered sensors based on the correspondent classification level
    def get_remained_sensors(self, sensors_set):
        excluded_sensors = self.get_excluded_sensors(sensors_set)
        remained_sensors = []
        for s in self.get_sensors:
            if s not in excluded_sensors:
                remained_sensors.append(s)
        return remained_sensors

    def get_sensors_set_features(self, sensors_set):
        feature_to_delete = []
        header = self.get_header
        for s in self.get_excluded_sensors(sensors_set):
            for x in header.values():
                if s in x:
                    feature_to_delete.append(x)
        features_list = (set(header.values()) - set(feature_to_delete))
        return features_list

    def get_sensor_features(self, sensor):
        feature_sensor = []
        header = self.get_header
        for x in header.values():
            if sensor in x:
                feature_sensor.append(x)
        return feature_sensor

if __name__ == "__main__":
	dataset = TMDataset()
	dataset.create_balanced_dataset()
