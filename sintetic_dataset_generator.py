import const
from TMDataset import TMDataset

from pprint import pprint
import os
import json
import random
import sys

class Generator:

    dataset = TMDataset()
    config = {}
    selected_samples = []
    samples_used = []
    extended_name = ""

    def __init__( self ):
        #load sintetic dataset's config
        self.config = self.__parse_json('sintetic_dataset_config.json')
        print(self.config)

        #if doesn't exists correct data create it
        if not os.path.exists(const.DIR_RAW_DATA_CORRECT):
            self.dataset.clean_files()
        else:
            print("CLEANED FILES ALREADY EXIST")

        self.__init_selected_samples()

        self.__build_sintetic_dataset()

        self.__create_config_copy()

    #take json configuration from a json file
    def __parse_json(self, path):
        with open(path) as data_file:
            data = json.load(data_file)
            return data

    #consider only file of samples inserted in the json config
    def __init_selected_samples(self):
        file_names = os.listdir(const.DIR_RAW_DATA_CORRECT)

        samples = self.config["samples"]
        samples_len = len(samples)
        first_sample = self.config["samples"][0]

        #check the number of samples you want to consider
        if not first_sample == "all":
            for file in file_names:
                file_sample_part = file.split("_")
                if file_sample_part[1] in samples:
                    self.selected_samples.insert(0,file)
        else:
            self.selected_samples = file_names

    #build the sintetic dataset
    def __build_sintetic_dataset(self):
        if not os.path.exists(const.DIR_SINTETIC_RAW_DATASET):
            os.makedirs(const.DIR_SINTETIC_RAW_DATASET)

        activities = []
        for activity in self.config["activities"]:
            activities.insert(len(activities), activity)

        print("ATTIVITA NEL FILE DI CONFIGURAZIONE  ")
        pprint(activities)
        print("\n")

        res_file_name = self.config["name"] + "_dataset"

        #check if extension of th file is defined
        if not self.config["name"].endswith(".csv"):
            res_file_name += ".csv"

        #check if name of file already exist
            #in this case it rename the file adding some digits
        while os.path.isfile(os.path.join(const.DIR_SINTETIC_RAW_DATASET, res_file_name)):
            print("FILE CALL " + res_file_name + " ALREADY EXIST")
            arr = list(res_file_name)
            rnd_num_name = str(random.randrange(0,9))
            self.extended_name += rnd_num_name
            arr.insert(len(arr)-12, rnd_num_name)
            res_file_name = ''.join(arr)
            print("NEW FILE NAME IS " + res_file_name)

        res_file_path = os.path.join(const.DIR_SINTETIC_RAW_DATASET, res_file_name)

        total_duration = 0

        #open the result file
        with open(res_file_path, "w") as file_result:
            print("WRITE IN " + res_file_name)
            #for every activity specified
            for activity_dict in activities:
                activity_list_files_sample = []
                used_list_files_sample = []
                current_activity_duration = 0

                #get name and duration of the activity
                activity = activity_dict.keys()[0]
                duration = int(activity_dict.values()[0])

                pprint("CURRENT TRANSPORTATION MODE: " + activity)

                #get the file correspondent to the specific activity
                for file in self.selected_samples:
                    file_sample_activity = file.split("_")[2]
                    if file_sample_activity == activity:
                        activity_list_files_sample.insert(len(activity_list_files_sample),file);

                if len(activity_list_files_sample) == 0:
                    os.remove(res_file_path)
                    sys.exit("NO FILE AVAILABLE FOR THE TRANSPORTATION MODE " + activity + " WITH THIS CONFIGURATION OF USERS")

                samples = self.__get_number_of_samples(activity_list_files_sample)
                n_of_samples = len(samples)
                duration_per_user = duration / n_of_samples
                #print(duration_per_user)

                duration = duration_per_user * n_of_samples
                #print(duration)

                duration_list = {}

                for sample in samples:
                    duration_list[sample] = duration_per_user
                    if sample not in self.samples_used:
                        self.samples_used.append(sample)

                #pprint(duration_list)

                end_of_activity = False

                #while it doesn't reach the duration specified in the config file
                while not end_of_activity:
                    current_file_duration = 0

                    #select random file from extracted ones
                    rnd_num = random.randrange(0,len(activity_list_files_sample))
                    current_file_name = activity_list_files_sample[rnd_num]
                    print("USING FILE " + current_file_name)

                    current_sample = current_file_name.split("_")[1]

                    source_file_path = os.path.join(const.DIR_RAW_DATA_CORRECT, current_file_name)

                    #open the random file
                    with open(source_file_path) as current_file:
                        #copy every line in the result file computing the correct timestamp
                        for line in current_file:
                            array_line = []
                            array_line = line.split(",")
                            timestamp_line = int(array_line[0])
                            current_file_duration = timestamp_line
                            array_line[len(array_line) - 1] =str(array_line[len(array_line)-1]).replace("\n","")
                            array_line.append(current_sample)
                            array_line.append(activity+"\n")
                            array_line[0] = str(total_duration + current_activity_duration + timestamp_line)
                            file_result.write(",".join(array_line))

                            #if duration is reached until the end of the file break
                            if current_activity_duration + timestamp_line > duration:
                                print("REACHED THE PREVIOUS DURATION OF THE TRANSPORTATION MODE " + activity)
                                total_duration += current_activity_duration + timestamp_line
                                end_of_activity = True
                                break

                            #check if user duration limit is reached
                            if int(timestamp_line) > int(duration_list[current_sample]):
                                #print("++++++"+str(timestamp_line))
                                #duration_list[current_sample] = 0
                                tmp_list = []
                                for name in activity_list_files_sample:
                                    if name.split("_")[1] != current_sample:
                                        tmp_list.insert(0,name)
                                activity_list_files_sample = tmp_list
                                #pprint(activity_list_files_sample)
                                break

                        current_activity_duration += current_file_duration
                        #pprint(timestamp_line)
                        duration_list[current_sample] = int(duration_list[current_sample]) - current_file_duration

                        pprint("FILE USED FOR " + str(current_file_duration) + "ms")
                        pprint(str(duration_list[current_sample]) + "ms REMAINED FOR SAMPLE " + current_sample);

                        #pprint(duration_list)

                        #check if repetition are requested
                        if self.config["repetition"] == "n":
                            if current_file_name in activity_list_files_sample:
                                activity_list_files_sample.remove(current_file_name)
                                #used_list_files_sample.insert(len(used_list_files_sample),current_file_name)
                            if len(activity_list_files_sample) == 0 and current_activity_duration < duration:
                                #os.remove(res_file_path)
                                sys.exit("INSUFFICIENT RECORDINGS TO CONSTRUCT A SYNTHETIC DATASET WITHOUT REPETITION WITH THIS CONFIGURATION")

    #create a copy of the config file correspondent to the created dataset
    def __create_config_copy(self):
        if self.config["samples"][0] == "all":
            self.config["samples_list"] = self.samples_used

        json_config = json.dumps(self.config, indent=4)
        with open(os.path.join(const.DIR_SINTETIC_RAW_DATASET, self.config["name"] + self.extended_name + "_dataset.json"), "w") as json_file:
            pprint(self.config["name"] + self.extended_name + "_dataset.json")
            json_file.write(json_config)

    def __get_number_of_samples(self, activity_samples_list):
        if not self.config["samples"][0] == "all":
            #pprint(len(self.config["samples"]))
            return self.config["samples"]
        else:
            samples_list = []
            for name in activity_samples_list:
                if not name.split("_")[1] in samples_list:
                    samples_list.insert(0,name.split("_")[1])
            #pprint(samples_list)
            #pprint(len(samples_list))
            return samples_list

if __name__ == "__main__":
    generator = Generator()
