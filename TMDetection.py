from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import itertools
import math

from TMDataset import TMDataset
import const
import util


class TMDetection:
    dataset = TMDataset()
    classes = []
    classes2string = {}
    classes2number = {}

    def __init__(self):
        if not const.HAVE_DT:
            self.dataset.create_balanced_dataset(const.SINTETIC_FILE_TRAINING)
        classes_dataset = self.dataset.get_dataset['target'].values
        print(classes_dataset)
        for i, c in enumerate(sorted(set(classes_dataset))):
            self.classes2string[i] = c
            self.classes2number[c] = i
            self.classes.append(c)

    def __get_sets_for_classification(self, df_train, df_test, features):
        train, test = util.fill_nan_with_mean_training(df_train, df_test)
        train_features = train[features].values
        train_classes = [self.classes2number[c] for c in train['target'].values]
        test_features = test[features].values
        test_classes = [self.classes2number[c] for c in test['target'].values]
        return train_features, train_classes, test_features, test_classes

    # decision tree algorithm training on training al train set and test on all test set
    def decision_tree(self, sensors_set):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("DECISION TREE.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))
        train_features, train_classes, test_features, test_classes = self.__get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features)
        classifier_decision_tree = tree.DecisionTreeClassifier()
        classifier_decision_tree.fit(train_features, train_classes)
        test_prediction = classifier_decision_tree.predict(test_features)
        acc = accuracy_score(test_classes, test_prediction)
        df_feature = pd.DataFrame(
            {'accuracy': acc, 'features': features, 'importance': classifier_decision_tree.feature_importances_})
        df_feature = df_feature.sort_values(by='importance', ascending=False)
        print("ACCURACY : " + str(acc))
        print("END TREE")

        if not os.path.exists(const.DIR_RESULTS):
            os.makedirs(const.DIR_RESULTS)
        df_feature.to_csv(const.DIR_RESULTS + "/" + str(sensors_set) + const.FILE_DECISION_TREE_RESULTS, index=False)

    # random forest algorithm training on training al train set and test on all test set
    def random_forest(self, sensors_set):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("RANDOM FOREST.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))
        train_features, train_classes, test_features, test_classes = self.__get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features)
        classifier_forest = RandomForestClassifier(n_estimators=const.PAR_RF_ESTIMATOR)
        classifier_forest.fit(train_features, train_classes)
        test_prediction = classifier_forest.predict(test_features)
        acc = accuracy_score(test_classes, test_prediction)
        df_feature = pd.DataFrame(
            {'accuracy': acc, 'featureName': features, 'importance': classifier_forest.feature_importances_})
        df_feature = df_feature.sort_values(by='importance', ascending=False)
        print("ACCURACY : " + str(acc))
        print("END RANDOM FOREST")

        if not os.path.exists(const.DIR_RESULTS):
            os.makedirs(const.DIR_RESULTS)
        df_feature.to_csv(const.DIR_RESULTS + "/" + str(sensors_set) + const.FILE_RANDOM_FOREST_RESULTS, index=False)

    # neural network algorithm training on training al train set and test on all test set
    def neural_network(self, sensors_set):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("NEURAL NETWORK.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))
        train_features, train_classes, test_features, test_classes = self.__get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features)
        train_features_scaled, test_features_scaled = util.scale_features(train_features, test_features)

        classifier_nn = MLPClassifier(hidden_layer_sizes=(const.PAR_NN_NEURONS[sensors_set],),
                                      alpha=const.PAR_NN_ALPHA[sensors_set], max_iter=const.PAR_NN_MAX_ITER,
                                      tol=const.PAR_NN_TOL)
        classifier_nn.fit(train_features_scaled, train_classes)
        test_prediction = classifier_nn.predict(test_features_scaled)
        acc = accuracy_score(test_classes, test_prediction)
        print("ACCURACY : " + str(acc))
        print("END NEURAL NETWORK")

        if not os.path.exists(const.DIR_RESULTS):
            os.makedirs(const.DIR_RESULTS)
        file_content = "acc\n" + str(acc)
        with open(const.DIR_RESULTS + "/" + str(sensors_set) + const.FILE_NEURAL_NETWORK_RESULTS, 'w') as f:
            f.write(file_content)

    # support vector machine algorithm training on training al train set and test on all test set
    def support_vector_machine(self, sensors_set):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("SUPPORT VECTOR MACHINE.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))
        train_features, train_classes, test_features, test_classes = self.__get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features)
        train_features_scaled, test_features_scaled = util.scale_features(train_features, test_features)

        classifier_svm = SVC(C=const.PAR_SVM_C[sensors_set], gamma=const.PAR_SVM_GAMMA[sensors_set], verbose=False)
        classifier_svm.fit(train_features_scaled, train_classes)
        test_prediction = classifier_svm.predict(test_features_scaled)
        acc = accuracy_score(test_classes, test_prediction)
        print("ACCURACY : " + str(acc))
        print("END SUPPORT VECTOR MACHINE.....")

        if not os.path.exists(const.DIR_RESULTS):
            os.makedirs(const.DIR_RESULTS)
        file_content = "acc\n" + str(acc)
        with open(const.DIR_RESULTS + "/" + str(sensors_set) + const.FILE_SUPPORT_VECTOR_MACHINE_RESULTS, 'w') as f:
            f.write(file_content)

    # use different algorithms changing target classes, try all combination of two target classes
    def classes_combination(self, sensors_set):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        class_combination = list(itertools.combinations(self.classes, 2))
        train = self.dataset.get_train.copy()
        test = self.dataset.get_test.copy()
        if not os.path.exists(const.DIR_RESULTS):
            os.makedirs(const.DIR_RESULTS)
        with open(const.DIR_RESULTS + "/" + str(sensors_set) + const.FILE_TWO_CLASSES_COMBINATION, 'w') as f:
            f.write("combination, algorithm, accuracy")
            for combination in class_combination:
                cc_train = train.loc[(train['target'] == combination[0]) | (train['target'] == combination[1])]
                cc_test = test.loc[(test['target'] == combination[0]) | (test['target'] == combination[1])]
                train_features, train_classes, test_features, test_classes = self.__get_sets_for_classification(
                    cc_train, cc_test, features)

                # buil all classifier
                classifier_tree = tree.DecisionTreeClassifier()
                classifier_forest = RandomForestClassifier(n_estimators=const.PAR_RF_ESTIMATOR)
                classifier_nn = MLPClassifier(hidden_layer_sizes=(const.PAR_NN_NEURONS[sensors_set],),
                                              alpha=const.PAR_NN_ALPHA[sensors_set], max_iter=const.PAR_NN_MAX_ITER,
                                              tol=const.PAR_NN_TOL)
                classifier_svm = SVC(C=const.PAR_SVM_C[sensors_set], gamma=const.PAR_SVM_GAMMA[sensors_set],
                                     verbose=False)

                # train all classifier
                classifier_tree.fit(train_features, train_classes)
                classifier_forest.fit(train_features, train_classes)
                classifier_nn.fit(train_features, train_classes)
                classifier_svm.fit(train_features, train_classes)

                # use classifier on test set
                test_prediction_tree = classifier_tree.predict(test_features)
                test_prediction_forest = classifier_forest.predict(test_features)
                test_prediction_nn = classifier_nn.predict(test_features)
                test_prediction_svm = classifier_svm.predict(test_features)

                # evaluate classifier
                acc_tree = accuracy_score(test_classes, test_prediction_tree)
                acc_forest = accuracy_score(test_classes, test_prediction_forest)
                acc_nn = accuracy_score(test_classes, test_prediction_nn)
                acc_svm = accuracy_score(test_classes, test_prediction_svm)

                # print result
                print(str(combination))
                print("DECISION TREE : ", str(acc_tree))
                f.write(str(combination) + ", DT ," + str(acc_tree) + "\n")
                print("RANDOM FOREST : ", str(acc_forest))
                f.write(str(combination) + ", RF ," + str(acc_forest) + "\n")
                print("NEURAL NETWORK : ", str(acc_nn))
                f.write(str(combination) + ", NN ," + str(acc_nn) + "\n")
                print("SUPPORT VECTOR MACHINE : ", str(acc_svm))
                f.write(str(combination) + ", SVM ," + str(acc_svm) + "\n")

    # use different algorithms leaving one subject out from training and testing only on this subject -
    # considering all classes in dataset and only user classes
    def leave_one_subject_out(self, sensors_set):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        train = self.dataset.get_train.copy()
        test = self.dataset.get_test.copy()
        if not os.path.exists(const.DIR_RESULTS):
            os.makedirs(const.DIR_RESULTS)
        with open(const.DIR_RESULTS + "/" + str(sensors_set) + const.FILE_LEAVE_ONE_SUBJECT_OUT, 'w') as f:
            f.write(
                "user, classes, complete training examples, user class training examples, test examples, algorithm, acc all classes, acc user classes\n")
            for u in self.dataset.get_users:
                user_train = train.loc[(train['user'] != u)]
                user_test = test.loc[(test['user'] == u)]
                train_features, train_classes, test_features, test_classes = self.__get_sets_for_classification(
                    user_train, user_test, features)

                # buil all classifier
                classifier_tree = tree.DecisionTreeClassifier()
                classifier_forest = RandomForestClassifier(n_estimators=const.PAR_RF_ESTIMATOR)
                classifier_nn = MLPClassifier(hidden_layer_sizes=(const.PAR_NN_NEURONS[sensors_set],),
                                              alpha=const.PAR_NN_ALPHA[sensors_set], max_iter=const.PAR_NN_MAX_ITER,
                                              tol=const.PAR_NN_TOL)
                classifier_svm = SVC(C=const.PAR_SVM_C[sensors_set], gamma=const.PAR_SVM_GAMMA[sensors_set],
                                     verbose=False)

                # train all classifier
                classifier_tree.fit(train_features, train_classes)
                classifier_forest.fit(train_features, train_classes)
                classifier_nn.fit(train_features, train_classes)
                classifier_svm.fit(train_features, train_classes)

                # use classifier on test set
                test_prediction_tree = classifier_tree.predict(test_features)
                test_prediction_forest = classifier_forest.predict(test_features)
                test_prediction_nn = classifier_nn.predict(test_features)
                test_prediction_svm = classifier_svm.predict(test_features)

                # evaluate classifier
                acc_tree = accuracy_score(test_classes, test_prediction_tree)
                acc_forest = accuracy_score(test_classes, test_prediction_forest)
                acc_nn = accuracy_score(test_classes, test_prediction_nn)
                acc_svm = accuracy_score(test_classes, test_prediction_svm)

                user_classes = []

                acc_class_tree = acc_tree
                acc_class_forest = acc_forest
                acc_class_nn = acc_nn
                acc_class_svm = acc_svm
                user_class_train = user_train

                for i, c in enumerate(sorted(set(user_test['target'].values))):
                    user_classes.append(c)
                # if user don't have collect all classes we need to calculate different acc with training
                # composed only by user classes
                if len(user_classes) != len(self.dataset.get_tm):
                    if len(user_classes) != 1:
                        user_class_train = user_train.loc[user_train['target'].isin(user_classes)]
                        train_class_features, train_class_classes, test_features, test_classes = self.__get_sets_for_classification(
                            user_class_train,
                            user_test,
                            features)
                        # train all classifier
                        classifier_tree.fit(train_class_features, train_class_classes)
                        classifier_forest.fit(train_class_features, train_class_classes)
                        classifier_nn.fit(train_class_features, train_class_classes)
                        classifier_svm.fit(train_class_features, train_class_classes)

                        # use classifier on test set
                        test_prediction_tree = classifier_tree.predict(test_features)
                        test_prediction_forest = classifier_forest.predict(test_features)
                        test_prediction_nn = classifier_nn.predict(test_features)
                        test_prediction_svm = classifier_svm.predict(test_features)

                        # evaluate classifier
                        acc_class_tree = accuracy_score(test_classes, test_prediction_tree)
                        acc_class_forest = accuracy_score(test_classes, test_prediction_forest)
                        acc_class_nn = accuracy_score(test_classes, test_prediction_nn)
                        acc_class_svm = accuracy_score(test_classes, test_prediction_svm)
                    else:
                        acc_class_tree = 1
                        acc_class_forest = 1
                        acc_class_nn = 1
                        acc_class_svm = 1

                pre = str(u) + "," + str(' '.join(user_classes)) + "," + str(user_train.shape[0]) + "," + str(
                    user_class_train.shape[0]) + "," + str(user_test.shape[0])
                f.write(pre + ", DT ," + str(acc_tree) + "," + str(acc_class_tree) + "\n")
                f.write(pre + ", RF ," + str(acc_forest) + "," + str(acc_class_forest) + "\n")
                f.write(pre + ", NN ," + str(acc_nn) + "," + str(acc_class_nn) + "\n")
                f.write(pre + ", SVM ," + str(acc_svm) + "," + str(acc_class_svm) + "\n")

    # use feature relative to one sensor to build model and evaluate
    def single_sensor_accuracy(self):
        sensor = []
        accuracy = []
        std = []
        for s in self.dataset.get_sensors:
            if s != "activityrecognition":
                print(s)
                features = self.dataset.get_sensor_features(s)
                train = self.dataset.get_train.copy()
                test = self.dataset.get_test.copy()

                train_features, train_classes, test_features, test_classes = self.__get_sets_for_classification(train,
                                                                                                                test,
                                                                                                                features)
                singleAcc = []
                for i in range(const.REPEAT):
                    # build classifier
                    classifier_forest = RandomForestClassifier(n_estimators=const.PAR_RF_ESTIMATOR)
                    classifier_forest.fit(train_features, train_classes)
                    test_prediction_forest = classifier_forest.predict(test_features)
                    acc_forest = accuracy_score(test_classes, test_prediction_forest)
                    singleAcc.append(acc_forest)
                accM = util.average(singleAcc)
                variance = list(map(lambda x: (x - accM) ** 2, singleAcc))
                standard_deviation = math.sqrt(util.average(variance))
                print(s, accM, standard_deviation)

                accuracy.append(accM)
                std.append(standard_deviation)
                sensor.append(s)
        df_single_sensor_acc = pd.DataFrame({'sensor': sensor, 'accuracy': accuracy, 'dev_standard': std})
        df_single_sensor_acc = df_single_sensor_acc.sort_values(by='accuracy', ascending=False)

        if not os.path.exists(const.DIR_RESULTS):
            os.makedirs(const.DIR_RESULTS)
        df_single_sensor_acc.to_csv(const.DIR_RESULTS + "/" + const.FILE_SINGLE_SENSOR_ANALYSIS, index=False)

if __name__ == "__main__":
	detection = TMDetection()

#	detection.decision_tree(1)
#	detection.decision_tree(2)
#	detection.decision_tree(3)

#	detection.random_forest(1)
#	detection.random_forest(2)
#	detection.random_forest(3)

#	detection.neural_network(1)
#	detection.neural_network(2)
#	detection.neural_network(3)

#	detection.support_vector_machine(1)
#	detection.support_vector_machine(2)
#	detection.support_vector_machine(3)

#	detection.classes_combination(1)
#	detection.classes_combination(2)
#	detection.classes_combination(3)

#	detection.leave_one_subject_out(1)
#	detection.leave_one_subject_out(2)
#	detection.leave_one_subject_out(3)

#	detection.single_sensor_accuracy()

