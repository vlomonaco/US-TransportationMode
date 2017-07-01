import const
from random import shuffle
from sklearn import preprocessing


# return time in hours, minutes and seconds from number of rows
def time_for_rows_number(rows_number):
    tot_shape = round(rows_number, 2)
    tot_h = int((tot_shape / (60 / const.WINDOW_DIMENSION)) / 60)
    tot_m = int((tot_shape / (60 / const.WINDOW_DIMENSION)) - (tot_h * 60))
    tot_s = int((tot_shape * const.WINDOW_DIMENSION) - (tot_h * 60 * 60) - (tot_m * 60))
    return tot_h, tot_m, tot_s


# return the number of row to return to be balanced with min_windows
def to_num(row, min_windows):
    return round((float(min_windows) / 100.00) * row['percent'], 0)


def split_data(df, train_perc=0.8, cv_perc=0.0, test_perc=0.2):
    assert train_perc + cv_perc + test_perc == 1.0
    # create random list of indices
    N = len(df)
    l = list(range(N))
    shuffle(l)
    # get splitting indicies
    trainLen = int(N * train_perc)
    cvLen = int(N * cv_perc)
    testLen = int(N * test_perc)
    # get training, cv, and test sets
    training = df.iloc[l[:trainLen]]
    cv = df.iloc[l[trainLen:trainLen + cvLen]]
    test = df.iloc[l[trainLen + cvLen:]]
    return training, cv, test


def average(l):
    return sum(l) * 1.0 / len(l)


def fill_nan_with_mean_training(training, test):
    trainingFill = training.copy()
    testFill = test.copy()
    trainingFill = trainingFill.fillna(trainingFill.mean())
    trainingFill = trainingFill.fillna(0)
    testFill = testFill.fillna(trainingFill.mean())
    testFill = testFill.fillna(0)
    return trainingFill, testFill

def scale_features(train, test):
    # build scaler to apply on training and test the same transformation
    scaler = preprocessing.StandardScaler().fit(train)
    train_features_scaled = scaler.transform(train)
    test_features_scaled = scaler.transform(test)
    return train_features_scaled, test_features_scaled
