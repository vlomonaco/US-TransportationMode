import urllib2
import os

# Constant
dataset_dir = './TransportationData'
datasetBalanced = dataset_dir + '/datasetBalanced'
rawOriginaldata = dataset_dir + '/_RawDataOriginal'
url_list = ['http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/5second/dataset_5secondWindow.csv',
            'http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/halfsecond/dataset_halfSecondWindow.csv']
            #'http://cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz']
dataset5second = 'dataset_5secondWindow.csv'
datasethalfsecond = 'dataset_halfSecondWindow.csv'


if __name__ == "__main__":
    # create folders
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(datasetBalanced):
        os.makedirs(datasetBalanced)

    if not os.path.exists(rawOriginaldata):
        os.makedirs(rawOriginaldata)

    print "DOWNLOAD........"
    for url in url_list:
        response = urllib2.urlopen(url)
        csv = response.read()
        if url == 'http://cs.unibo.it/projects/us-tm2017/static/dataset/extension/5second/dataset_5secondWindow.csv':
            outfile = datasetBalanced + '/' +dataset5second
        else:
            outfile = datasetBalanced + '/' + datasethalfsecond

        with open(outfile, 'wb') as f:
            f.write(csv)

    print "DOWNLOAD ENDED."