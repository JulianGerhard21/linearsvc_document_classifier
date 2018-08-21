# scikit.learn LinearSVC implementation for classifying labeled documents

## Installation

Before running the specified python files, please use:
```
pip3 install -r requirements.txt
```

assuming that you have both python2 and python3 installed. Otherwise use:
```
pip install -r requirements.txt
```

## Usage

To plot and print the confusion matrix on the dataset, just use:
```
python3 classify_lsvc.py
```

assuming that you have both python2 and python3 installed. Otherwise use:
```
python classify_lsvc.py
```

Keep in mind, that you might have to close the plot for proceeding to print
the confusion matrix.

### Use your own dataset

If you want to use your own dataset, just create a file with the following data-format:
```
label document
label document
label document
```

where *label* corresponds to the class, the document was previously given and *document* represents
a given text. Keep in mind, that your classes should not be over- or underrepresented. 

## Evaluation

The performance on the given sample dataset is pretty okay, regarding the fact
that there is no preprocessing like lemmatizing or stopword removal.

                 precision    recall  f1-score   support

        arrival       0.87      0.73      0.80       109
      departure       0.79      0.64      0.71        87
      greetings       0.96      0.98      0.97        44
         return       0.80      1.00      0.89        16
       schedule       0.90      0.88      0.89        97
        support       0.79      0.85      0.81        13
           trip       0.84      0.95      0.90       243
             ts       0.93      0.81      0.87        16

    avg / total       0.86      0.86      0.85       625


## Author

**Julian Gerhard**