# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics


def load_dataset(filename):
  """
  Loads the given Dataset into a Pandas Dataframe.
  :param filename:
  :return:
  """
  dataset = []

  with open(filename, 'r') as infile:
    for line in infile:
      label, document = line.strip().split(' ', 1)
      dataset.append((label, document))

  df = pd.DataFrame(data=dataset, columns=['label','document'])

  return df


def plot_labels_rel_documents(df):
  """
  :param df:
  :return:
  """
  fig = plt.figure(figsize=(8, 6))
  df.groupby('label').document.count().plot.bar(ylim=0)
  plt.show()


def preprocess_df(df_train):
  """
  Removes empty documents from dataset.
  :param df_train:
  :return:
  """

  df_train = df_train[pd.notnull(df_train['document'])]
  df_train.columns = ['label', 'document']
  df_train['label_id'] = df_train['label'].factorize()[0]
  df_train.head()
  return df_train


def train_lsvc_classifier(df_train):
  """
  Trains a LinearSVC Classifier which has proven to be best performing on given case.
  :param df_train:
  :return:
  """
  classifier = {}
  tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                          analyzer='char')
  features = tfidf.fit_transform(df_train.document).toarray()
  labels = df_train.label_id
  features.shape

  model = LinearSVC()
  X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df_train.index,
                                                                                   test_size=0.33, random_state=0)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  classifier['y_test'] = y_test
  classifier['y_pred'] = y_pred
  classifier['df_train'] = df_train

  return classifier


def classification_report(y_test, y_pred, df_train):
  return metrics.classification_report(y_test, y_pred, target_names=df_train['label'].unique())


if __name__ == '__main__':
  train = load_dataset('traintest')
  preprocessed_df = preprocess_df(train)
  plot_labels_rel_documents(preprocessed_df)
  classifier = train_lsvc_classifier(preprocessed_df)

  print(classification_report(classifier['y_test'], classifier['y_pred'], classifier['df_train']))
