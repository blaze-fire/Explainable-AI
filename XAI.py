import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import itertools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud, STOPWORDS

from sklearn.preprocessing import OrdinalEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# To evaluate results locally 
def explain_local(df , model, target_col_name, num_cols=None,embedings=None):
  if 'linearsvc' in str(type(model)).lower() or 'linear_model' in str(type(model)).lower():
    
    df = df.reset_index(drop=True)
  
    print('\n \n Value Counts of target class \n{} \n\n'.format(df[target_col_name].value_counts()))
    
    if embedings != None:
      
        pred =  model.predict(embedings['embeddings_values'])
        y_true = df[target_col_name].values
        print('Accuracy Score : {}'.format(accuracy_score(y_true, pred)))
        word_freq = dict(zip(embedings['feature_names'], abs(np.sum(model.coef_, axis=0)/model.coef_.shape[0])))
        
        print('\n\n WordCloud of most important words in columns \n\n')


        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        min_font_size = 10).fit_words(word_freq)
        
        
        # plot the WordCloud image                       
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()


    else:
      pred = model.predict(df)
      y_true = df[target_col_name]
      print('Accuracy Score : {}'.format(accuracy_score(df[target_col_name], pred)))
      
      idx = np.where(y_true.values != pred)[0]
      temp = np.random.choice(idx,10)
    
      print('\nSome examples where your model misclassified \n\n')
      
      for i in temp:
          print("{} \n predicted : {}, \t true : {} \n\n".format(df[num_cols].iloc[i], pred[i], y_true.values[i]))
      print(classification_report(y_true, pred))
      
      d = classification_report(y_true, pred, output_dict=True)
      new_dict = {}
      
      for key in d:
        try:
          if int(key) in df[target_col_name].unique():
            new_dict[key] = d[key]
        except:
          pass
      
      for key in new_dict:
        
        if new_dict[key]['precision'] <= 0.2:
          print("Alert! your precison for target label {} is low (value : {}), supporting values {}".format(key, new_dict[key]['precision'], new_dict[key]['support']))
          count = np.count_nonzero(df[target_col_name] == int(key))
          print('Reason could be {} appears only {} times while length of dataframe is {}'.format(key, count, len(df)))
        if new_dict[key]['recall'] <= 0.2:
          print('Alert! your recall   for target label {} is low (value : {}), supporting values {}'.format(key, new_dict[key]['recall'], new_dict[key]['support']))
          count = np.count_nonzero(df[target_col_name] == int(key))
          print('Reason could be {} appears only {} times while length of dataframe is {}'.format(key, count, len(df)))
        
        print('\n\n')
      
      print('Plot for feature importances \n \n')
      
      x = pd.Series(abs(np.sum(model.coef_, axis=0)/model.coef_.shape[0]), index=num_cols).nlargest(len(num_cols))
      x.plot(kind='barh')
      
      plt.show()
      idx = x.index
      
      if len(x)>2:
        print('\n\nSome features with low feature importance are: \n')
      
        for i in range(1, len(x)):
          if x[0]/x[i] > 1e3:
            print("feature : ['{}'] , the reason could be its variance is only : {} compared to {} of most important feature ['{}'] ".format(idx[i], df[idx[i]].var(), df[idx[0]].var(), idx[0]))






# To evaluate results globallly 
def explain_global(df , model, target_col_name, training=True, test_size = 0.33):
  plt.figure(figsize=(10,6))
  sns.heatmap(df.isnull(), cmap='viridis', yticklabels=False)
  plt.show()

  def missing_predictions(y_test, pred):
    counter = 0
    missing_pred = []
    
    for i in y_test.unique():
      if i not in np.unique(pred):
        counter = 1
        missing_pred.append(i)
    
    if counter>0:
      return True, missing_pred
    
    return False, missing_pred
      
  
  def evaluate_result(df, y_test, target_col_name, d):
      new_dict = {}
      
      for key in d:
        try:
          if int(key) in y_test.unique():
            new_dict[key] = d[key]
        except:
          pass
      
      for key in new_dict:
        
        if new_dict[key]['precision'] <= 0.2:
          print("Alert! your precison for target label {} is low (value : {}), supporting values {}".format(key, new_dict[key]['precision'], new_dict[key]['support']))
          count = np.count_nonzero(df[target_col_name] == int(key))
          print('Reason could be {} appears only {} times while length of dataframe is {}'.format(key, count, len(df)))
        
        if new_dict[key]['recall'] <= 0.2:
          print('Alert! your recall   for target label {} is low (value : {}), supporting values {}'.format(key, new_dict[key]['recall'], new_dict[key]['support']))
          count = np.count_nonzero(df[target_col_name] == int(key))
          print('Reason could be {} appears only {} times while length of dataframe is {}'.format(key, count, len(df)))
        
        print('\n\n')
  
  def plot_feature_importance(model, columns, df):
    print('Plot for feature importances \n \n')
    x = pd.Series(abs(np.sum(model.coef_, axis=0)/model.coef_.shape[0]), index=columns).nlargest(len(columns))
    x.plot(kind='barh')
    plt.show()
    
    idx = x.index
    
    if len(x)>2:
      print('\n\nSome features with low feature importance are: \n')
      for i in range(1, len(x)):
        if x[0]/x[i] > 1e3:
          print("feature : ['{}'] , the reason could be its variance is only : {} compared to {} of most important feature ['{}'] ".format(idx[i], df[idx[i]].var(), df[idx[0]].var(), idx[0]))
    

  if training == False:
    cat_cols = df.select_dtypes(include = 'object').columns
    num_cols = df.drop(target_col_name, axis=1).select_dtypes(exclude = 'object').columns

    embed_cols = []
    for col in cat_cols: 
      if max(df[col].apply(lambda x: len(re.findall(r'\w+', str(x))))) > 2:
        embed_cols.append(col)
      

    if embed_cols:
      ordinal_encode_cols = [item for item in cat_cols if item not in embed_cols]
    
    else:
      ordinal_encode_cols = None
    
    
    X = df.drop(target_col_name, axis=1)
    X[num_cols] = X[num_cols].fillna(-1)
    X[cat_cols] = X[cat_cols].fillna('None')
    y = df[target_col_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    
    def only_cat(df, embed_cols, model, target_col_name, X_train, X_test, y_train, y_test):

      feat_names = []

      columns = embed_cols
      print("Columns on which embedings will be applied : {}".format(columns))

      cvec = CountVectorizer(tokenizer=word_tokenize, stop_words='english', token_pattern=None).fit(X_train[columns[0]])
      X_train_transform = cvec.transform(X_train[columns[0]])
      X_test_transform = cvec.transform(X_test[columns[0]])

      feat_names = feat_names + cvec.get_feature_names()


      for i in range(1, len(columns)):
        cvec = CountVectorizer(tokenizer=word_tokenize, stop_words='english', token_pattern=None).fit(X_train[columns[i]])
        X_train_trans = cvec.transform(X_train[columns[i]])
        X_test_trans = cvec.transform(X_test[columns[i]])

        feat_names = feat_names + cvec.get_feature_names()

        X_train_transform = hstack((X_train_transform, X_train_trans))
        X_test_transform = hstack((X_test_transform, X_test_trans))

      model.fit(X_train_transform, y_train.values)
      pred = model.predict(X_test_transform)
    
      whether_missing_pred, missing_pred = missing_predictions(y_test, pred)
      
      if whether_missing_pred:
        print('Alert! Model on categorical columns is unable to predict classes : {}, \nAll unique classes are : {}\n\n'.format(missing_pred, y_test.unique()))
      

      print("\n\nAccuracy with only Categorical columns is : {} \n\n".format(accuracy_score(y_test.values, pred)))

      print(classification_report(y_test, pred))
      
      d = classification_report(y_test, pred, output_dict=True)

      evaluate_result(df, y_test, target_col_name, d)

      print('\n\n Most important words in columns \n\n')
      pd.Series(abs(np.sum(model.coef_, axis=0)/model.coef_.shape[0]), index=feat_names).nlargest(20).plot(kind='barh')

      word_freq = dict(zip(feat_names, abs(np.sum(model.coef_, axis=0)/model.coef_.shape[0])))

      print('\n\n WordCloud of most important words in columns \n\n')

      wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      min_font_size = 10).fit_words(word_freq)
        
      # plot the WordCloud image                       
      plt.figure(figsize = (8, 8), facecolor = None)
      plt.imshow(wordcloud)
      plt.axis("off")
      plt.tight_layout(pad = 0)
        
      plt.show()

      return X_train_transform, X_test_transform
      


    
    def only_numerical(df, model, num_cols, ordinal_encode_cols, target_col_name, X_train, X_test, y_train, y_test):

      if ordinal_encode_cols == None:
        X_train_num = X_train[num_cols]
        X_test_num = X_test[num_cols]
        columns = num_cols

      
      else:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[ordinal_encode_cols] = enc.fit_transform(X_train[ordinal_encode_cols])
        X_test[ordinal_encode_cols] = enc.transform(X_test[ordinal_encode_cols])
      
        col_to_train = list(itertools.chain(num_cols , ordinal_encode_cols))

        X_train_num = X_train[col_to_train]

        X_test_num = X_test[col_to_train]

        columns = col_to_train

      model.fit(X_train_num, y_train.values)
      
      pred = model.predict(X_test_num)
    
      whether_missing_pred, missing_pred = missing_predictions(y_test, pred)
    
      if whether_missing_pred:
        print('Alert! Model on Numerical columns is unable to predict classes : {}, \nAll unique classes are : {}\n\n'.format(missing_pred, y_test.unique()))

      
      print("\n\nAccuracy with only numerical columns is : {}\n\n".format(accuracy_score(y_test.values, pred)))

      idx = np.where(y_test.values != pred)[0]

      temp = np.random.choice(idx,10)
    
      print('\nSome examples where your model misclassified \n\n')
      
      for i in temp:
          print("{} \n predicted : {}, \t true : {} \n\n".format(X_test_num[columns].iloc[i], pred[i], y_test.values[i]))
      
      print(classification_report(y_test, pred))
      
      d = classification_report(y_test, pred, output_dict=True)

      evaluate_result(df, y_test, target_col_name, d)

      plot_feature_importance(model, columns, df)

      
      return X_train_num, X_test_num


    
    def combined(df, model, X_train_final, X_test_final, y_train, y_test):
      

      model.fit(X_train_final, y_train.values)
      pred = model.predict(X_test_final)
      
      whether_missing_pred, missing_pred = missing_predictions(y_test, pred)

      if whether_missing_pred:
        print('Alert! Combined model is unable to predict classes : {}, \nAll unique classes are : {}\n\n'.format(missing_pred, y_test.unique()))

      print("\n\nAccuracy with entire dataset is : {}\n\n".format(accuracy_score(y_test.values, pred)))
      


    train_combined = True

    try:
      train_embed, test_embed = only_cat(df, embed_cols, model, target_col_name, X_train, X_test, y_train, y_test)
    
    except:
      train_combined = False
    
    try:
      train_num, test_num = only_numerical(df, model, num_cols, ordinal_encode_cols, target_col_name, X_train, X_test, y_train, y_test)

    except:
      train_combined = False

    if train_combined == True:
        X_train_final = hstack((train_embed, train_num))
        X_test_final = hstack((test_embed, test_num))

        combined(df, model, X_train_final, X_test_final, y_train, y_test)

  else:
    print("Value counts of each target class : \n{} \n\n".format(df[target_col_name].value_counts()))
    X = df.drop(target_col_name, axis=1)
    y = df[target_col_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    columns = X.columns

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    pred = model.predict(X_test)
    
    whether_missing_pred, missing_pred = missing_predictions(y_test, pred)

    if whether_missing_pred:
      print('Alert! Your model is unable to predict classes : {}, \nAll unique classes are : {}\n\n'.format(missing_pred, y_test.unique()))
        

    print("\n\nAccuracy with only numerical columns is : {}\n\n".format(accuracy_score(y_test.values, pred)))
    idx = np.where(y_test.values != pred)[0]
    temp = np.random.choice(idx,10)
  
    print('\nSome examples where your model misclassified \n\n')
    
    for i in temp:
        print("{} \n predicted : {}, \t true : {} \n\n".format(X_test[columns].iloc[i], pred[i], y_test.values[i]))
    
    print(classification_report(y_test, pred))
    
    d = classification_report(y_test, pred, output_dict=True)
    
    evaluate_result(df, y_test, target_col_name, d)
    
    plot_feature_importance(model, columns, df)
