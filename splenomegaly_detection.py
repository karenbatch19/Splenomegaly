#!/usr/bin/env python
# coding: utf-8

# # README

# 2020.08.29 version of this code

# This is an ensemble model for training on data comprised of electronic radiologist notes to predict the presence of metastatic cancers. Model written in Python 3.
# 
# **NOTE: the model presented here was created, trained, tested, and validated on data that is not available for public use. The code will not run as is. This code has been cleaned of any Protected Health Information and can be viewed as a means of inspecting the architecture of the model.**
#
# Training data is accepted in csv form. The csv is required to have a column of notes for a specified location (i.e. "spleen") for the x data and an associated column of metastases (i.e. "spleen_metastases") comprised of either Yes/No or Yes/Indeterminate/No values for the y data (0/1/2 is also accepted). An "impression" column is also required. These columns can be have any name, so long as those names are identified in the "Setup" portion of the code. So long as the data has these three columns the model should work. Any other columns will not disrupt prediction or be used at all.
# 
# Libraries you'll need:
# - pandas
# - keras
# - numpy
# - matplotlib
# - sklearn
# - seaborn

# # Setup

# In[1]:


# Enter the names of the columns you'll be using to train the model.
# Later on, the Impression section will be added to the x-data, but since that is static,
# we don't need to specify it.
# For paths, specify from the root directory, i.e. /Users/name/documents/...,
x_column_name = "" #i.e. "BONES_SOFT_TISSUES"
y_column_name = "" #i.e. "bones_metastases"
impression_column_name = "" #i.e. "IMPRESSION"
training_csv_path = ""
prediction_csv_path = ""
export_csv_path = ""

# Prediction styles seem to come in two forms: (1) Yes/Indeterminate/No, and (2) Yes/No.
# The following variable is to account for the case where data is received in form (1), 
# but the goal is to predict for form (2) by setting all Indeterminate values to No.
# If binary_prediction = True: Predictions will be made on the Yes/No scale.
# If binary_prediction = False: Predictions will be made on the Yes/Indeterminate/No scale [input data-willing, of course].
binary_prediction = True


# # Imports & Function Creation

# In[2]:


import pandas as pd
import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier


# ### Mappings Displayer

# In[3]:


def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values from an SKlearn LabelEncoder.
    This is literally just to be able to reference what target values are 
    mapped to what integers when label encoding. It is only used for clarity,
    sanity checking, and labelling graphs.
    
    Args:
        le: a fitted SKlearn LabelEncoder
        
    Returns:
        dict: Dictionary showing applied mappings (i.e. 0:"No").
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res


# ### Text Standardizer

# In[4]:


def standardize_text(df, text_field):
    """
    Function standardizes all text in one column of an identified dataframe with string manipulation.

    Args:
        df (DataFrame): DataFrame to be edited.
        text_field (str): Name of column to be edited.

    Returns:
        DataFrame: The edited DataFrame.
    """
    df[text_field] = df[text_field].str.replace(r"@\S+", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r"\d\.\s", " ")
    
    return df


# ### Visualizing using PCA

# In[5]:


# def plot_LSA(test_data, test_labels, savepath="PCA.csv", plot=True):
#     """
#     Function to plot test data and labels using principal component analysis.
#     Not used anywhere within this file.

#     Args:
#         df (DataFrame): DataFrame to be edited.
#         text_field (str): Name of column to be edited.

#     Returns:
#         DataFrame: The edited DataFrame.
#     """
#     lsa = TruncatedSVD(n_components=2)
#     lsa.fit(test_data)
#     lsa_scores = lsa.transform(test_data)
#     color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
#     color_column = [color_mapper[label] for label in test_labels]
#     colors = ['red', 'blue', 'green', 'yellow', 'black']

#     red_patch = mpatches.Patch(color='red',label='init')
#     blue_patch = mpatches.Patch(color='blue',label='init')
#     green_patch = mpatches.Patch(color='green',label='init')
#     yellow_patch = mpatches.Patch(color='yellow',label='init')
#     black_patch = mpatches.Patch(color='blue',label='init')

#     patches = [red_patch, blue_patch, green_patch, yellow_patch, black_patch]

#     if plot:
#         plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
#         count = 0
#         for key,val in chosenMappings.items():
#             patches[count] = mpatches.Patch(color=colors[count], label=key)
#             count+=1

#         plt.legend(handles=patches[0:len(chosenMappings)], prop={'size': 10})


# ### Prediction Metric Evaluation

# In[6]:


def get_metrics(y_test, y_predicted, binary_prediction): 
    """
    Function calculates accuracy metrics for a given set of predictions.

    Args:
        y_test (list): Target data truth.
        y_predicted (list): Target data predictions.
        binary_prediction (bool): whether or not you're making a binary prediction (assigned in setup)

    Returns:
        DataFrame: The edited DataFrame.
    """
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, 
                              y_predicted)
        
    if binary_prediction == True:
        # true positives / (true positives+false positives)
        precision = precision_score(y_test, 
                                    y_predicted, 
                                    average="binary", 
                                    pos_label=1) 

        # true positives / (true positives + false negatives)
        recall = recall_score(y_test, 
                              y_predicted, 
                              average="binary", 
                              pos_label=1)

        # harmonic mean of precision and recall
        f1 = f1_score(y_test, 
                      y_predicted, 
                      average="binary", 
                      pos_label=1)
    
    else:
        # true positives / (true positives+false positives)
        precision = precision_score(y_test, 
                                    y_predicted, 
                                    pos_label=None,
                                    average='weighted') 

        # true positives / (true positives + false negatives)
        recall = recall_score(y_test, 
                              y_predicted, 
                              pos_label=None,
                              average='weighted')

        # harmonic mean of precision and recall
        f1 = f1_score(y_test, 
                      y_predicted, 
                      pos_label=None, 
                      average='weighted')

    return accuracy, precision, recall, f1
        


# ### Confusion Matrix Inspection

# In[7]:


def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    """
    Function generates a confusion matrix.
    Not currently used in this file.

    Args:
        cm (): 
        classes (): 
        normalize (bool): whether or not to normalize the data. Default is false.
        title (string): Title to be displayed.
        cmap (???): Colour mapping.

    Returns:
        plt: The confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap, origin='lower')
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt


# ### TFIDF Tokenizer

# In[8]:


def tfidf(data):
    """
    Function creates a tfidf tokenized training set and associated vectorizer for test data.

    Args:
        data (list): Training data to be tokenized.

    Returns:
        list: The edited training data.
        tfidf_vectorizer: tfidf vectorizer to be applied to test data.
    """
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer


# # Data Import & Preprocessing

# In[9]:


df_input = pd.read_csv(training_csv_path)
df_input.head()


# In[10]:


# Label encode the data.
le = LabelEncoder()
mappingsList = []

le.fit(df_input[y_column_name])
integerMapping = get_integer_mapping(le)
mappingsList.append(integerMapping) # This is only used for graphs and viz later, if those things are included.
df_input[y_column_name] = le.transform(df_input[y_column_name])

print("Mappings: \n" + str(integerMapping))


# In[11]:


# Convert Indeterminate values to No for binary prediction
# Contingent on the binary_prediction variable being set to True in Setup.
if binary_prediction == True:
    df_input.loc[df_input[y_column_name] == 'Indeterminate', y_column_name] = 'No'
    
print(df_input[y_column_name].value_counts())


# In[12]:


# Standardizing both the location notes and the impression notes. Will combine them shortly.
df_input = standardize_text(df_input, x_column_name) 
df_input = standardize_text(df_input, impression_column_name)
df_input.head()


# In[13]:


# Making a list out of the labels from the dataframe for use in creating train and test sets.
chosen_labels_for_training = df_input[y_column_name].tolist()
chosen_labels_for_training


# In[14]:


# Making a list out of the note data (both location and impression) for use in creating train and test sets.
list_corpus = (df_input[x_column_name] + df_input[impression_column_name]).tolist()
list_corpus


# In[15]:


# Creating train and test sets. First two variables there are your x and y.
X_train, X_test, y_train, y_test = train_test_split(list_corpus, 
                                                    chosen_labels_for_training, 
                                                    test_size=0.2, 
                                                    random_state=40)

print(f"X_train length: {len(X_train)}\nX_test length: {len(X_test)}\ny_train length: {len(y_train)}\ny_test length: {len(y_test)}")


# In[16]:


# TFIDF tokenization.
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# x_pred_tfidf = tfidf_vectorizer.transform(df_mets_targets.iloc[:,0])


# # Classification

# ### Creating Classifiers

# In[17]:


#Linear Regression

#parameters:
#C selected through iterative testing, changed from the time above (lower worked better this time)
#weight balanced, all yvalues considered equally
#newton algorithm used to solve, best for multinomial data
#n_jobs, -1 means use all processors
clf_lr = LogisticRegression(C=15.0, 
                               class_weight='balanced', 
                               solver='newton-cg', 
                               multi_class='multinomial', 
                               n_jobs=-1, 
                               random_state=40)


# In[18]:


#Support Vector Machine
from sklearn import svm

#Create a svm Classifier
clf_svm = svm.SVC(kernel='linear',
                  probability=True) # Linear Kernel


# In[19]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
clf_rf = RandomForestClassifier(n_estimators=2000, 
                               bootstrap = True,
                               max_features = 'sqrt')


# In[20]:


#XGBoost
from xgboost import XGBClassifier
clf_xgb = XGBClassifier()


# ### Training Classifiers

# In[21]:


#lr 
clf_lr.fit(X_train_tfidf, y_train)
y_pred_lr = clf_lr.predict(X_test_tfidf)
accuracy_lr, precision_lr, recall_lr, f1_lr = get_metrics(y_test, y_pred_lr, binary_prediction)


# In[22]:


#svm
clf_svm.fit(X_train_tfidf, y_train)
y_pred_svm = clf_svm.predict(X_test_tfidf)
accuracy_svm, precision_svm, recall_svm, f1_svm = get_metrics(y_test, y_pred_svm, binary_prediction)


# In[23]:


#rf
clf_rf.fit(X_train_tfidf, y_train)
y_pred_rf = clf_rf.predict(X_test_tfidf)

# # I feel like the code below was important for something. Something about multiple columns maybe?
# rf_probs = clf_rf.predict_proba(X_test_tfidf)[:, 1]
# from sklearn.metrics import roc_auc_score
# roc_value = roc_auc_score(y_test, rf_probs)

accuracy_rf, precision_rf, recall_rf, f1_rf = get_metrics(y_test, y_pred_rf, binary_prediction)


# In[24]:


#xgb
clf_xgb.fit(X_train_tfidf, y_train)
y_pred_xgb = clf_xgb.predict(X_test_tfidf)
accuracy_xgb, precision_xgb, recall_xgb, f1_xgb = get_metrics(y_test, y_pred_xgb, binary_prediction)


# In[25]:


# Calculating weightings for how to appropriately weight performance of models in ensemble voting.
avgAcc = (accuracy_lr+accuracy_svm+accuracy_rf+accuracy_xgb)/4
avgPre = ((precision_lr+precision_svm+precision_rf+precision_xgb)/4)
avgRec = (recall_lr+recall_svm+recall_rf+recall_xgb)/4
sumAverages = avgAcc + avgPre + avgRec

orderedWeights = [((accuracy_lr+(precision_lr*1.25)+recall_lr-sumAverages)*100)+10, 
                  ((accuracy_svm+(precision_svm*1.25)+recall_svm-sumAverages)*100)+10, 
                  ((accuracy_rf+(precision_rf*1.25)+recall_rf-sumAverages)*100)+10, 
                  ((accuracy_xgb+(precision_xgb*1.25)+recall_xgb-sumAverages)*100)+10]

print("Calculated weights: \n" + str(orderedWeights))


# ### Making Ensemble Voters

# In[26]:


#Voting Classifier - Soft
ensemble_soft=VotingClassifier(estimators=[('Regression', clf_lr), 
                                           ('SVM', clf_svm), 
                                           ('RF', clf_rf), 
                                           ('XGB', clf_xgb)], 
                               voting='soft',
                               weights=orderedWeights)


# ### Running Ensemble Voters

# In[27]:


#ensemble_soft
ensemble_soft.fit(X_train_tfidf, y_train)
y_pred_ensemble_soft = ensemble_soft.predict(X_test_tfidf)
accuracy_ensemble_soft, precision_ensemble_soft, recall_ensemble_soft, f1_ensemble_soft = get_metrics(y_test, y_pred_ensemble_soft, binary_prediction)


# In[28]:


print("E-S accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_ensemble_soft, precision_ensemble_soft, recall_ensemble_soft, f1_ensemble_soft))
print("\nBuilt on...")
print("LRe accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_lr, precision_lr, recall_lr, f1_lr))
print("SVM accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_svm, precision_svm, recall_svm, f1_svm))
print("RFo accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_rf, precision_rf, recall_rf, f1_rf))
print("XGB accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_xgb, precision_xgb, recall_xgb, f1_xgb))


# # Predict Unannotated Cases

# In[29]:


# Import based on location identified in setup
df_unannotated = pd.read_csv(prediction_csv_path)

# Standardize text
df_unannotated = standardize_text(df_unannotated, x_column_name) 
df_unannotated = standardize_text(df_unannotated, impression_column_name)

# Create combined impression + location corpus
df_unannotated["Corpus"] = df_unannotated[x_column_name] + df_unannotated[impression_column_name]

# Create tfidf vectorized data
x_pred_tfidf = tfidf_vectorizer.transform(df_unannotated["Corpus"])

# Make predictions 
predicted = ensemble_soft.predict(x_pred_tfidf)

# Add predictions as column to dataframe
df_unannotated["Predicted"] = predicted


# In[30]:


# Export dataframe
df_unannotated.to_csv(export_csv_path)
