import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules # Geeks for geeks
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict # Import train_test_split functionn
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import export_graphviz
from IPython.display import Image  
from sklearn import tree
import graphviz

sns.set()

'''
# DM Project - Question 2: Insurance Product Recommendation
'''

st.title("DM Project")
st.header("Question 2: Insurance Product Recommendation")

st.subheader("Insurance dataframe")
insurance = pd.read_csv('Insurance_Data.csv')
st.dataframe(insurance)

st.subheader("Plot of purchasedplan")
insurance["PurchasedPlan1"].value_counts().plot(kind="bar")
st.pyplot()

insurance["PurchasedPlan2"].value_counts().plot(kind="bar")
st.pyplot()

st.subheader("Are there any nulls?")
st.text(insurance.isna().sum())
st.subheader("OMG, yes！")

# data cleaning
insurance_ = insurance.copy()


cols = ['Gender', 'MaritalStatus', 'SmokerStatus', 'LifeStyle',
       'LanguageSpoken', 'HighestEducation', 'Race', 'Nationality',
       'MalaysiaPR', 'MovingToNewCompany', 'Occupation', 'Telco',
       'HomeAddress', 'ResidentialType', 'Transport',
       'MedicalComplication']
for i in cols:
  insurance_[i].fillna("Not_Specified", inplace = True) 
# insurance_.boxplot(column=['AnnualSalary'])

cols = ['Age', 'FamilyExpenses(month)', 'AnnualSalary']
for i in cols:
  insurance_[i].fillna(insurance_[i].median(), inplace = True) 

# df_insurance.boxplot(column = 'Age') # to prove why we fillna with mean
# insurance_['age_bins'] = pd.cut(x=insurance_['Age'], bins=[10, 20, 30, 40, 50])

insurance_['NoOfDependent'].fillna(insurance_['NoOfDependent'].min(), inplace = True) 

insurance_ = insurance_[['Age', 'Gender', 'MaritalStatus', 'SmokerStatus', 'LifeStyle',
       'LanguageSpoken', 'HighestEducation', 'Race', 'Nationality',
       'MalaysiaPR', 'MovingToNewCompany', 'Occupation', 'Telco',
       'HomeAddress', 'ResidentialType', 'NoOfDependent',
       'FamilyExpenses(month)', 'AnnualSalary', 'Customer_Needs_1',
       'Customer_Needs_2', 'Transport',
       'MedicalComplication', 'PurchasedPlan1', 'PurchasedPlan2']]

insurance_['Age'] = insurance_['Age'].apply(np.int64)
insurance_['NoOfDependent'] = insurance_['NoOfDependent'].apply(np.int64)


# insurance_.boxplot(column=cols[2])

# st.text(insurance_.info)

# Encoding 
# Label encode categorical columns
df_insurance = insurance_.copy()

le = preprocessing.LabelEncoder()

cols = ['Gender', 'MaritalStatus', 'SmokerStatus', 'LifeStyle',
       'LanguageSpoken', 'HighestEducation', 'Race', 'Nationality',
       'MalaysiaPR', 'MovingToNewCompany', 'Occupation', 'Telco',
       'HomeAddress', 'ResidentialType', 'Customer_Needs_1',
       'Customer_Needs_2', 'Transport',
       'MedicalComplication', 'PurchasedPlan1', 'PurchasedPlan2']

for i in cols:
    le.fit(df_insurance[i])
    df_insurance[i] = le.transform(df_insurance[i])
df_insurance

st.subheader("Label Encode Insurance data")
st.write(df_insurance)

# SMOTE data

df_insurance = df_insurance[['Age', 'Gender', 'MaritalStatus', 'SmokerStatus', 'LifeStyle',
       'LanguageSpoken', 'HighestEducation', 'Race', 'Nationality',
       'MalaysiaPR', 'MovingToNewCompany', 'Occupation', 'Telco',
       'HomeAddress', 'ResidentialType',  'Customer_Needs_1',
       'Customer_Needs_2', 'Transport', 'MedicalComplication',  
       'NoOfDependent',
       'FamilyExpenses(month)', 'AnnualSalary',
       'PurchasedPlan1', 'PurchasedPlan2']]


import imblearn

smt = imblearn.over_sampling.SMOTENC(categorical_features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], random_state=42)

# Min Max
X = df_insurance.drop(["PurchasedPlan1", "PurchasedPlan2"], 1)
y = df_insurance["PurchasedPlan1"]
features = X.columns

# min_max_scaler = MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(X)
# x_scaled = pd.DataFrame(x_scaled, columns = features)
# x_scaled.head(2)

X_res, y_res = smt.fit_resample(X, y)

# print(y_res.value_counts())
y_res.value_counts().plot(kind="bar")

X_res['age_bins'] = pd.cut(x=X_res['Age'], bins=[10, 20, 30, 40, 50])

le = preprocessing.LabelEncoder()

le.fit(X_res['age_bins'])
X_res['age_bins'] = le.transform(X_res['age_bins'])

df_insurance = X_res.copy()
df_insurance['PurchasedPlan1'] = y_res

st.subheader("SMOTE-d: PurchasedPlan1")
df_insurance["PurchasedPlan1"].value_counts().plot(kind="bar")
st.pyplot()

#Normalize data

X = df_insurance.drop(['PurchasedPlan1'], axis=1)
y = df_insurance['PurchasedPlan1']

from sklearn.preprocessing import normalize

data_scaled = normalize(X)
data_scaled = pd.DataFrame(data_scaled, columns=X.columns)
data_scaled.head()

# AgglomerativeClustering
st.subheader("AgglomerativeClustering: Age vs. AnnualSalary")

from sklearn.cluster import AgglomerativeClustering
n = 3

cluster = AgglomerativeClustering(n_clusters = n, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)
sns.relplot(x="Age", y="AnnualSalary", hue=cluster.labels_, data=data_scaled).set(xlim=(0, 0.0009),ylim=(0.979, 1.0005))
st.pyplot()

# Silhouette
st.subheader("Silhouette: Age vs. AnnualSalary")

from sklearn.metrics import silhouette_score 
from yellowbrick.cluster import silhouette_visualizer
from sklearn.cluster import KMeans 

st.write("Silhoutte Score (n = {}) = ".format(n), silhouette_score(data_scaled, cluster.labels_))

silhouette_visualizer(KMeans(n, random_state=12), data_scaled, colors='yellowbrick')
st.pyplot()

# KMeans clustering
st.subheader("KMeans clustering: Age vs. AnnualSalary")

km = KMeans(n_clusters = 2, random_state = 1)
km.fit(data_scaled)
km.labels_

distortions = []

# your codes here...
for i in range(1, 4):
    km = KMeans(
        n_clusters = i, 
        init = 'random',
        n_init = 10, 
        max_iter = 300, 
        tol = 1e-04, random_state = 0
    )
    km.fit(data_scaled)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 4), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
st.pyplot()

fig, axes = plt.subplots(1, 2, figsize=(13,6))

sns.scatterplot(x="Age", y="AnnualSalary", hue=y, data=data_scaled, ax=axes[0]).set(xlim=(0, 0.0009),ylim=(0.978, 1.001))
sns.scatterplot(x="Age", y="AnnualSalary", hue=km.labels_, data=data_scaled, ax=axes[1]).set(xlim=(0, 0.0009),ylim=(0.978, 1.001))
st.pyplot(fig)

# binning AnnualSalary & FamilyExpenses(month) then label encode the bins
binned_insurance = df_insurance.copy()

binned_insurance['AS_bins'] = pd.cut(x=binned_insurance['AnnualSalary'], bins=5)
binned_insurance['FE_bins'] = pd.cut(x=binned_insurance['FamilyExpenses(month)'], bins=5)

le = preprocessing.LabelEncoder()

le.fit(binned_insurance['AS_bins'])
binned_insurance['AS_bins'] = le.transform(binned_insurance['AS_bins'])

le.fit(binned_insurance['FE_bins'])
binned_insurance['FE_bins'] = le.transform(binned_insurance['FE_bins'])

X = binned_insurance.drop(['PurchasedPlan1', 'AnnualSalary', 'FamilyExpenses(month)', 'Age'], axis=1)
y = binned_insurance['PurchasedPlan1']

# normalize X
data_scaled = normalize(X)
data_scaled = pd.DataFrame(data_scaled, columns=X.columns)
X = data_scaled.copy()

# SelectKbest
st.subheader("SelectKBest Features")

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
#Recall that the chi-square test measures dependence between stochastic variables, so using this function “weeds out” 
#the features that are the most likely to be independent of class and therefore irrelevant for classification.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Univariate Selection (SelectKBest)
# X = df_insurance.iloc[:, 1:-1]  #independent columns
# y = df_insurance.iloc[:, -1]    #target column i.e Purchase Plan    


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Columns','Score']  #naming the dataframe columns
featureScores = featureScores.nlargest(20,'Score')
st.text("--------------Top 10--------------")
st.text(featureScores)  #print 10 best features

feat_cols = list(featureScores.index.values)

# # Boruta
# st.subheader("Boruta Features")

# def ranking(ranks, names, order=1):
#     minmax = MinMaxScaler()
#     ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
#     ranks = map(lambda x: round(x,2), ranks)
#     return dict(zip(names, ranks))


from sklearn.ensemble import RandomForestClassifier
# from boruta import BorutaPy

# # For random forest, class weight should be "balanced" and maximum of tree depth is 5.
# # set random_state = 1
# # set n_jobs=-1
# # n_estimators="auto"

# # sort the boruta score descending

# rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5)

# feat_selector = BorutaPy(rf, n_estimators = "auto", random_state = 1)

# # X = df_insurance.iloc[:, 1:-1]  #independent columns
# # y = df_insurance.iloc[:, -1]    #target column i.e Purchase Plan

# feat_selector = feat_selector.fit(X.values, y.values.ravel())

# colnames = X.columns
# boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
# boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])

# # sort the boruta score descending
# boruta_score = boruta_score.sort_values("Score", ascending = False)

# st.text('--------------Top 10--------------')
# st.text(boruta_score.head(18))

# import seaborn as sns

# sns.catplot(x="Score", y="Features", data = boruta_score, kind = "bar", 
#                height=14, aspect=1.9, palette='coolwarm')
# plt.title("Boruta all Features")
# st.pyplot()

# RFECV
st.subheader("RFECV")

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rfecv = RandomForestClassifier(random_state=1) 
rfecv = RFECV(estimator=clf_rfecv, step=1, cv=5, scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X, y)

print('Optimal number of features :', rfecv.n_features_)
# print('Best features :', X_train.columns[rfecv.support_])

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
st.pyplot()



def printResult(X_test, y_test, y_pred, clf, auc):
	# st.text("Accuracy on training set : {:.3f}".format(clr_rf.score(X_train, y_train)))
	st.text("Accuracy on test set : {:.3f}".format(clf.score(X_test, y_test)))
	st.text('AUC: %.2f\n' % auc)

	st.text('Precision= {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
	st.text('Recall= {:.2f}'. format(recall_score(y_test, y_pred, average='weighted')))
	st.text('F1= {:.2f}'. format(f1_score(y_test, y_pred, average='weighted')))
	st.text('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))


# DecisionTree
st.subheader("DecisionTreeClassifier")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

clf_DT = DecisionTreeClassifier(random_state=42)
clf_DT = clf_DT.fit(X_train, y_train)

y_pred = clf_DT.predict(X_test)
prob_DT = clf_DT.predict_proba(X_test)
# prob_DT = prob_DT[:, 1]
auc_DT = roc_auc_score(y_test, prob_DT, multi_class="ovr", average='weighted')

printResult(X_test, y_test, y_pred, clf_DT, auc_DT)

select_feature = SelectKBest(chi2, k=rfecv.n_features_).fit(X_train, y_train)

# Feature select then DT again
st.subheader("DecisionTreeClassifier with Feature Selection(fs)")

x_train_2 = select_feature.transform(X_train)
x_test_2 = select_feature.transform(X_test)

clf_DT_fs = clf_DT.fit(x_train_2, y_train)

y_pred = clf_DT_fs.predict(x_test_2)
prob_DT_fs = clf_DT_fs.predict_proba(x_test_2)
# prob_DT_fs = prob_DT_fs[:, 1]
auc_DT_fs = roc_auc_score(y_test, prob_DT_fs, multi_class="ovr", average='weighted')

printResult(x_test_2, y_test, y_pred, clf_DT_fs, auc_DT_fs)

# RF with original data
# split data train 70 % and test 30 %
st.subheader("RandomForestClassifier with Feature Selection(fs)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_RF = RandomForestClassifier(random_state=1)      
clf_RF = clf_RF.fit(X_train,y_train)

y_pred = clf_RF.predict(X_test)
prob_RF = clf_RF.predict_proba(X_test)
# prob_RF = prob_RF[:, 1]
auc_RF = roc_auc_score(y_test, prob_RF, multi_class="ovr", average='weighted')

printResult(X_test, y_test, y_pred, clf_RF, auc_RF)

select_feature = SelectKBest(chi2, k=rfecv.n_features_).fit(X_train, y_train)

# Feature select then RF again
st.subheader("RandomForestClassifier with Feature Selection(fs)")

x_train_2 = select_feature.transform(X_train)
x_test_2 = select_feature.transform(X_test)

clr_RF_fs = clf_RF.fit(x_train_2, y_train)

y_pred = clr_RF_fs.predict(x_test_2)
prob_RF_fs = clr_RF_fs.predict_proba(x_test_2)
# prob_RF_fs = prob_RF_fs[:, 1]
auc_RF_fs = roc_auc_score(y_test, prob_RF_fs, multi_class="ovr", average='weighted')

printResult(x_test_2, y_test, y_pred, clr_RF_fs, auc_RF_fs)


# Extra tree classifier
st.subheader("ExtraTreesClassifier")

# split data train 70 % and test 30 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_ET = ExtraTreesClassifier(random_state=1)      
clf_ET = clf_ET.fit(X_train,y_train)

y_pred = clf_ET.predict(X_test)
prob_ET = clf_ET.predict_proba(X_test)
# prob_clf = prob_clf[:, 1]
auc_ET = roc_auc_score(y_test, prob_ET, multi_class="ovr", average='weighted')

printResult(X_test, y_test, y_pred, clf_ET, auc_ET)

select_feature = SelectKBest(chi2, k=rfecv.n_features_).fit(X_train, y_train)

# Feature select then ExtraTreeClassifier again
st.subheader("ExtraTreesClassifier with Feature Selection(fs)")

x_train_2 = select_feature.transform(X_train)
x_test_2 = select_feature.transform(X_test)

clf_ET_fs = clf_ET.fit(x_train_2,y_train)

y_pred = clf_ET_fs.predict(x_test_2)
prob_ET_fs = clf_ET_fs.predict_proba(x_test_2)
# prob_clf = prob_clf[:, 1]
auc_ET_fs = roc_auc_score(y_test, prob_ET_fs, multi_class="ovr", average='weighted')


printResult(x_test_2, y_test, y_pred, clf_ET_fs, auc_ET_fs)

# Plot ROC for original data
st.subheader("Plot ROC for original data")

#DT
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_DT, pos_label='your_label') 
#RF
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, prob_RF) 
#ET
fpr_ET, tpr_ET, thresholds_ET = roc_curve(y_test, prob_ET) 

plt.plot(fpr_DT, tpr_DT, color='orange', label='DT') 
plt.plot(fpr_RF, tpr_RF, color='blue', label='RF')  
plt.plot(fpr_ET, fpr_ET, color='red', label='ET')  

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

# Apriori
# insurance1 = pd.get_dummies(insurance_)

# for i in range(-6, 0):
# 	st.subheader("Association Rule Mining: {}".format(insurance1.columns[i]))
# 	insurance_PurchasedPlan1_HomeSafe = pd.concat([insurance1[insurance1.columns[4:-6]], insurance1[insurance1.columns[i]]], axis=1)

# 	#Build model
# 	frequent_itemsets = apriori(insurance_PurchasedPlan1_HomeSafe, min_support=0.05, use_colnames=True)
# 	st.write(frequent_itemsets)

# 	# Collecting the inferred rules in a dataframe 
# 	rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) 
# 	rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False]) 
# 	st.write(rules[rules['consequents'] == frozenset({insurance1.columns[i]})])





