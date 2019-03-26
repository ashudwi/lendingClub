
###############################################################
# Title: Risk Prediction
###############################################################

###############################################################
# 1. Data Preparation
###############################################################
###############################################################
# 1.1 Loading packages
###############################################################

import pandas as pd
import numpy as np
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objs as go
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.naive_bayes as nb
from sklearn.tree import DecisionTreeClassifier
# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

###############################################################
# 1.2 Read csv data file
###############################################################
df = pd.read_csv("LoanStats3a.csv",  low_memory=False)  # Low_memory set to false to remove datatype warning

# Check dimension of data
df.shape

# Check a sample data
df.head(10)


# Check null values
df.isnull().sum()
a = (df.isnull().sum() * 100 / df.index.size).round(2)

# Remove features with >80% NUll values
limitPer = len(df) * .80
# Remove columns with more than 80 % Null
df = df.dropna(thresh=limitPer,axis=1)
df.shape
df.head(10)

# Convert datatype
df['int_rate'] = pd.to_numeric(df['int_rate'].str.strip('%')) / 100
df['revol_util'] = pd.to_numeric(df['revol_util'].str.strip('%')) / 100

# Dependant variable class
df.loan_status.unique()
df['loan_status'].replace('Does not meet the credit policy. Status:Charged Off', 'Charged Off',inplace=True)
df['loan_status'].replace('Does not meet the credit policy. Status:Fully Paid', 'Fully Paid',inplace=True)
df.loan_status.unique()

df.groupby('loan_status').count()


###############################################################
# 2 Data Exploration
###############################################################
###############################################################
# 2.1 View data
###############################################################
df.head(10)

# Correlation - Remove highly correlated features (Source - Kaggle)
numeric_variables = df.select_dtypes(exclude=["object"])
df_correlations = df.corr()

trace = go.Heatmap(z=df_correlations.values,
                   x=df_correlations.columns,
                   y=df_correlations.columns,
                  colorscale=[[0.0, 'rgb(165,0,38)'],
                              [0.1111111111111111, 'rgb(215,48,39)'],
                              [0.2222222222222222, 'rgb(244,109,67)'],
                              [0.3333333333333333, 'rgb(253,174,97)'],
                              [0.4444444444444444, 'rgb(254,224,144)'],
                              [0.5555555555555556, 'rgb(224,243,248)'],
                              [0.6666666666666666, 'rgb(171,217,233)'],
                              [0.7777777777777778, 'rgb(116,173,209)'],
                              [0.8888888888888888, 'rgb(69,117,180)'],
                              [1.0, 'rgb(49,54,149)']],
            colorbar = dict(
            title = 'Level of Correlation',
            titleside = 'top',
            tickmode = 'array',
            tickvals = [-0.52,0.2,0.95],
            ticktext = ['Negative Correlation','Low Correlation','Positive Correlation'],
            ticks = 'outside'
        )
                  )


layout = {"title": "Correlation Heatmap"}
data=[trace]

fig = dict(data=data, layout=layout)
plot(fig, filename='labelled-heatmap')


# Box plot to see interest rate by grade
df = df.sort_values('grade', ascending= True)
sns.boxplot(x="grade", y="int_rate", data=df)

# Bar plot to see Loan grade with more defaulter


# Plot distribution on Map of Interest rate, loan amount, average amount  -- Credit(kaggle)

by_loan_amount = df.groupby(['addr_state'], as_index=False).loan_amnt.sum()
by_interest_rate = df.groupby(['addr_state'], as_index=False).int_rate.mean()
by_income = df.groupby(['addr_state'], as_index=False).annual_inc.mean()
states = by_loan_amount['addr_state'].values.tolist()
average_loan_amounts = by_loan_amount['loan_amnt'].values.tolist()
average_interest_rates = by_interest_rate['int_rate'].values.tolist()
average_annual_income = by_income['annual_inc'].values.tolist()
from collections import OrderedDict
metrics_data = OrderedDict([('state_codes', states),
                            ('issued_loans', average_loan_amounts),
                            ('interest_rate', average_interest_rates),
                            ('annual_income', average_annual_income)])

metrics_df = pd.DataFrame.from_dict(metrics_data)
metrics_df = metrics_df.round(decimals=2)
metrics_df.head()

# Now it comes the part where we plot out plotly United States map
for col in metrics_df.columns:
    metrics_df[col] = metrics_df[col].astype(str)

scl = [[0.0, 'rgb(210, 241, 198)'], [0.2, 'rgb(188, 236, 169)'], [0.4, 'rgb(171, 235, 145)'], \
       [0.6, 'rgb(140, 227, 105)'], [0.8, 'rgb(105, 201, 67)'], [1.0, 'rgb(59, 159, 19)']]

metrics_df['text'] = metrics_df['state_codes'] + '<br>' + \
                     'Average loan interest rate: ' + metrics_df['interest_rate'] + '<br>' + \
                     'Average annual income: ' + metrics_df['annual_income']

data = [dict(
    type='choropleth',
    colorscale=scl,
    autocolorscale=False,
    locations=metrics_df['state_codes'],
    z=metrics_df['issued_loans'],
    locationmode='USA-states',
    text=metrics_df['text'],
    marker=dict(
        line=dict(
            color='rgb(255,255,255)',
            width=2
        )),
    colorbar=dict(
        title="$s USD")
)]

layout = dict(
    title='Issued Loans',
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)')
)

fig = dict(data=data, layout=layout)
plot(fig, filename='d3-cloropleth-map')



###############################################################
# 2.2 Feature selection/Engineering
###############################################################

# Remove features with unique values
df = df.loc[:, df.apply(pd.Series.nunique) != 1]
df.shape

# Drop features based on Business understanding (We can run schi test or F test for significance and accordingly remove)
#                                               (If time permits, I will perform this for better result)
df = df.drop(['loan_amnt'], axis=1)              # Remove Highly correlated
df = df.drop(['funded_amnt_inv'], axis=1)        # Remove Highly correlated
df = df.drop(['total_pymnt'], axis=1)            # Remove Highly correlated
df = df.drop(['total_pymnt_inv'], axis=1)        # Remove Highly correlated
df = df.drop(['total_rec_prncp'], axis=1)        # Remove Highly correlated
df = df.drop(['total_rec_int'], axis=1)          # Remove Highly correlated
df = df.drop(['recoveries'], axis=1)             # Remove Highly correlated
df = df.drop(['installment'], axis=1)            # Remove Highly correlated
df = df.drop(['emp_title'], axis=1)              # Employee title does not significant
df = df.drop(['emp_length'], axis=1)             # Employee length does not significant
df = df.drop(['title'], axis=1)                  # Employee Title does not significant
df = df.drop(['purpose'], axis=1)                # Remove Purpose (We can reduce levels)
df = df.drop(['zip_code'], axis=1)               # For now zip code remove; we can use for visualization
df = df.drop(['earliest_cr_line'], axis=1)       # earliest_cr_line does not significant
df = df.drop(['last_pymnt_d'], axis=1)           # Last payment date does not significant
df = df.drop(['pub_rec_bankruptcies'], axis=1)   # For now remove
df = df.drop(['tax_liens'], axis=1)              # Remove
df = df.drop(['issue_d'], axis=1)                # As data is for only one year
df = df.drop(['last_credit_pull_d'], axis=1)     # As data is for only one year
# Check Null values to Impute or remove
df.isnull().sum()
# Remove/drop Null observations
df = df.dropna(how='any', axis=0)

# Check if still any null
df.isnull().sum()
df.shape

df.to_csv('clean.csv')

###############################################################
# 2.3 Categorical to Dummy Variables
###############################################################

# Convert to dummy variables
dummy_cols = ['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'addr_state', 'debt_settlement_flag']
df = pd.get_dummies(df, columns=dummy_cols)
df.shape
df.head(5)


###############################################################
# 3. Model Planning
###############################################################
# Split dependent and independent features
X = np.array(df.ix[:, df.columns != 'loan_status'])
y = np.array(df.ix[:, df.columns == 'loan_status'])

# (X_train, X_test, y_train, y_test) = train_test_split(df, 'loan_status', test_size=.2)

###############################################################
# 3.1 Train, Test Split
###############################################################

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
X_test.shape

# Check class data imbalance
sum(y_train =='Fully Paid')
sum(y_train =='Charged Off')  # Data is highly imbalance

# Balance data using SMOTE or ROSE balancing method
smt = SMOTE(random_state=2)
X_train, y_train = smt.fit_sample(X_train, y_train)

# Check class data balance - Should be balance
sum(y_train =='Fully Paid')
sum(y_train =='Charged Off')


###############################################################
# 4. Model Building
###############################################################
###############################################################
# 4.1 Logistic Regression
###############################################################
def model_LR():
    # creating classifier
    clf = LogisticRegression(tol=1e-8, penalty='l2', C=2)
    # training classifier
    clf.fit(X_train, y_train)
    # model type
    print("Model: ",type(clf))
    # Predicting probabilities
    p = clf.predict_proba(X_test)
    return (clf.predict(X_test),p)

###############################################################
# 4.2 Decision Tree
###############################################################
def model_DT():
    # creating classifier
    clf = DecisionTreeClassifier(max_depth=100)
    # training classifier
    clf.fit(X_train, y_train)
    # model type
    print("Model: ",type(clf))
    # Predicting probabilities
    p = clf.predict_proba(X_test)
    return (clf.predict(X_test), p)

###############################################################
# 4.3 Naive Bayes
###############################################################
def model_BernoulliNB():
    # creating classifier
    clf = nb.BernoulliNB(alpha=1.0, binarize=0.0)
    # training classifier
    clf.fit(X_train, y_train)
    # model type
    print("Model: ",type(clf))
    # Predicting probabilities
    p = clf.predict_proba(X_test)
    return (clf.predict(X_test), p)

###############################################################
# 4.4 Random Forest
###############################################################
def model_RF():
    # creating classifier
    clf = RandomForestClassifier(n_estimators=100)
    # training classifier
    clf.fit(X_train, y_train)
    # model type
    print("Model: ",type(clf))
    # Predicting probabilities
    p = clf.predict_proba(X_test)
    return (clf.predict(X_test),p)

###############################################################
#  Model Evaluation Function
###############################################################

def model_evaluation(model, label_test):
    # confusion matrix:
    cm = confusion_matrix(label_test, model, labels=None, sample_weight=None)
    tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + tn)
    accuracy = np.mean(model == label_test)
    print_results(precision, recall, accuracy)
    print("*****PRECISION****")
    print("%.4f" % (tp / (tp + fp)))
    print("*****RECALL****")
    print("%.4f" % (tp / (tp + tn)))
    return accuracy


def print_results(precision, recall, accuracy):
    banner = "Here is the classification report"
    print('\n', banner)
    print('=' * len(banner))
    print('{0:10s} {1:.1f}'.format('Precision', precision * 100))
    print('{0:10s} {1:.1f}'.format('Recall', recall * 100))
    print('{0:10s} {1:.1f}'.format('Accuracy', accuracy * 100))

    return accuracy

###############################################################
# 4.5 Model Execute
###############################################################
clf_LR, p = model_LR()
acc_LR = model_evaluation(clf_LR, y_test)
# ROC and AUC score
from sklearn.metrics import roc_auc_score as auc_score

print ('{0:10s} {1:.1f}'.format('AUC Score',auc_score(y_test, p[:,1])*100))

# Decison Tree
clf_DT,p=model_DT()
# model evaluation
acc_DT = model_evaluation(clf_DT, y_test)
print ('{0:10s} {1:.1f}'.format('AUC Score',auc_score(y_test, p[:,1])*100))


# Naive Bayes
clf_NB,p=model_BernoulliNB()
# model evaluation
acc_NB = model_evaluation(clf_NB, y_test)
print ('{0:10s} {1:.1f}'.format('AUC Score',auc_score(y_test, p[:,1])*100))

# Random Forest
clf_RF,p =model_RF()
# model evaluation
acc_RF = model_evaluation(clf_RF, y_test)
print ('{0:10s} {1:.1f}'.format('AUC Score',auc_score(y_test, p[:,1])*100))

# Accuracy for all Models
accuracy_normal=[acc_LR, acc_NB, acc_RF, acc_DT]
accuracy_normal=[('{0:2f}'.format(i*100)) for i in accuracy_normal]

###############################################################
# 4.5 Cross Validation for all models
###############################################################
# Logistic Regression
clf1 = LogisticRegression(tol=1e-8, penalty='l2', C=2)
# Naive Bayes
clf2 = nb.BernoulliNB(alpha=1.0, binarize=0.0)
# Decision Tree
clf3 = DecisionTreeClassifier(max_depth=100)
# Random Forest
clf4 = RandomForestClassifier(n_estimators=100)

models=[clf1,  clf2, clf3, clf4]

n_Folds = 10
# Accuracy after cross validation:
accuracy_cv = []
for clf in models:
    accuracy_common = 0
    for test_run in range(n_Folds):
        # (X_train, X_test, y_train, y_test) = train_test_split(X,, test_size=.2)
        # call classifier
        clf.fit(X_train, y_train)
        model = clf.predict(X_test)
        # compare result
        accuracy = np.mean(model == y_test)
        # append to common
        accuracy_common += accuracy
        # final score
    print('{0:10s} {1:.1f}'.format('Accuracy', float(accuracy_common) / 10 * 100))
    accuracy_cv.append('{0:.1f}'.format(float(accuracy_common) / 10 * 100))


###############################################################
# 5. Result
###############################################################
print("Normal Accuracy")
print("================")
print(accuracy_normal)
print("Accuracy post CV")
print("================")
print(accuracy_cv)

print('The best suitable model with highest accuracy:', max(accuracy_cv))

###############################################################
# 6. Score Generation and Interest (%)
###############################################################
# Generate score
def cal_score(p):
    score = p * 998 + 1
    return score

def generate_score(df):
    df['score'] = None
    df['score'] = df.apply(lambda x: cal_score(clf.predict_proba(df(x))))


def generate_interest(df):
    # From data:
    min_int = 5.42    # Min interest = 5.42%
    max_int = 24.59    # Max interest = 24.59%
    diff = max_int - min_int # Difference = 24.59 - 5.42 = 19.17
    max_score = 999
    df['interest_rate'] = None
    df['interest_rate'] = df['interest_rate'].apply(lambda x: (max_int - df['score'] * diff / max_score ))
                    # max_interest - score * diff_interest / max_credit score
    return df

###############################################################
# 6.2 Score Generation and Interest (%)
###############################################################
df_without_loanstatus = pd.read_csv("GenerateScore.csv",  low_memory=False)

df_with_score  = generate_score(df_without_loanstatus)
