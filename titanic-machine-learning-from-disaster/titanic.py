import os
from zipfile import ZipFile
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

print(os.getcwd())
os.chdir("titanic-machine-learning-from-disaster")

#----------------------- DATA PREPARATION -----------------------#
# unzip folder
with ZipFile("titanic.zip", 'r') as zip_ref:
    zip_ref.extractall("titanic")

# import train dataset to data frame
titanic_df = pd.read_csv(r"titanic\train.csv", index_col=False)

#----------------------- DATA OVERVIEW -----------------------#
# get summary of data
titanic_df.head(10)
titanic_df.tail()
titanic_df.info()

titanic_df.count()

len(titanic_df['PassengerId'].unique())
titanic_df['Sex'].unique()

#check duplicate values
titanic_df.duplicated().unique()
ids = titanic_df['Name']
titanic_df[ids.isin(ids[ids.duplicated()])].sort_values("Name")

# check null values
titanic_df.isna().any()
titanic_df.isna().sum()

# can replace with mean
titanic_df['Age'].max()
titanic_df['Age'].min()

# can replace with 0
titanic_df['Cabin'].value_counts()

# replace with mode
titanic_df['Embarked'].value_counts()
titanic_df['Embarked'].unique()

# print null row
# https://saturncloud.io/blog/python-pandas-selecting-rows-whose-column-value-is-null-none-nan/
null_rows = titanic_df.loc[titanic_df['Embarked'].isnull() | titanic_df['Age'].isnull()]

# get unique data for each column
unique_val = {}
for col in titanic_df:
    unique_val[col] = titanic_df[col].unique()
unique_df = pd.DataFrame(unique_val.items(), columns=['Col', 'Value'])

#----------------------- DATA CLEANING -----------------------#
# (inplace=True to replace to the original df)
# when doing 'df[col].method(value, inplace=True)', 
# try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead,
# for all col
# titanic_df.dropna(inplace = True)   # remove row with null val 
# titanic_df.fillna(130, inplace = True)  # fill null value with val

# specific col. replace uses the mean() median() and mode()

def clean_titanic(titanic_df):
    # replace with mean
    age_mean = round(titanic_df['Age'].mean())
    titanic_df['Age'] = titanic_df['Age'].fillna(age_mean)

    # replace fare with mean
    age_mean = round(titanic_df['Fare'].mean())
    titanic_df['Fare'] = titanic_df['Fare'].fillna(age_mean)

    # replace with '0'
    titanic_df['Cabin'] = titanic_df['Cabin'].fillna(0)

    # replace with mode
    embarked_mode = titanic_df['Embarked'].mode()[0]
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna(embarked_mode)

    titanic_df.isna().any()

    #----------------------- DATA TRANSFORMATION -----------------------#
    # transform dataset - feature extraction + label
        # obj data = name, sex, ticket, cabin, embarked

    label_encoder = preprocessing.LabelEncoder()
    titanic_df['Embarked']= label_encoder.fit_transform(titanic_df['Embarked'])
    titanic_df['Embarked'].unique()

    titanic_df['Sex']= label_encoder.fit_transform(titanic_df['Sex'])
    titanic_df['Sex'].unique()

    titanic_df['Cabin'] = titanic_df['Cabin'].mask(titanic_df['Cabin'] == 0, 0)
    titanic_df['Cabin'] = titanic_df['Cabin'].mask(titanic_df['Cabin'] != 0, 1)
    titanic_df['Cabin'].unique()
    titanic_df['Cabin'] = titanic_df['Cabin'].astype('int64')

    # select certain column only
    titanic_df = titanic_df.drop(['Name', 'Ticket'], axis=1)

    return titanic_df

titanic_df_clean = clean_titanic(titanic_df)
#----------------------- FEATURE EXTRACTION -----------------------#
# find correlation
titanic_df_clean.corr()

# get data and label
titanic_df_clean_x = titanic_df_clean.drop(['Survived'], axis=1)
# titanic_df_clean_x = titanic_df_clean[['Sex']]
titanic_df_clean_y = titanic_df_clean['Survived']

# split dataset 80:20
x_train, x_test, y_train, y_test = train_test_split(titanic_df_clean_x, titanic_df_clean_y, test_size=0.2, random_state=11)
print("X_train Shape:",  x_train.shape)
print("X_test Shape:", x_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test Shape:", y_test.shape)

#----------------------- DATA MODELLING -----------------------#
# logistic regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

# evaluate model
print("Logistic Regression model accuracy:", metrics.accuracy_score(y_test, y_pred))

#----------------------- DATA VALIDATION -----------------------#
# import test.csv to dataframe
titanic_test = pd.read_csv(r"titanic\test.csv", index_col=False)

# check test data
titanic_test.info()
titanic_test.isna().any()

# clean test data
titanic_test.loc[titanic_test['Fare'].isnull()]
titanic_test.loc[titanic_test['Ticket'] == '3701']

titanic_test_clean = clean_titanic(titanic_test)
titanic_test_clean.info()

# predict test.csv - label -> Survived (0, 1)
y_pred = log_reg.predict(titanic_test_clean)

#----------------------- DATA SUBMISSION -----------------------#
submission = pd.DataFrame(list(zip(titanic_test_clean['PassengerId'], y_pred)), columns =['PassengerId', 'Survived'])
submission.to_csv(r"titanic\titanic_submission.csv", index=False, header=True)

# kaggle competitions submit -c titanic -f titanic_submission.csv -m "Message"

#----------------------- DATA TEST -----------------------#
# gender_submission.csv a set of predictions that assume all and only female 
# passengers survive, as an example of what a submission file should look like.
# extract only PassengerId and Survived
titanic_answer = pd.read_csv(r"titanic\gender_submission.csv", index_col=False)
titanic_answer.info()
# Evaluate the score with gender_submission.csv
y_test_answer = titanic_answer['Survived'].to_list()

print("Logistic Regression model accuracy:", metrics.accuracy_score(y_test_answer, y_pred))
