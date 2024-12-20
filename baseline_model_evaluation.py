import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

FILE_PATH = 'dialog_acts.dat'

df = pd.read_table(FILE_PATH, header=None, names=['combined'])

## First dataset
df['combined'] = df['combined'].astype(str)
df[['dialog_act', 'utterance_content']] = df['combined'].str.split(n=1, expand=True)
df = df[['dialog_act', 'utterance_content']]
df['utterace_content'] = df['utterance_content'].str.lower()

# Test-train split
x_train, x_test, y_train, y_test = train_test_split(df['utterance_content'], df['dialog_act'], test_size=0.15)

## Second Dataset
df_nodupes = df.drop_duplicates(subset=['utterance_content'])

# Test-train split
x_train_nodupes, x_test_nodupes, y_train_nodupes, y_test_nodupes = train_test_split(df_nodupes['utterance_content'], df_nodupes['dialog_act'], test_size=0.15)

###### Baseline 1: Assigns majority class of the data ###################################
def majorityclass_baseline(y_labels):
    majority_class = y_labels.mode()[0]
    y_pred = [majority_class] * len(y_labels)
    return y_pred

# classify utterance for baseline 1. doesn't take an argument because it just
# returns the majority class of the training data. this function is just for good measure.
def bsl1_classify_utterance():
    return majorityclass_baseline(y_train)[0]

###### Baseline 2: Rule-based keyword matching (accuracy around 85%) #######################
def rulebased_baseline(x_data):
    y_pred = [bsl2_classify_utterance(utterance) for utterance in x_data]
    return y_pred

def bsl2_classify_utterance(utterance):
    if re.search(r'thank you|thanks', utterance):
        return 'thankyou'
    elif re.search(r'goodbye|bye', utterance):
        return 'bye'
    elif re.search(r'hello|hi', utterance):
        return 'hello'
    elif re.search(r'how about|is there|what about', utterance):
        return 'reqalts'
    elif re.search(r'unintelligible|laughing|noise|sil|cough|sig', utterance):
        return 'null'
    elif re.search(r'yes|sure|right', utterance):
        return 'affirm'
    elif re.search(r'phone number|address|post code', utterance):
        return 'request'
    elif re.search(r'repeat|again', utterance):
        return 'repeat'
    elif 'no' in utterance:
        return 'negate'
    elif 'more' in utterance:
        return 'reqmore'
    else:
        return 'inform'
    
####### Evaluate Baseline Models ##############################################
acc_base1_train = accuracy_score(y_train, majorityclass_baseline(y_train))
acc_base2_train = accuracy_score(y_train, rulebased_baseline(x_train))

acc_base1_test= accuracy_score(y_test, majorityclass_baseline(y_test))
acc_base2_test = accuracy_score(y_test, rulebased_baseline(x_test))

acc_base1_train_nodupes = accuracy_score(y_train_nodupes, majorityclass_baseline(y_train_nodupes))
acc_base2_train_nodupes = accuracy_score(y_train_nodupes, rulebased_baseline(x_train_nodupes))

acc_base1_test_nodupes = accuracy_score(y_test_nodupes, majorityclass_baseline(y_test_nodupes))
acc_base2_test_nodupes = accuracy_score(y_test_nodupes, rulebased_baseline(x_test_nodupes))


### Machine Learning ########################################################################

# Bag of Words
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train_bow = vectorizer.transform(x_train)
x_test_bow = vectorizer.transform(x_test)

# Bag of Words (no duplicates) # Question: Should we fit a new vectorizer or can we use the same from the duplicates?
vectorizer_noduplicates = CountVectorizer()
vectorizer_noduplicates.fit(x_train_nodupes)
x_train_bow_nodupes = vectorizer_noduplicates.transform(x_train_nodupes)
x_test_bow_nodupes = vectorizer_noduplicates.transform(x_test_nodupes)
####### Decision tree  ##################################################
clf_tree = DecisionTreeClassifier()
clf_tree = clf_tree.fit(x_train_bow, y_train)
y_clf_train_tree = clf_tree.predict(x_train_bow)
y_clf_test_tree = clf_tree.predict(x_test_bow)

# Decision tree (no duplicates)
clf_tree_nodupes = DecisionTreeClassifier(random_state=42)
clf_tree_nodupes = clf_tree_nodupes.fit(x_train_bow_nodupes, y_train_nodupes)
y_clf_train_tree_nodupes = clf_tree_nodupes.predict(x_train_bow_nodupes)
y_clf_test_tree_nodupes = clf_tree_nodupes.predict(x_test_bow_nodupes)

####### Logistic regression ############################################
clf_logreg = LogisticRegression(random_state=42)
clf_logreg.fit(x_train_bow, y_train)
y_clf_train_logreg = clf_logreg.predict(x_train_bow)
y_clf_test_logreg = clf_logreg.predict(x_test_bow)

# Logistic regression (no duplicates)
clf_logreg_nodupes = LogisticRegression()
clf_logreg_nodupes.fit(x_train_bow_nodupes, y_train_nodupes)
y_clf_train_logreg_nodupes = clf_logreg_nodupes.predict(x_train_bow_nodupes)
y_clf_test_logreg_nodupes = clf_logreg_nodupes.predict(x_test_bow_nodupes)

# Predict dialog act based on user-defined utterance
# classifiers:
    # 1 = baseline 1 (majority class)
    # 2 = baseline 2 (rule-based)
    # 3 = decision tree
    # 4 = logistic regression
def predict(text, duplicates, classifier):
    if duplicates:
        text_transformed = vectorizer.transform([text])
    else:
        text_transformed = vectorizer.transform([text])

    match classifier:
        case 1:
            return bsl1_classify_utterance()
        case 2:
            return bsl2_classify_utterance(text)
        case 3:
            prediction = clf_tree.predict(text_transformed)
            return prediction[0]
        case 4:
            prediction = clf_logreg.predict(text_transformed)
            return prediction[0]
        

# user interaction function, opens a dialog that accepts user inputs and produces
# dialog act prediction from our models
def user_interaction():
    while True:
        user_input = input("Enter a sentence (type 'exit' to quit): ").lower()
        if user_input == 'exit':
            break

        predictions = {
            'Baseline 1 (with dupes)': predict(user_input, True, 1),
            'Baseline 1 (no dupes)': predict(user_input, False, 1),
            'Baseline 2 (with dupes)': predict(user_input, True, 2),
            'Baseline 2 (no dupes)': predict(user_input, False, 2),
            'Decision Tree (with dupes)': predict(user_input, True, 3),
            'Decision Tree (no dupes)': predict(user_input, False, 3),
            'Logistic Regression (with dupes)': predict(user_input, True, 4),
            'Logistic Regression (no dupes)': predict(user_input, False, 4)
        }

        print("\nPredictions:")
        for model, prediction in predictions.items():
            print(f"{model}: {prediction}")
        print("\n")

## PRINT ACCURACIES ###
# Baseline 1 (Majority Class)
print(f'Accuracy Baseline 1 (Majority Class) on train data: {acc_base1_train: .2f}')
print(f'Accuracy Baseline 1 (Majority Class) on test data: {acc_base1_test: .2f}')
print(f'Accuracy Baseline 1 (Majority Class) on train data (no duplicates): {acc_base1_train_nodupes: .2f}')
print(f'Accuracy Baseline 1 (Majority Class) on test data (no duplicates): {acc_base1_test_nodupes: .2f}')

# Baseline 2 (Rule-based keyword matching)
print(f'Accuracy Baseline 2 (Rule-based keyword matching) on train data: {acc_base2_train: .2f}')
print(f'Accuracy Baseline 2 (Rule-based keyword matching) on test data: {acc_base2_test: .2f}')
print(f'Accuracy Baseline 2 (Rule-based keyword matching) on train data (no duplicates): {acc_base2_train_nodupes: .2f}')
print(f'Accuracy Baseline 2 (Rule-based keyword matching) on test data (no duplicates): {acc_base2_test_nodupes: .2f}')

# Classifier 1 (Decision tree)
print(f'Accuracy Classifier 1 (Decision tree) on train data: {accuracy_score(y_train, y_clf_train_tree): .2f}')
print(f'Accuracy Classifier 1 (Decision tree) on test data: {accuracy_score(y_test, y_clf_test_tree): .2f}')
print(f'Accuracy Classifier 1 (Decision tree) on train data (no duplicates): {accuracy_score(y_train_nodupes, y_clf_train_tree_nodupes): .2f}')
print(f'Accuracy Classifier 1 (Decision tree) on test data (no duplicates): {accuracy_score(y_test_nodupes, y_clf_test_tree_nodupes): .2f}')

# Classifier 2 (Logistic regression)
print(f'Accuracy Classifier 2 (Logistic regression) on train data: {accuracy_score(y_train, y_clf_train_logreg): .2f}')
print(f'Accuracy Classifier 2 (Logistic regression) on test data: {accuracy_score(y_test, y_clf_test_logreg): .2f}')
print(f'Accuracy Classifier 2 (Logistic regression) on train data (no duplicates): {accuracy_score(y_train_nodupes, y_clf_train_logreg_nodupes): .2f}')
print(f'Accuracy Classifier 2 (Logistic regression) on test data (no duplicates): {accuracy_score(y_test_nodupes, y_clf_test_logreg_nodupes): .2f}')

def print_evaluations():
    majorityclass_baseline_train = majorityclass_baseline(y_train)
    majorityclass_baseline_train_nodupes = majorityclass_baseline(y_train_nodupes)
    majorityclass_baseline_test = majorityclass_baseline(y_test)
    majorityclass_baseline_test_nodupes = majorityclass_baseline(y_test_nodupes)
    print(f'Evaluation Report Baseline 1 (Majority Class) on train data:\n {classification_report(y_train, majorityclass_baseline_train)}')
    print(f'Evaluation Report Baseline 1 (Majority Class) on train data (no duplicates):\n {classification_report(y_train_nodupes, majorityclass_baseline_train_nodupes)}')
    print(f'Evaluation Report Baseline 1 (Majority Class) on test data:\n {classification_report(y_test, majorityclass_baseline_test)}')
    print(f'Evaluation Report Baseline 1 (Majority Class) on test data (no duplicates):\n {classification_report(y_test_nodupes, majorityclass_baseline_test_nodupes)}')

    rulebased_baseline_train = rulebased_baseline(x_train)
    rulebased_baseline_train_nodupes = rulebased_baseline(x_train_nodupes)
    rulebased_baseline_test = rulebased_baseline(x_test)
    rulebased_baseline_test_nodupes = rulebased_baseline(x_test_nodupes)

    print(f'Evaluation Report Baseline 2 (Rule-based keyword matching) on train data:\n {classification_report(y_train, rulebased_baseline_train)}')
    print(f'Evaluation Report Baseline 2 (Rule-based keyword matching) on train data (no duplicates):\n {classification_report(y_train_nodupes, rulebased_baseline_train_nodupes)}')
    print(f'Evaluation Report Baseline 2 (Rule-based keyword matching) on test data:\n {classification_report(y_test, rulebased_baseline_test)}')
    print(f'Evaluation Report Baseline 2 (Rule-based keyword matching) on test data (no duplicates):\n {classification_report(y_test_nodupes, rulebased_baseline_test_nodupes)}')

    print(f'Evaluation Report Classifier 1 (Decision tree) on train data:\n {classification_report(y_train, y_clf_train_tree)}')
    print(f'Evaluation Report Classifier 1 (Decision tree) on train data (no duplicates):\n {classification_report(y_train_nodupes, y_clf_train_tree_nodupes)}')
    print(f'Evaluation Report Classifier 1 (Decision tree) on test data:\n {classification_report(y_test, y_clf_test_tree)}')
    print(f'Evaluation Report Classifier 1 (Decision tree) on test data (no duplicates):\n {classification_report(y_test_nodupes, y_clf_test_tree_nodupes)}')

    print(f'Evaluation Report Classifier 2 (Logistic regression) on train data:\n {classification_report(y_train, y_clf_train_logreg)}')
    print(f'Evaluation Report Classifier 2 (Logistic regression) on train data (no duplicates):\n {classification_report(y_train_nodupes, y_clf_train_logreg_nodupes)}')
    print(f'Evaluation Report Classifier 2 (Logistic regression) on test data:\n {classification_report(y_test, y_clf_test_logreg)}')
    print(f'Evaluation Report Classifier 2 (Logistic regression) on test data (no duplicates):\n {classification_report(y_test_nodupes, y_clf_test_logreg_nodupes)}')

print_evaluations()