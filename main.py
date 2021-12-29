from Data_Preprop.extract_text import *
from Data_Preprop.clean_data import *
from summarize import *
import os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
from num2words import num2words
from googletrans import Translator, constants
from deep_translator import GoogleTranslator

gt = GoogleTranslator(source='auto', target='id')

def numtowords(text):
    txt = text.split(" ")
    newtext = [num2words(w) if w.isdigit() else w for w in txt]
    return " ".join(newtext)
    
def split(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state = 42, stratify=y)
    return X_train, X_test, y_train, y_test

def classify(X_train, X_test, y_train, y_test,n_sum="None"):

    # nb = Pipeline([('vect', CountVectorizer()),
    #             ('tfidf', TfidfTransformer()),
    #             ('clf', MultinomialNB()),
    #             ])
    # nb.fit(X_train, y_train)
    # from sklearn.metrics import classification_report
    # y_pred = nb.predict(X_test)
    # print('accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred))

    # filename = 'naivebayes_'+n_sum+'.sav'
    # pickle.dump(nb, open(filename, 'wb'))

################################################################
    kn = KNeighborsClassifier(n_neighbors=10)
    kneigh= Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(kn)),
                ])
    kneigh.fit(X_train, y_train)

    from sklearn.metrics import classification_report
    y_pred = kneigh.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))

    filename = 'knn_'+n_sum+'.sav'
    pickle.dump(kneigh, open(filename, 'wb'))
################################################################

    # Svm
    sv = SVC(gamma='auto',kernel='linear')
    svcModel= Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', OneVsRestClassifier(sv)),
                ])
    svcModel.fit(X_train, y_train)

    y_pred = svcModel.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))

    filename = 'svm_'+n_sum+'.sav'
    pickle.dump(svcModel, open(filename, 'wb'))
################################################################

    # Random forest
    rr = RandomForestClassifier()
    model = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(rr)),
               ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
    filename = 'randomforest_'+n_sum+'.sav'
    pickle.dump(model, open(filename, 'wb'))

def encode(text):
    if text == "E-Apps": 
        return 1 
    elif text == "Interactive Multimedia": 
        return 2
    elif text == "AI":
        return 3
    else:
        return 4

if __name__ == "__main__":
    df = pd.DataFrame()
    # load Data
    if os.path.exists("Loaded_Data.xlsx"):
        df = pd.read_excel("Loaded_Data.xlsx")
        df["Text"] = df["Text"].str.replace("\uf0b7","")
        # df["Summarized_Text"] = df.apply(lambda row:generate_summary(row["Text"],10,False), axis=1)
        # df.to_excel("Loaded_Data_new.xlsx")
        print("Data Loaded from excel")
    else:
        df = get_texts()
        print("Data loaded from pdfs") 

    # df = df[["Label","Text","Summarized_Text_5","Summarized_Text_10","Summarized_Text_15","Summarized_Text_20"]]
    df = df[["Label","Text"]]
    print(df.head())
    df["Label"] = df.apply(lambda row: encode(row["Label"]),axis=1)
    df["Text"] = df['Text'].str.replace('[^\w\s]',' ')
    df["Text"] = df.apply(lambda row: numtowords(row["Text"]),axis=1)
    max_len = 0
    for i in range(len(df)):
        if max_len < len(df["Text"].iloc[i]):
            max_len = len(df["Text"].iloc[i])
    
    print(max_len)
    # print(df.head())
    # tmp = df["Text"].iloc[1]
    # tmp = [w for w in tmp.split(" ") if w!=""]
    # print(tmp)
    # print(gt.translate_batch(tmp))
    # df["Text"] = df.apply(lambda row: " ".join([gt.translate(w) for w in row["Text"].split(" ")]),axis=1)

    X_train, X_test, y_train, y_test = split(df["Text"],df["Label"])
    classify(X_train, X_test, y_train, y_test)

    # print("summarized Text 5")
    # X_train, X_test, y_train, y_test = split(df["Summarized_Text_5"],df["Label"])
    # classify(X_train, X_test, y_train, y_test,"5")
    
    # print("summarized Text 10")
    # X_train, X_test, y_train, y_test = split(df["Summarized_Text_10"],df["Label"])
    # classify(X_train, X_test, y_train, y_test,"10")

    # print("summarized Text 15")
    # X_train, X_test, y_train, y_test = split(df["Summarized_Text_15"],df["Label"])
    # classify(X_train, X_test, y_train, y_test,"15")

    # print("summarized Text 20")
    # X_train, X_test, y_train, y_test = split(df["Summarized_Text_20"],df["Label"])
    # classify(X_train, X_test, y_train, y_test,"20")
