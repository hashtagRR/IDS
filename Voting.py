# Data Processing
import pandas as pd
#from sklearn import preprocessing

# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import time

import warnings
#warnings.filterwarnings('ignore')

# Start timing
start_time = time.time()

columns = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min",
    "Packet Length Max", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Avg Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init Fwd Win Bytes",
    "Init Bwd Win Bytes", "Fwd Act Data Packets", "Fwd Seg Size Min", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Y"
]

raw_data = pd.read_csv("E:\stuff\DS\CICIDS2018.csv", usecols=columns, index_col=None)

# Split the data into training and testing sets
x = raw_data.drop(columns=["Y"])
y = raw_data["Y"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50) 

#Voting Boost Classifier
type = "Voting > RF, LR, NB"
clf_lr = LogisticRegression(random_state=1, solver='lbfgs', max_iter=300)
clf_rf = RandomForestClassifier(random_state=1)
clf_knn = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_knn)], voting='soft')
eclf = eclf.fit(x_train, y_train)

# Make predictions on the test set
y_pred_eclf = eclf.predict(x_test)

# Evaluate the model
def evaluation(y_test, y_pred, type):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    print(type)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion)
    print(" ")

evaluation(y_test, y_pred_eclf, type)

#end timing
end_time = time.time()

# Calculate the Application time
program_time = end_time - start_time

print(f'Program Time: {program_time} seconds')