import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the dataset
data = pd.read_csv('Measurements2.csv', parse_dates=['timestamp'], infer_datetime_format=True, usecols=['timestamp', 'flow'])

# Remove the seasonal component using seasonal differencing
data['flow'] = data['flow'] - data['flow'].shift(288*7)

# Smooth the data using a rolling window of size 20
data['flow'] = data['flow'].rolling(window=20).mean()
data['flow'].fillna(data['flow'].mean(), inplace=True)

# Fit the DBSCAN model
dbscan = DBSCAN(eps=0.18, min_samples=10)
dbscan.fit(data[['flow']])

# Get the cluster labels
labels = dbscan.labels_

# Create a new DataFrame that includes both the original data and the cluster labels
results = pd.concat([data, pd.DataFrame({'label': labels})], axis=1)

# Filter the data based on the label
anomalies = results[results['label'] == -1]

anomalies = anomalies[anomalies['flow'] > data['flow'].mean()]

# Add a new column with the row number
anomalies = anomalies.reset_index()

# Group the anomalies by row number proximity
row_window = 5000
anomalies['group'] = (anomalies['index'].diff() > row_window).cumsum()

# Calculate the start and end row number of each group and the average flow rate for each group
grouped_anomalies = anomalies.groupby(['group']).agg(start_row=('index', 'min'),
                                                      end_row=('index', 'max'))

# Create a new column called 'anomaly' in the original DataFrame and initialize all its values to 0
data['anomaly'] = 0

# Loop through each group of anomalies
for i, row in grouped_anomalies.iterrows():
    start_row = row['start_row']
    end_row = row['end_row']
    
    # Set the values of 'anomaly' in the rows between start_row and end_row (inclusive) to 1
    data.loc[start_row:end_row, 'anomaly'] = 1

# Plot the results
plt.scatter(np.arange(len(data)), data['flow'], c=data['anomaly'])
plt.xlabel('Timestamp')
plt.ylabel('Flow')
plt.title('DBSCAN Clustering Results')

plt.show()

# Print the start and end times of the anomalies 
print(grouped_anomalies)

# Load the anomaly dataset
ground_truth_labels = pd.read_csv('anomalies1.csv', usecols=['anomalies'])

# Calculate the confusion matrix
results['predicted_label'] = np.where(results['label']==-1, 1, 0)
confusion = confusion_matrix(ground_truth_labels, data['anomaly'])

# Calculate the precision, recall, and F1 score
TP = confusion[1, 1]
FP = confusion[0, 1]
FN = confusion[1, 0]
TN = confusion[0, 0]
precision = TP / (TP + FP)
# tpr is the same as recall
recall = TP / (TP + FN)
FPR = FP / (FP+ TN)
f1_score = 2 * (precision * recall) / (precision + recall)
RAND_INDEX = (TP + TN)/(TP + TN + FP + FN)

# Print the evaluation metrics
print('F1 score:', f1_score)
print('RI', RAND_INDEX)

# Plot the confusion matrix
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Add a column to identify groups
ground_truth_labels['group'] = (ground_truth_labels['anomalies'] != ground_truth_labels['anomalies'].shift()).cumsum()

# Create a dataframe to store the result
realresult = pd.DataFrame({'start_row': ground_truth_labels.groupby('group').apply(lambda x: x.index.min()),
                       'end_row': ground_truth_labels.groupby('group').apply(lambda x: x.index.max()),
                       'anomalies': ground_truth_labels.groupby('group')['anomalies'].first()})

# Filter the result to keep only the groups with anomalies labelled as 1
realresult = realresult[realresult['anomalies'] == 1]

# Count the number of groups
num_groups = realresult.shape[0]

# Rename the first group to "group 1"
realresult.index = [str(i) for i in range(0, num_groups)]
    
# Print the result
print('Number of groups with anomalies')
print(realresult[['start_row', 'end_row']])

# Loop through each group of anomalies detected by DBSCAN
for i, row in grouped_anomalies.iterrows():
    start_row = row['start_row']
    end_row = row['end_row']
    
    # Find the corresponding group in the ground truth dataset
    match = False
    for j, truth_row in realresult.iterrows():
        truth_start = truth_row['start_row']
        truth_end = truth_row['end_row']
        
        # Check if the groups overlap
        overlap = max(0, min(end_row, truth_end) - max(start_row, truth_start) + 1)

row_diff_list = []

# Loop through each actual group
for i, actual_row in realresult.iterrows():
    actual_start = actual_row['start_row']
    actual_end = actual_row['end_row']
    
    # Loop through each predicted group
    for j, predicted_row in grouped_anomalies.iterrows():
        predicted_start = predicted_row['start_row']
        predicted_end = predicted_row['end_row']
        
        # Check if the predicted group overlaps with the actual group
        if (predicted_start <= actual_end) and (predicted_end >= actual_start):
            
            # Calculate the difference in rows between the start of the predicted group and the actual group
            row_diff = abs(actual_start - predicted_start)
            
            # Append the row difference to the list
            row_diff_list.append(row_diff)

# Calculate the mean row difference
mean_row_diff = np.mean(row_diff_list)/288

# Print the result
print(f"Mean row difference: {mean_row_diff:.2f} row(s)")

