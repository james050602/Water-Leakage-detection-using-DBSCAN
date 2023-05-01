# Water-Leakage-detection-using-DBSCAN

Firstly, the code imported the necessary libraries including Seaborn, Matplotlib, NumPy, Pandas and Scikit-learn after which it loaded the dataset ‘Measurements2.csv ‘. Once loaded, a seasonal differencing metric was applied in order to remove the seasonal component from the data, following this, the data was smoothed with a rolling window, with any missing values filled with the mean value of the data. 
After the data had been pre-processed, the code fitted the DBSCAN model, which clustered the data and attributed labels to these clusters. Then a DataFrame was created which included the original data as well as the labelled clusters. Then it filtered this data based on the given label values and any anomalies with a flow rate that exceeded the mean flow rate were removed. These anomalies were then grouped by their proximity in terms of row number and a calculation of both the first and last row number of each group. The code was then able to create a new column in the original DataFrame, titled ‘anomaly ‘, and initialised all of its values to 0. It then looped through each individual group of anomalies, setting the values of ‘anomaly ‘from the starting row to the final row, to 1 inclusively. Matplotlib was then used to plot the results. 
Then the code loaded the anomaly dataset ‘anomalies. csv’ and using the actual labels and predicted labels a calculation of the confusion matrix was made. After this, calculations for the chosen evaluation metrics were made, including the F1 score, the Rand Index, precision and recall, and then using Seaborn, it plotted the confusion matrix. Finally, the code created a new DataFrame in which it stored the result from adding a column identifying groups in the actual labels. The result was then filtered to retain just the groups with anomalies labelled as 1 and the results were then printed.  Then a loop was created to see which predicted anomalies overlapped with the actual anomalies. This allowed a comparison between the actual labels with the predicted labels, and from this, the mean time to detect was calculated.

