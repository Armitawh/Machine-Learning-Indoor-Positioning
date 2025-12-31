Data1 = Data("trainingData.csv", "validationData.csv");
KNNnumNeighbors = 9;

[KNN_Building_prediction, KNN_Floor_prediction, KNN_Longitude_prediction, KNN_Latitude_prediction] = KNN_Prediction(Data1, KNNnumNeighbors);

KNN_Result_Matrix = [Data1.location_test_df KNN_Longitude_prediction KNN_Latitude_prediction KNN_Building_prediction KNN_Floor_prediction];

True_Buildings = Data1.location_test_df(:, 316);
True_Floors = Data1.location_test_df(:, 315);
True_Position = [Data1.location_test_df(:, 313) Data1.location_test_df(:, 314)];

KNN_Building_Accuracy = Classification_Accuracy(KNN_Building_prediction, True_Buildings);
KNN_Floor_Accuracy = Classification_Accuracy(KNN_Floor_prediction, True_Floors);

[KNN_Pos_Mean, KNN_Pos_Median, KNN_Pos_STD, KNN_Pos_Min, KNN_Pos_Max, KNN_Pos_Quartiles] = Regression_Accuracy([KNN_Longitude_prediction KNN_Latitude_prediction], True_Position);

fprintf('KNN Building Accuracy: %.2f%%\n', KNN_Building_Accuracy * 100);
fprintf('KNN Floor Accuracy: %.2f%%\n', KNN_Floor_Accuracy * 100);
fprintf('KNN Position Mean Error: %.2f\n', KNN_Pos_Mean);
fprintf('KNN Position Median Error: %.2f\n', KNN_Pos_Median);
fprintf('KNN Position Standard Deviation: %.2f\n', KNN_Pos_STD);
fprintf('KNN Position Min Error: %.2f\n', KNN_Pos_Min);
fprintf('KNN Position Max Error: %.2f\n', KNN_Pos_Max);
fprintf('KNN Position Quartiles: %.2f, %.2f, %.2f\n', KNN_Pos_Quartiles(1), KNN_Pos_Quartiles(2), KNN_Pos_Quartiles(3));

