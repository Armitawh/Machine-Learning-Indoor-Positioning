clc;
clear;

trainingData = readtable('trainingData.csv');
validationData = readtable('validationData.csv');

train_features = trainingData{:, 1:520};  
train_longitude = trainingData.LONGITUDE;
train_latitude = trainingData.LATITUDE;
train_floor = trainingData.FLOOR;
train_building = trainingData.BUILDINGID;

test_features = validationData{:, 1:520};  
test_longitude = validationData.LONGITUDE;
test_latitude = validationData.LATITUDE;
test_floor = validationData.FLOOR;
test_building = validationData.BUILDINGID;

RF_Data.location_train_df = [train_features, train_longitude, train_latitude, train_floor, train_building];
RF_Data.location_test_df = test_features;

numBags = 3;

[RF_Building_prediction, RF_Floor_prediction, RF_Longitude_prediction, RF_Latitude_prediction] = RF_Prediction(RF_Data, numBags);

True_Buildings = validationData.BUILDINGID;
True_Floors = validationData.FLOOR;
True_Position = [validationData.LONGITUDE, validationData.LATITUDE];

predicted_coords = [RF_Longitude_prediction, RF_Latitude_prediction];
true_coords = [test_longitude, test_latitude];
[Mean, Median, STD, Min, Max, Quartiles] = Regression_Accuracy(predicted_coords, true_coords);


building_accuracy = Classification_Accuracy(RF_Building_prediction, True_Buildings);
floor_accuracy = Classification_Accuracy(RF_Floor_prediction, True_Floors);

disp('Regression Accuracy (Longitude and Latitude Predictions):');
disp(['Mean Distance Error: ', num2str(Mean)]);
disp(['Median Distance Error: ', num2str(Median)]);
disp(['Standard Deviation of Distance Error: ', num2str(STD)]);
disp(['Minimum Distance Error: ', num2str(Min)]);
disp(['Maximum Distance Error: ', num2str(Max)]);
disp(['25th Percentile Distance Error: ', num2str(Quartiles(1))]);
disp(['50th Percentile Distance Error: ', num2str(Quartiles(2))]);
disp(['75th Percentile Distance Error: ', num2str(Quartiles(3))]);

disp('Classification Accuracy:');
disp(['Building Prediction Accuracy: ', num2str(building_accuracy)]);
disp(['Floor Prediction Accuracy: ', num2str(floor_accuracy)]);

figure;

scatter(true_coords(:,1), true_coords(:,2), 'b');
hold on;
scatter(predicted_coords(:,1), predicted_coords(:,2), 'r');
legend('True Coordinates', 'Predicted Coordinates');
title('True vs Predicted Coordinates');
xlabel('Longitude');
ylabel('Latitude');
grid on;
hold off;

save('RF_Models.mat', 'RF_Data', 'numBags', 'RF_Building_prediction', 'RF_Floor_prediction', 'RF_Longitude_prediction', 'RF_Latitude_prediction');


function [RF_Building_prediction, RF_Floor_prediction, RF_Longitude_prediction, RF_Latitude_prediction] = RF_Prediction(RF_Data, numBags)
        
    model = fitensemble(RF_Data.location_train_df(:, 1:520), RF_Data.location_train_df(:, 521),'Bag',numBags,'Tree','Type','regression');
    
    RF_Longitude_prediction = predict(model, RF_Data.location_test_df(:, 1:520)); 
  
    model = fitensemble(RF_Data.location_train_df(:, 1:520), RF_Data.location_train_df(:, 522),'Bag',numBags,'Tree','Type','regression');
    
    RF_Latitude_prediction = predict(model, RF_Data.location_test_df(:, 1:520));
    
    model = fitensemble(RF_Data.location_train_df(:, 1:520), RF_Data.location_train_df(:, 523),'Bag',numBags,'Tree','Type','classification');

    RF_Floor_prediction = predict(model, RF_Data.location_test_df(:, 1:520));
    
    model = fitensemble(RF_Data.location_train_df(:, 1:520), RF_Data.location_train_df(:, 524),'Bag',numBags,'Tree','Type','classification');
    
    RF_Building_prediction = predict(model, RF_Data.location_test_df(:, 1:520));
    
end

function [Mean, Median, STD, Min, Max, Quartiles] = Regression_Accuracy(Predicted_Values, True_Values)
    
    D1 = (Predicted_Values(:, 1) - True_Values(:, 1)).^2;
    D2 = (Predicted_Values(:, 2) - True_Values(:, 2)).^2;
    D = sqrt(D1 + D2);
    
    Quartiles = zeros(3, 1);

    Mean = mean(D);
    Median = median(D);
    STD = std(D);
    Min = min(D);
    Max = max(D);
    Quartiles(1) = prctile(D.', 25);
    Quartiles(2) = prctile(D.', 50);
    Quartiles(3) = prctile(D.', 75);
end

function [Accuracy] = Classification_Accuracy(Predicted_Values, True_Values)

    Accuracy = sum(Predicted_Values == True_Values) / length(True_Values);

end

