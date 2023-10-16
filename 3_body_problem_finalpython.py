import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import random
import skl2onnx
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from datetime import datetime

train_data = pd.read_csv(r'C:/Users/duart/OneDrive/Ambiente_de_Trabalho/Master_Analysis_Engineering_Big_Data/23-24/1st_semester/AA_ML/Kaggle_challenges/3_body_problem/3_body_problem/X_train.csv')


###### DROPPING THE SIMULATIONS WITH COLLISIONS ######
# Identify null rows based on the criterion (all values = 0.0 except for Id)
zero_rows = train_data[(train_data.drop('Id', axis=1) == 0).all(axis=1)]

# Remove the simulations with collisions from the DataFrame
train_data_preprocessed = train_data[~train_data.index.isin(zero_rows.index)]
train_data_preprocessed.reset_index(drop=True, inplace=True)
#train_data_preprocessed.to_csv('train_preprocessed.csv', index=False)


###### BASIC STATS OF THE INITIAL MATRIX ######
# Gives the stats for the preprocessed data (without the rows with zeros)
summary_stats_filtered = train_data_preprocessed.describe(include='all')

# Gives the stats for the nonprocessed data
summary_stats = train_data.describe(include="all")


###### CREATING THE FEATURES-LABELS MATRIX  ######
# We're not going to use the velocity components as features
train_data_without_velocity = train_data_preprocessed.drop(columns=['Id','v_x_1','v_x_2','v_y_1', 
                                                           'v_y_2', 'v_x_3', 'v_y_3'])
# Divide by simulations
list_of_times = [values for values in train_data_without_velocity['t']]
time_index_tuples = list(enumerate(list_of_times))
zeros_indexes = list(filter(lambda value: value[1] == 0, time_index_tuples))
zeros_indexes = [value[0] for value in zeros_indexes] 
list_of_simulations = []
lower_bound = 0
for i in range(len(zeros_indexes)-1):
    simulation = train_data_without_velocity.iloc[lower_bound:zeros_indexes[i+1]]
    list_of_simulations.append(simulation)
    lower_bound = zeros_indexes[i+1]
from tqdm import tqdm

# Add label and put the starting position at every row
for simulation in tqdm(list_of_simulations):
    first_row_values = simulation.head(1)
    simulation.loc[:,'x_1_label'] = simulation.loc[:, 'x_1']
    simulation.loc[:,'y_1_label'] = simulation.loc[:, 'y_1']
    simulation.loc[:,'x_2_label'] = simulation.loc[:, 'x_2']
    simulation.loc[:,'y_2_label'] = simulation.loc[:, 'y_2']
    simulation.loc[:,'x_3_label'] = simulation.loc[:, 'x_3']
    simulation.loc[:,'y_3_label'] = simulation.loc[:, 'y_3']
    for index, row in simulation.iterrows():
        simulation.at[index, 'x_1'] = first_row_values['x_1']
        simulation.at[index, 'y_1'] = first_row_values['y_1']
        simulation.at[index, 'x_2'] = first_row_values['x_2']
        simulation.at[index, 'y_2'] = first_row_values['y_2']
        simulation.at[index, 'x_3'] = first_row_values['x_3']
        simulation.at[index, 'y_3'] = first_row_values['y_3']

list_of_simulations_copy = list_of_simulations.copy()
random.shuffle(list_of_simulations_copy)     # we shuffle the data here so we are only shuffling different simulations and not amongst them (minimise data leakage - important for the vanilla model)
all_simulations = pd.concat(list_of_simulations_copy, ignore_index=True)
all_simulations.to_csv('feature_matrix.csv', index=False)


###### LOADING THE FEATURES-LABELS MATRIX  ###### (Only need to create once)
total_data = np.genfromtxt('feature_matrix.csv', delimiter=',')
total_data = total_data[1:] 


###### Train-Vali-Test SPLIT  ######  - Pay attention to the splitting we are doing, as it will depend on the model we want to use for each way of predicting

#### For parametric regression models
# splitting the dataset into training and validation
# sample, discard = train_test_split(total_data, train_size=0.7, shuffle=False)
data_train, data_temp = train_test_split(total_data, train_size=0.7, shuffle=False)
data_vali, data_test = train_test_split(data_temp, test_size=0.5, shuffle=False)

#### Splitting Features and Labels for parametric regression
features_train = data_train[:, :7]  
labels_train = data_train[:, 7:] 

features_vali = data_vali[:, :7]  
labels_vali = data_vali[:, 7:] 

features_test = data_test[:, :7]  
labels_test = data_test[:, 7:]


#### For KNN regression models
data_train_knn = total_data

#split the labels from the features
features_train_knn = data_train_knn[:, :7]  
labels_train_knn = data_train_knn[:, 7:] 


###### ACCELERATION FUNCTION  ###### - calculates and appends accels to the feature matrix
def accel_proxy(X):
    c1 = X[:, [1,2]]
    c2 = X[:, [3,4]]
    c3 = X[:, [5,6]]

    def _acc(c1, c2):
        c = c1-c2
        return c/(np.linalg.norm(c)**3)

    a1 = _acc(c1, c2) + _acc(c1, c3)
    a2 = _acc(c2, c1) + _acc(c2, c3)
    a3 = _acc(c3, c1) + _acc(c3, c2)
    #a_net = a1 + a2 + a3   # net acceleration of the system?
    
    # Calculate the norm of a_net for each sample (magnitude of the system's acceleration?)
    #a_net_norm = np.linalg.norm(a_net, axis=1, keepdims=True)

    X = np.hstack((X, a1, a2, a3))
    #X = np.hstack((X, a_net_norm))

    return X

accel_transformer = FunctionTransformer(func=accel_proxy)


###### CORRELATION MATRIX, PAIRWISE PLOT AND HISTOGRAMS ######

### Run just to create a correlation matrix and a pairwise plot of the feature matrix W/ OR WITHOUT ACCEL - (OTHERWISE DON'T)
features_train_transformed = accel_transformer.transform(features_train)
trainaccel = pd.DataFrame(data=features_train_transformed, columns=['t', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'a_x1', 'a_y1', 'a_x2', 'a_y2', 'a_x3','a_y3'])
#trainaccel.to_csv('trainwithaccel.csv', index=False)

### Calculate the correlation matrix and plot it as a heatmap - with acceleration
corr_matrix = trainaccel.corr()

plt.figure(figsize=(20, 16), dpi=800)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.4f',
            linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('Corr_matrix_heatmap_w_accel.jpg', dpi=800)
plt.show()

### Without acceleration
selectedcolumns = trainaccel.iloc[:, 0:7]
corr_matrix2 = selectedcolumns.corr()

plt.figure(figsize=(20, 16), dpi=800)
sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', fmt='.4f',
            linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('Corr_matrix_heatmap.jpg', dpi=800)
plt.show()


# Select columns 0 to 6 from the trainaccel matrix
selected_columns = trainaccel.iloc[:, 0:7]

# Create a pairwise scatter plot matrix
scatter_matrix = pd.plotting.scatter_matrix(selected_columns, figsize=(30, 30))

plt.suptitle('Pairwise scatter plots of selected variables in the feature matrix')
plt.show()

### Histogram of the distribution of time
plt.hist(trainaccel.iloc[:, 0])
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Distribution of Time')
plt.show()

### Histogram of the distribution of x2
plt.hist(trainaccel.iloc[:, 3])
plt.xlabel('x2')
plt.ylabel('Count')
plt.title('Distribution of x2')
plt.show()

### Histogram of the distribution of y2
plt.hist(trainaccel.iloc[:, 4])
plt.xlabel('y2')
plt.ylabel('Count')
plt.title('Distribution of y2')
plt.show()

### gives the SUMMARY STATS for the feature matrix with acceleration
summary_stats_feat_matrix = trainaccel.describe(include='all')
summary_stats_feat_matrix = pd.DataFrame(data=summary_stats_feat_matrix)
summary_stats_feat_matrix


############################# MODEL TIME! #############################

###### VANILLA MODEL WITH RIDGE REGRESSION ###### - removing unnecessary variables (positions of body 1 and 3)
# Create a pipeline object
vanilla = Pipeline([
    ('column_transformer', ColumnTransformer([
        ("pw_transf", PowerTransformer(method='yeo-johnson'), [3, 4]),
        ("std_scaler", StandardScaler(), [0])
    ], remainder='drop')),
    ('polynomial_features', PolynomialFeatures(18)),
    ('regressor', Ridge(0.001))
])

vanilla.fit(features_train,labels_train)

labels_pred_train = vanilla.predict(features_train)

mse_train = mean_squared_error(labels_train, labels_pred_train, squared=False)
print(f"RMSE Train:\t{mse_train}")

# Make predictions on the validation data using the best model
labels_pred_vali = vanilla.predict(features_vali)

# Evaluate the performance of the best model on the validation data
mse_vali = mean_squared_error(labels_vali, labels_pred_vali, squared=False)
print(f"RMSE Vali:\t{mse_vali}")

# Make predictions on the test data using the best model
labels_pred_test = vanilla.predict(features_test)

# Evaluate the performance of the best model on the test data
mse_test = mean_squared_error(labels_test, labels_pred_test, squared=False)
print(f"RMSE Test:\t{mse_test}")

vanilla

##### TRYING MODEL W/accelerations in the pipeline and Ridge ###### - acceleration does not affect model performance compared to just positions of body 2 and time (SEE CORRELATION MATRIX AS TO WHY)
reg_accel = Pipeline([
    ('acceleration_transformer', accel_transformer),
    ('column_transformer', ColumnTransformer([
        ("std_scaler", StandardScaler(), [0, 7, 8, 9, 10, 11, 12]),
        ("pwr_scaler", PowerTransformer(method='yeo-johnson'), [3, 4]),
    ], remainder='drop')),
    ('polynomial_features', PolynomialFeatures(12)),
    ('regressor', Ridge(10)),
])

# Fit the pipeline using features_train and labels_train
reg_accel.fit(features_train, labels_train)

labels_pred_train = reg_accel.predict(features_train)

mse_train = mean_squared_error(labels_train, labels_pred_train, squared=False)
print(f"RMSE Train:\t{mse_train}")

# Make predictions on the validation data using the best model
labels_pred_vali = reg_accel.predict(features_vali)

# Evaluate the performance of the best model on the validation data
mse_vali = mean_squared_error(labels_vali, labels_pred_vali, squared=False)
print(f"RMSE Vali:\t{mse_vali}")

# Make predictions on the test data using the best model
labels_pred_test = reg_accel.predict(features_test)

# Evaluate the performance of the best model on the test data
mse_test = mean_squared_error(labels_test, labels_pred_test, squared=False)
print(f"RMSE Test:\t{mse_test}")


###### PLOTTING SOME (RANDOM) TRAJECTORIES OF THE VANILLA PARAMETRIC MODEL ######
k = np.random.randint(len(list_of_simulations))
copycopy = list_of_simulations[k].copy()

feats = copycopy.loc[:, ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]
predi = vanilla.predict(feats)

# Define colors for each body
colors = {
    'x1y1': {'real': 'tab:green', 'predict': 'tab:red'},
    'x2y2': {'real': 'tab:blue', 'predict': 'tab:orange'},
    'x3y3': {'real': 'tab:purple', 'predict': 'tab:pink'}
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot x1y1 graph
axes[0].plot(copycopy['x_1_label'], copycopy['y_1_label'], color=colors['x1y1']['real'])
axes[0].plot(predi[:, 0], predi[:, 1], color=colors['x1y1']['predict'])
axes[0].legend(['Real', 'Predict'])
axes[0].set_title('x1y1 graph')

# Plot x2y2 graph
axes[1].plot(copycopy['x_2_label'], copycopy['y_2_label'], color=colors['x2y2']['real'])
axes[1].plot(predi[:, 2], predi[:, 3], color=colors['x2y2']['predict'])
axes[1].legend(['Real', 'Predict'])
axes[1].set_title('x2y2 graph')

# Plot x3y3 graph
axes[2].plot(copycopy['x_3_label'], copycopy['y_3_label'], color=colors['x3y3']['real'])
axes[2].plot(predi[:, 4], predi[:, 5], color=colors['x3y3']['predict'])
axes[2].legend(['Real', 'Predict'])
axes[2].set_title('x3y3 graph')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('plot_vanilla_combined_colored')
plt.show()


###### LAZY LEARNING - K Nearest Neighbours Regression ###### - acceleration does not affect our models performance (See correlation matrix) - ALWAYS choose K value w/GridsearchCV beforehand
knn_reg = Pipeline([
    #('acceleration_transformer', accel_transformer),
    ('column_transformer', ColumnTransformer([
        #("std_scaler", StandardScaler(), [0]),
        ("pwr_scaler", PowerTransformer(method='yeo-johnson'), [0, 3, 4]),
    ], remainder='drop')),
    ('knn_reg', KNeighborsRegressor(n_neighbors=47)),
    ])

# Fit the pipeline using features_train and labels_train
knn_reg.fit(features_train_knn, labels_train_knn)

labels_pred_train_knn = knn_reg.predict(features_train_knn)

mse_train = mean_squared_error(labels_train_knn, labels_pred_train_knn, squared=False)
print(f"RMSE Train:\t{mse_train}")

knn_reg

### GridSearch Best K Value - then plug that value in the above model
# Define K values to be tested
k_values = [33, 35, 37, 39, 41, 43, 45, 47, 49] 
    # Start with some values; If the highest one is picked, erase the smaller ones, plug in higher values and keep trying until the best K value is somewhere in the middle of the tested values

# Parameter grid for GridSearchCV
param_grid = {'knn_reg__n_neighbors': k_values}

# Search Time!
grid_search = GridSearchCV(knn_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)  # using all cores in parallel with n_jobs = -1; we use negative so that the smaller (i.e. less negative, is picked)

# Fit training data
grid_search.fit(features_train_knn, labels_train_knn)

# Best parameters and corresponding RMSE
print("Best Parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))

# Extract the best model - we can run this directly or plug above to see what the model is actually doing (more visual)
best_knn_reg = grid_search.best_estimator_


###### PLOTTING SOME (RANDOM) TRAJECTORIES OF NON-PARAMETRIC (KNN) MODEL ######
p = np.random.randint(len(list_of_simulations))
copycopy_knn = list_of_simulations[p].copy()

feats_knn = copycopy_knn.loc[:, ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]
predi_knn = knn_reg.predict(feats_knn)

# Define colors for each body
colors = {
    'x1y1': {'real': 'tab:green', 'predict': 'tab:red'},
    'x2y2': {'real': 'tab:blue', 'predict': 'tab:orange'},
    'x3y3': {'real': 'tab:purple', 'predict': 'tab:pink'}
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot x1y1 graph
axes[0].plot(copycopy_knn['x_1_label'], copycopy_knn['y_1_label'], color=colors['x1y1']['real'])
axes[0].plot(predi_knn[:, 0], predi_knn[:, 1], color=colors['x1y1']['predict'])
axes[0].legend(['Real', 'Predict'])
axes[0].set_title('x1y1 graph')

# Plot x2y2 graph
axes[1].plot(copycopy_knn['x_2_label'], copycopy_knn['y_2_label'], color=colors['x2y2']['real'])
axes[1].plot(predi_knn[:, 2], predi_knn[:, 3], color=colors['x2y2']['predict'])
axes[1].legend(['Real', 'Predict'])
axes[1].set_title('x2y2 graph')

# Plot x3y3 graph
axes[2].plot(copycopy_knn['x_3_label'], copycopy_knn['y_3_label'], color=colors['x3y3']['real'])
axes[2].plot(predi_knn[:, 4], predi_knn[:, 5], color=colors['x3y3']['predict'])
axes[2].legend(['Real', 'Predict'])
axes[2].set_title('x3y3 graph')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('plot_knn_combined_colored')
plt.show()


##################### REAL WORLD DATA PREDICTIONS ##################### (X_Test)
X_realworld = pd.read_csv('X_test.csv')
id_column = X_realworld['Id']
X_realworld.drop('Id', axis=1, inplace=True)
X_realworld.to_csv('X_realworld.csv', index=False)

### Creating predictions file - Choose the adequate model pipeline name from the models above
# Make predictions on the preprocessed real-world data
X_realworld = pd.read_csv('X_realworld.csv')
X_realworld = X_realworld.to_numpy()
predictions_realworld = knn_reg.predict(X_realworld)


# Create a new Pandas DataFrame with the predictions
df_predictions = pd.DataFrame(predictions_realworld)
df_predictions.insert(loc=0, column='Id', value = id_column)
df_predictions.columns=['Id', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

# .csv file to submit on Kaggle
df_predictions.to_csv('predictions.csv', index=False)   # change the name (either here directly or in the actual file so we know which model, with which parameters created each set of predictions - and describe them when plugging it on kaggle)


###### Creating a ONNX file to store the final model (KNN regression, k=47, Yeo-Johnson) ######
initial_types = [
   ('t', FloatTensorType([None, 1])),
   ('x_1', FloatTensorType([None, 1])),
   ('y_1', FloatTensorType([None, 1])),
   ('x_2', FloatTensorType([None, 1])),
   ('y_2', FloatTensorType([None, 1])),
   ('x_3', FloatTensorType([None, 1])),
   ('y_3', FloatTensorType([None, 1]))
]

model_onnx = convert_sklearn(
   knn_reg, 'pipeline_knn', initial_types=initial_types
)

model_path = "Models\model" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".onnx"


with open(model_path, 'wb') as f:
 f.write(model_onnx.SerializeToString())