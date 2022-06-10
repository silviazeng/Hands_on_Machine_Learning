#-----Download Data-----
import os
import tarfile  #tar = "tape archive", Linux's couterpart of zip files
from six.moves import urllib
import pandas as pd

os.getcwd()
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")  #local directory to store the data*
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


#-----Preview Data-----
pd.set_option('display.max_columns', None)
housing=load_housing_data()
print(housing.head())
print(housing.info()) #including number of non-null values
print(housing.describe())

print(housing["ocean_proximity"].value_counts())

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(10,10))  #figsize = figure size
plt.show()


#-----Create a Test Set-----
import numpy as np
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_test_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


#-----Discover and Visualize the Data to Gain Insights-----
housing=strat_train_set.copy() #create a copy to play with without harming training set
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    #Setting alpha to 0.1 makes it easier to visualize the places where there is a high density of data points
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True) #s: size of bubble
plt.legend()
plt.show()


corr_matrix = housing.corr()
    # The correlation coefficient only measures linear correlations (“if x goes up,
    # then y generally goes up/down”). It may completely miss out on nonlinear
    # relationships (e.g., “if x is close to zero then y generally goes up”).

#another way to check for corr:
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

#Experimenting with Attribute Combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


#--------------------------------------------------------------------
#-----Preprocessing Data-----
# separate the predictors and the labels since we don’t necessarily want to apply
# the same transformations to the predictors and the target values (note that drop()
# creates a copy of the data and does not affect strat_train_set)
housing = strat_train_set.drop("median_house_value", axis=1)  #X
housing_labels = strat_train_set["median_house_value"].copy() #y

#For missing data in "total_bedrooms":
  # option 1: Get rid of the corresponding districts (subset defines in which col. to look for NA)
housing.dropna(subset=["total_bedrooms"])

  # option 2: Get rid of the whole attribute
housing.drop("total_bedrooms", axis=1)

  # option 3.1: Set the values to some value (zero, the mean, the median, etc.)
total_bedrooms_median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(total_bedrooms_median, inplace=True)
    # don’t forget to save the median that you have computed. You will need it later to replace missing values
    # in the test set when you want to evaluate your system, and also once the system goes live to replace missing values in new data.

  # option 3.2: using sklearn.impute
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") #First, you need to create a SimpleImputer instance
housing_num = housing.drop("ocean_proximity", axis=1)
    #Since the median can only be computed on numerical attributes, we need to create a copy of the data without the text attribute ocean_proximity
imputer.fit(housing_num)
  #print(imputer.statistics_) #The imputer computed the median of each attribute and stored the result in statistics_ (equal to housing_num.median().values)
X = imputer.transform(housing_num) #Now use this “trained” imputer to transform the training set by replacing missing values by the learned medians
type(X) # The result is a plain NumPy array
housing_tr = pd.DataFrame(X, columns=housing_num.columns) #If you want to put it back into Pandas DataFrame,
type(housing_tr)


#Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.value_counts())

  #Label Encoding
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()  #LabelEncoder() is for y, OrdinalEncoder() is for X (tho not really 'ordinal' unless "OrdinalEncoder(categories='cold','warm','hot'))
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

  #One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
print(type(housing_cat_1hot)) #output is a SciPy sparse matrix, instead of a NumPy array.
        #After onehot encoding we get a matrix with thousands of columns, and the matrix is full of zeros except for a single 1 per row. Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the non‐zero elements. You can use it mostly like a normal 2D array but if you really want to convert it to a (dense) NumPy array, just call the toarray() method:
housing_cat_1hot.toarray()

#Custom Transformer
  # You will want your transformer to work seamlessly with Scikit-Learn functionalities, and since Scikit-Learn relies on duck typing (not inheritance),all you need is to create a class and implement three methods: fit()(returning self), transform(), and fit_transform(). You can get the last one for free by simply adding TransformerMixin as a base class. Also, if you add BaseEstima tor as a base class (and avoid *args and **kargs in your constructor) you will get two extra methods (get_params() and set_params()) that will be useful for automatic hyperparameter tuning.
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    #let it inherit both BaseEstimator and TransformerMinxin)
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    #the transformer has one hyperparameter, add_bedrooms_per_room, set to True by default (it is often helpful to provide sensible defaults). This hyperparameter will allow you to easily find out whether adding this attribute helps the Machine Learning algorithms or not.
housing_extra_attribs = attr_adder.transform(housing.values)

attr_adder.get_params()
attr_adder.set_params()

#Numerical Feature Scaling
  #As with all the transformations, it is important to fit the scalers to the training data only, not to the full dataset (including the test set). Only then can you use them to transform the training set and the test set (and new data).
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#Pipeline
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)
    #Note that the OneHotEncoder returns a sparse matrix, while the num_pipeline returns a dense matrix. When there is such a mix of sparse and dense matrices, the ColumnTransformer estimates the density of the final matrix (i.e., the ratio of non-zero cells), and it returns a sparse matrix if the density is lower than a given threshold (by default, sparse_threshold=0.3). In this example, it returns a dense matrix.


#--------------------------------------------------------------------
#-----Train and Evaluate on the Training set-----
from sklearn.metrics import mean_squared_error

  #-----Model 1: Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression_RMSE: %d'%lin_rmse)

  #-----Model 2: Decision Tree-----
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels, housing_predictions)
tree_rmse=np.sqrt(tree_mse)
print('Tree Regressor_RMSE: %d' %tree_rmse)

  #-----Model 3: RandomForestRegressor-----
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions=forest_reg.predict(housing_prepared)
forest_rmse=np.sqrt(mean_squared_error(housing_labels, housing_predictions))
print('Random Forest Regressor_RMSE: %d' %forest_rmse)

  #cross validation score

from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores) #return the score on validation set
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10) #cv =cross-validation
    #Scikit-Learn’s cross-validation features expect a utility function(greater is better) rather than a cost function (lower is better), so the scoring function is actually the opposite of the MSE (i.e., a negative value), which is why the preceding code computes -scores before calculating the square root.
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)
    #Notice that cross-validation allows you to get not only an estimate of the performance of your model, but also a measure of how precise this estimate is (i.e., its standard deviation). The Decision Tree has a score of approximately 71,407, generally ±2,439.

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
  #the Decision Tree model is overfitting so badly (RMSE=0) that it performs worse than the Linear Regression model(higher score mean).

forest_scores=cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
forest_rmse_cores = np.sqrt(-forest_scores)
display_scores(forest_rmse_cores)
    #Random Forests look very promising. However, note that the score on the training set is still much lower than on the validation sets, meaning that the model is still overfitting the training set. Possible solutions for overfitting are to simplify the model, constrain it (i.e., regularize it), or get a lot more training data. However, before you dive much deeper in Random Forests, you should try out many other models from various categories of Machine Learning algorithms (several Support Vector Machines with different kernels, possibly a neural network, etc.), without spending too much time tweaking the hyperparameters. The goal is to shortlist a few (two to five) promising models.

import joblib
joblib.dump(forest_reg, "forest_reg.pkl")
# and later...
my_model_loaded = joblib.load("forest_reg.pkl")

#---------------------------------------------------------------
#-----Fine-Tune the Model-----
  #Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    #This param_grid tells Scikit-Learn to first evaluate all 3 × 4 = 12 combinations of n_estimators and max_features hyperparameter values specified in the first dict, then try all 2 × 3 = 6 combinations of hyperparameter values in the second dict, but this time with the bootstrap hyperparameter set to False instead of True (which is the default value for this hyperparameter).
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_ #You can also get best estimator directly
cvres = grid_search.cv_results_
list(cvres)
for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):

  #Randomized Search
  # The grid search approach is fine when you are exploring relatively few combinations, like in the previous example, but when the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV instead. Instead of trying out all possible combinations,it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

   #Ensemble Method
   # Another way to fine-tune your system is to try to combine the models that perform best. The group (or “ensemble”) will often perform better than the best individual model (just like Random Forests perform better than the individual Decision Trees they rely on), especially if the individual models make very different types of errors. We will cover this topic in more detail in Chapter 7.

   #Analyze the Best Models and Their Errors
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


#-----Evaluate on the Test Set-----
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),scale=stats.sem(squared_errors)))
    #df: degree of freedom; sem: Standard Error of mean
