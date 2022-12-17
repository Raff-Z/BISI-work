#%matplotlib inline
#from sklearn.datasets import fetch_california_housing
#dataset = fetch_california_housing()
def description():
    print('*********************Let us start our Description process*****************************')
    import pandas as pd
    #print(california.DESCR)
    print('1. The number of samples in dataset(rows, columns):' )
    print(dataset.data.shape)
    print()
    print('2. The feature names in dataset:')
    print(dataset.feature_names)
    print()
    dataset_df = pd.DataFrame(dataset.data, 
                columns=dataset.feature_names)
    dataset_df['MedHouseValue'] = pd.Series(dataset.target)
    print('3. The first five row of the dataset:')
    print(dataset_df.head())
    print()
    print('4. Summary of the descriptive statistics for the dataset')
    print(dataset_df.describe())


def description2():
    print('*********************Let us start our Description process*****************************')
    import pandas as pd
    #print(diabetes.DESCR)
    print('1. The number of samples in dataset(rows, columns):' )
    print(dataset.data.shape)
    print()
    print('2. The feature names in dataset:')
    print(dataset.feature_names)
    print()
    dataset_df = pd.DataFrame(dataset.data, 
                columns=dataset.feature_names)
    dataset_df['DiseaseProg'] = pd.Series(dataset.target)
    print('3. The first five row of the dataset:')
    print(dataset_df.head())
    print()
    print('4. Summary of the descriptive statistics for the dataset')
    print(dataset_df.describe())

#description()

#%matplotlib inline
#from sklearn.datasets import fetch_california_housing
#dataset = fetch_california_housing()
def virtualization():
    print('**********************Let us start our Virtualization process**************************')
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    dataset_df = pd.DataFrame(dataset.data, 
                columns=dataset.feature_names)
    dataset_df['MedHouseValue'] = pd.Series(dataset.target)
    sample_df = dataset_df.sample(frac=0.1, random_state=17)
    print('Each shows feature on x-axis and median home value on y-axis')
    for feature in dataset.feature_names:
        plt.figure(figsize=(8, 4.5))
        sns.scatterplot(data=sample_df, x=feature, 
                    y='MedHouseValue', hue='MedHouseValue', 
                    palette='cool', legend=False)


def virtualization2():
    print('**********************Let us start our Virtualization process**************************')
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    dataset_df = pd.DataFrame(dataset.data, 
                columns=dataset.feature_names)
    dataset_df['DiseaseProg'] = pd.Series(dataset.target)
    sample_df = dataset_df.sample(frac=0.1, random_state=17)
    print('Each shows feature on x-axis and median home value on y-axis')
    for feature in dataset.feature_names:
        plt.figure(figsize=(8, 4.5))
        sns.scatterplot(data=sample_df, x=feature, 
                    y='DiseaseProg', hue='DiseaseProg', 
                    palette='cool', legend=False)

#virtualization()

#%matplotlib inline
#from sklearn.datasets import fetch_california_housing
#dataset = fetch_california_housing()
def ML():
    print('*************************Let us start our Mahine Learning process*********************************')
    print('Splitting the Data for Training and Testing')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=11)
    print('The number of training dataset sapmles:(rows, colunms)')
    print(X_train.shape)
    print('The number of testing dataset sapmles:(rows, colunms)')
    print(X_test.shape)
    print('Training the Model')
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X=X_train, y=y_train)
    print('the coefficients of a linear regression model:')
    for i, name in enumerate(dataset.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')
    print('The y-intercept of the linear regression line:')
    print(linear_regression.intercept_)
    from sklearn import metrics
    predicted = linear_regression.predict(X_test)
    expected = y_test
    print('Metrics for regression estimators,coefficient of determination 𝑅2 score:')
    print('R2 value ranges from 0 (no predictive power) to 1 (perfect predictive power)')
    metrics.r2_score(expected, predicted)    

#ML()


print('Please choose the dataset: A.California Housing dataset, B.Diabetes dataset')
dataset = input('Please enter A or B:')
if dataset == 'A':
    from sklearn.datasets import fetch_california_housing
    dataset = fetch_california_housing()
    description()
    conti_1= input('Do you want to continue? if yes,please enter y, if not,please enter n')
    if conti_1 == 'y':
        virtualization()
        conti_2= input('Do you want to continue? if yes,please enter y, if not,please enter n')
        if conti_2 == 'y':
          ML()
        else:
            print('Thank you for your participation,see you next time') 
    else:
        print('Thank you for your participation,see you next time' )   
elif dataset == 'B':
    from sklearn.datasets import load_diabetes
    dataset = load_diabetes()
    description2()
    conti_1= input('Do you want to continue? if yes,please enter y, if not,please enter n')
    if conti_1 == 'y':
        virtualization2()
        conti_2= input('Do you want to continue? if yes,please enter y, if not,please enter n')
        if conti_2 == 'y':
          ML()
        else:
            print('Thank you for your participation,see you next time') 
    else:
        print('Thank you for your participation,see you next time')
else:
    print('see you!')