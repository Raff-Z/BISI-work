#

#Choose a dataset from California House or Diabetes
#Or choose exit
def option():
    import matplotlib.pyplot as plt
    import seaborn as sns
    print('Please choose a dataset from California House or Diabetes, or you can choose exit. ')
    option = int(input('Enter 1: California | Enter 2: Diabetes | Enter 3: Exit'))
    if option == 1:
        print('You chose the California House dataset')
        caliLoad()
    elif option == 2:
        print('You chose the Diabetes dataset')
        diabLoad()
    elif option == 3:
        print('You chose to exit, see you next time')
        exit
    else:
        print('You can only choose a dataset from California House or Diabetes, or you can exit')

#Show basic information of California Dataset
#Ask User to choose to continue analysis or change to Diabetes Dataset or Exit
def caliLoad():
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    
    print('California Dataset Shape :' , california.data.shape)
    print('California Dataset Target Varibles :' , california.target )
    print('California Dataset Features Names :' , california.feature_names) 
    print('California Dataset Description:', california.DESCR)
   
    print('Do you want continue to Analysis this database or Change to another one? Or you can exit')
    conti = int(input('Enter 1: Continue | Enter 2: Change to another dataset | Enter 3: Exit'))
    if conti == 1:
        print('You can continue this analysis')
        caliExp()
    elif conti == 2:
        print('Success change database to Diabetes')
        diabLoad()
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Explore the California House dataset
def caliExp():
    from sklearn.datasets import fetch_california_housing     
    california = fetch_california_housing()   
    
    import pandas as pd  
    california_df = pd.DataFrame(california.data, 
                                 columns = california.feature_names)    
    california_df['MedHouseValue'] = pd.Series(california.target)    
    print('Dataset Exploreation :',california_df.head())      
    print('Dataset Description :',california_df.describe())
    
    print('After the explore, do you want continue Analysis this database or Change to another one? Or you can exit')
    conti = int(input('Enter 1: Continue | Enter 2: Change to another dataset | Enter 3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        caliSplit()        
    elif conti == 2:
        print('Success change database to Diabetes')
        diabLoad()    
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Split the California House dataset
def caliSplit():
    from sklearn.datasets import fetch_california_housing     
    california = fetch_california_housing()   
    
    import pandas as pd  
    california_df = pd.DataFrame(california.data, 
                                 columns = california.feature_names)    
    california_df['MedHouseValue'] = pd.Series(california.target)
    
    print(california_df.head())  
    
    california_df.describe()
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state = 11)
    
    print('Training dataset :',X_train.shape)  
    print('Testing dataset :', X_test.shape)
    
    print('After the splited, do you want continue Analysis this database or Change to another one? Or you can exit')
    conti = int(input('Enter 1: Continue | Enter 2: Change to another dataset | Enter 3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        caliTrain()        
    elif conti == 2:
        print('Success change database to Diabetes')
        diabLoad()    
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Training the California House dataset
def caliTrain():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_california_housing     
    california = fetch_california_housing()   
    
    import pandas as pd  
    california_df = pd.DataFrame(california.data, 
                                 columns = california.feature_names)    
    california_df['MedHouseValue'] = pd.Series(california.target)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state = 11)
  
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(california.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')  
       
    print('Model function Intercept :', linear_regression.intercept_)
    
    print('After the training, do you want continue Train this DataModel or Change to another one? Or you can exit')
    conti = int(input('Enter 1: Continue | Enter 2: Change to another dataset | Enter 3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        caliTest()        
    elif conti == 2:
        print('Success change database to Diabetes')
        diabLoad()    
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Test the California DataModel
def caliTest():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_california_housing     
    california = fetch_california_housing()   
    
    import pandas as pd  
    california_df = pd.DataFrame(california.data, 
                                 columns = california.feature_names)    
    california_df['MedHouseValue'] = pd.Series(california.target)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state = 11)
    
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(california.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')  
       
    linear_regression.intercept_
    
    predicted = linear_regression.predict(X_test)
    expected = y_test
    
    print('First 5 predictions' ,predicted[:5])
    print('First 5 targets ', expected[:5])
    
    print('After the Test, do you want continue Visualize this DataModel or Change to another one? Or you can exit')
    conti = int(input('Enter 1: Continue | Enter 2: Change to another dataset | Enter 3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        caliVisua()
    elif conti == 2:
        print('Success change database to Diabetes')
        diabLoad()    
    elif conti == 3:
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Visualize the expected vs. predicted for California House
def caliVisua():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_california_housing     
    california = fetch_california_housing()   
    
    import pandas as pd  
    california_df = pd.DataFrame(california.data, 
                                 columns = california.feature_names)    
    california_df['MedHouseValue'] = pd.Series(california.target)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state = 11)
    
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(california.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')  
       
    linear_regression.intercept_
    
    predicted = linear_regression.predict(X_test)
    expected = y_test
    
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)
    
    figure = plt.figure(figsize = (9, 9))
    axes = sns.scatterplot(data=df, x = 'Expected', y = 'Predicted', 
        hue = 'Predicted', palette = 'cool', legend = False)
    start = min(expected.min(), predicted.min())
    end = max(expected.max(), predicted.max())
    axes.set_xlim(start, end)
    axes.set_ylim(start, end)
    plt.show()
    line = plt.plot([start, end], [start, end], 'k--')   
    
    print('Visulization Chart of Linear Regress Model for California Housing')
    
    print('After the Visulizaiton, do you want continue to Create the model for this DataModel or Change to another one? Or you can exit')
    conti = int(input('Enter 1: Continue | Enter 2: Change to another dataset | Enter 3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        caliMod()
    elif conti == 2:
        print('Success change database to Diabetes')
        diabLoad()    
    elif conti == 3:
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Create the regression model metrics for the California House
def caliMod():
    from sklearn.datasets import fetch_california_housing     
    california = fetch_california_housing()   
    
    import pandas as pd  
    california_df = pd.DataFrame(california.data, 
                                 columns = california.feature_names)    
    california_df['MedHouseValue'] = pd.Series(california.target)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state = 11)
     
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(california.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')  
       
    linear_regression.intercept_
    
    predicted = linear_regression.predict(X_test)
    expected = y_test
    
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)

    from sklearn import metrics
    metrics.r2_score(expected, predicted)
    print('################This is the Final Results################')
    print('Line Regression Estimators Metrics: ')
    print('Final model metrics r2_score is:')
    print('r2_score : ', metrics.r2_score(expected, predicted))

#Show basic information of Diabetes Dataset
#Ask User to choose to continue analysis or change to Califonia House Dataset or Exit
def diabLoad():
    from sklearn.datasets import load_diabetes    
    diabetes = load_diabetes()     
    print(diabetes.DESCR)
   
    print('Diabetes Dataset Shape :' , diabetes.data.shape)
    print('Diabetes Dataset Target Varibles :' , diabetes.target )
    print('Diabetes Dataset Features Names :' , diabetes.feature_names) 
    print('Diabetes Dataset Description:', diabetes.DESCR)
    
    print('Do you want continue to Analysis this database or Change to another one? Or you can exit')
    conti = int(input('Enter1: Continue | Enter2: Change to another dataset | Enter3: Exit'))
    if conti == 1:
        print('You can continue this analysis')
        diabExp()
    elif conti == 2:
        print('Success change database to Califonia House')
        caliLoad()
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Explore the Diabetes dataset
def diabExp():
    from sklearn.datasets import load_diabetes    
    diabetes = load_diabetes()     
    
    import pandas as pd  
    diabetes_df = pd.DataFrame(diabetes.data, 
                             columns=diabetes.feature_names)    
    
    diabetes_df['DiseaseProg'] = pd.Series(diabetes.target)
    diabetes_df.info()
    print('Diabetes Dataset Explore',diabetes_df.head() )    
    print('Diabetes Dataset Description', diabetes_df.describe())
    
    print('After the explore, do you want continue Analysis this database or Change to another one? Or you can exit')
    conti = int(input('Enter1: Continue | Enter2: Change to another dataset | Enter3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        diabSplit()        
    elif conti == 2:
        print('Success change database to Califonia House')
        caliLoad()    
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Split the diabetes dataset
def diabSplit():
    from sklearn.datasets import load_diabetes    
    diabetes = load_diabetes()
    
    import pandas as pd  
    diabetes_df = pd.DataFrame(diabetes.data, 
                             columns = diabetes.feature_names)   
    diabetes_df['DiseaseProg'] = pd.Series(diabetes.target)
    
    print(diabetes_df.head())
    
    diabetes_df.describe()
       
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state = 11)
    
    print('X_train.shape ',X_train.shape)
    print('X_train.shape' ,X_test.shape)

    print('After the splited, do you want continue Analysis this database or Change to another one? Or you can exit')
    conti = int(input('Enter1: Continue | Enter2: Change to another dataset | Enter3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        diabTrain()        
    elif conti == 2:
        print('Success change database to California House')
        caliLoad()    
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Training the Diabetes dataset
def diabTrain():
    from sklearn.datasets import load_diabetes    
    diabetes=load_diabetes()     
    
    import pandas as pd  
    diabetes_df = pd.DataFrame(diabetes.data, 
                             columns = diabetes.feature_names)   
    
    diabetes_df['DiseaseProg'] = pd.Series(diabetes.target)
       
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state = 11)
    
    print('X_train.shape ',X_train.shape)
    print('X_train.shape' ,X_test.shape)
    
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(diabetes.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')    
       
    print('Linear Regression Intercept : ',linear_regression.intercept_)
    
    print('After the training, do you want continue Train this DataModel or Change to another one? Or you can exit')
    conti = int(input('Enter1: Continue | Enter2: Change to another dataset | Enter3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        diabTest()        
    elif conti == 2:
        print('Success change database to California House')
        caliLoad()    
    elif conti == 3: 
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Test the Diabetes DataModel
def diabTest():
    from sklearn.datasets import load_diabetes    
    diabetes = load_diabetes()     
    
    import pandas as pd  
    diabetes_df = pd.DataFrame(diabetes.data, 
                             columns=diabetes.feature_names)   
    
    diabetes_df['DiseaseProg'] = pd.Series(diabetes.target)
       
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state = 11)

    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(diabetes.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')
    
    predicted = linear_regression.predict(X_test)
    expected = y_test
    
    print('First 5 predictions' ,predicted[:5])
    print('First 5 targets ', expected[:5])
    
    print('After the Test, do you want continue Visualize this DataModel or Change to another one? Or you can exit')
    conti = int(input('Enter1: Continue | Enter2: Change to another dataset | Enter3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        diabVisua()
    elif conti == 2:
        print('Success change database to California House')
        caliLoad()    
    elif conti == 3:
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Visualize the expected vs. predicted for Diabetes
def diabVisua():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_diabetes    
    diabetes=load_diabetes()     
    
    import pandas as pd  
    diabetes_df = pd.DataFrame(diabetes.data, 
                             columns = diabetes.feature_names)   
    
    diabetes_df['DiseaseProg'] = pd.Series(diabetes.target)
       
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state = 11)
 
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(diabetes.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')    
    
    predicted = linear_regression.predict(X_test)
    expected = y_test
    
    print('First 5 predictions' ,predicted[:5])
    print('First 5 targets ', expected[:5])
    
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)
    
    figure = plt.figure(figsize = (9, 9))
    axes = sns.scatterplot(data = df, x = 'Expected', y = 'Predicted', 
        hue = 'Predicted', palette = 'cool', legend = False)
    start = min(expected.min(), predicted.min())
    end = max(expected.max(), predicted.max())
    axes.set_xlim(start, end)
    axes.set_ylim(start, end)
    plt.show()
    line = plt.plot([start, end], [start, end], 'k--')
   
    print()
    print()    
    print('Visulization Chart of Linear Regress Model for Diabetes')
    
    print('After the Visulizaiton, do you want continue to Create the model for this DataModel or Change to another one? Or you can exit')
    conti = int(input('Enter1: Continue | Enter2: Change to another dataset | Enter3: Exit'))
    if conti == 1: 
        print('You can continue this analysis')
        diabMod()
    elif conti == 2:
        print('Success change database to Diabetes')
        caliLoad()    
    elif conti == 3:
        print('You chose to exit, see you next time')    
        exit
    else:
        print('You can only choose Continue or Change the dataset, or you can exit')
        
#Create the regression model metrics for the Diabetes
def diabMod():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_diabetes    
    diabetes=load_diabetes()     
    
    import pandas as pd  
    diabetes_df = pd.DataFrame(diabetes.data, 
                             columns = diabetes.feature_names)   
    
    diabetes_df['DiseaseProg'] = pd.Series(diabetes.target)
       
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11)
    
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X = X_train, y = y_train)
    
    for i, name in enumerate(diabetes.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')   
       
    predicted = linear_regression.predict(X_test)
    expected = y_test
    
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)
    
    from sklearn import metrics
    metrics.r2_score(expected, predicted)
    print('################This is the Final Results################')
    print('Final model metrics r2_score is:')
    print('r2_score : ', metrics.r2_score(expected, predicted))

option()