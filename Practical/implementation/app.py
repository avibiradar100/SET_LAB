# importing packages
import streamlit as st
import pandas as pd 
import numpy as np
import helper
import scipy.stats
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
  
arr=["1","2","3","4","5","6","7","8","9","10"]
assignNo=st.sidebar.selectbox("Assignment No",arr)
# st.title("Data Mining")
st.title("Asssignment No."+assignNo)
st.sidebar.title("Select the dataset....")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file :
    df  = pd.read_csv(uploaded_file)

    # data shape and checking whether any attribute contains null values
    # st.text(df.shape)
    # st.text(df.isna().any())

    st.header("Dataset")
    st.write(pd.DataFrame(df))
    st.text("")
    if assignNo=="1":
        #filterning according to attribute and class
        colums=df.columns
        attribute= st.sidebar.selectbox("Select attribute",colums)

        # Measures of central tendency
        st.header("Measures of central tendency")
        st.text("")

        col1, col2, col3= st.columns(3)
        data=df[attribute].to_list()
        
        with col1:
            # Mean
            st.subheader('Mean')
            st.write(helper.mean(data))
        with col2:
            # Median
            st.subheader('Median')
            st.write(helper.median(data))
        with col3:
            # Mode
            st.subheader('Mode')
            st.write(helper.mode(data))
        st.text("")
        
        col1, col2, col3= st.columns(3)
        with col1:
            #Mid Range
            st.subheader("Mid Range (max+min)/2")
            st.write(round((max(data)+min(data))/2,3))
        with col2:
            # Variance
            st.subheader('Variance')
            st.write(helper.variance(data))
        with col3:
            # Standard Deviation
            st.subheader('standard deviation')
            st.write(helper.stddeviation(data))
        st.text("")
    
        #Dispersion of data
        st.header("Dispersion of data")
        
        length=len(df)
        data=df[attribute].to_list()

        col1, col2, col3= st.columns(3)
        with col1:
            #Range
            st.subheader("Range (max-min)")
            st.write(round(max(data)-min(data),3))
        with col2:
            #Quartile Q1
            st.subheader("Quartile (Q1)")
            Q1=helper.median(data[0:length//2])
            st.write(round(Q1,3))
        with col3:
            #Quartile Q2
            st.subheader("Quartile (Q2)")
            Q2=helper.median(data)
            st.write(round(Q2,3))
        st.text("")
        
        col1, col2= st.columns(2)
        with col1:
            #Quartile Q3
            st.subheader("Quartile (Q3)")
            Q3=helper.median(data[length//2:])
            st.write(round(Q3,3))
        with col2:
            #IQR
            st.subheader("Interquartile range (Q3-Q1)")
            Q1=helper.median(data[0:length//2])
            Q3=helper.median(data[length//2:])
            st.write(round(Q3-Q1,3))
        st.text("")
        
        # Five Number Summary
        st.subheader("Five Number Summary")
        col1, col2, col3 ,col4,col5 = st.columns(5)
        data.sort()
        with col1:
            st.text('Min')
            st.write(min(data))
        with col2:
            st.text('Q1')
            st.write(helper.median(data[0:length//2]))
        with col3:
            st.text('Median')
            st.write(helper.median(data))
        with col4:
            st.text('Q2')
            st.write(helper.median(data[length//2:]))
        with col5:
            st.text('max')
            st.write(max(data))
        st.text("")

        #GRaphical Representation:
        st.header('Graphical Representation:')

        #Histogram optin x Count
        plt.rcParams['figure.figsize'] = [8, 4]
        st.write("Histogram")
        fig, ax = plt.subplots()
        plt.locator_params(nbins = 15)
        plt.xlabel(attribute)
        plt.ylabel("count")
        ax.hist(data)
        st.pyplot(fig)

        plt.clf()
        
        st.text("")
        st.text("")
        st.text("")

        # Scatter plot
        xlabel = st.selectbox("xLabel",df.columns)
        ylabel = st.selectbox("yLabel",df.columns)
        plt.locator_params(nbins = 10)
        plt.scatter(df[xlabel],df[ylabel], c ="green", s=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.rcParams['figure.figsize'] = [8, 4]
        st.write("Scatter Plot")
        st.pyplot(plt)
        plt.clf()

        indices = []
        ind=0
        for i in df.columns:
            if df.dtypes[i]!=object:
                indices.append(ind)
            ind+=1
        st.text("")
        st.text("")
        st.text("")
        #box plot
        # df.columns = ["sepal length","sepal width","petal length","petal width", "species" ]
        st.write("Box Plot")
        data= df.iloc[:,indices].values
        fig = plt.figure(figsize =(10, 7))
        plt.boxplot(data)
        # plt.show()
        st.pyplot(plt)

        # Verify Results:
        st.subheader('Verify Results:')
        st.write(df.describe())
    if assignNo=="2":
        colums=df.columns
        st.header("Chi Sqaure Test")
        st.sidebar.subheader("Select Attributes For Chi square")
        att1= st.sidebar.selectbox("Select attribute 1",colums)
        att2= st.sidebar.selectbox("Select categarical attribute",colums)

        contigency= pd.crosstab(df[att1], df[att2],margins=True) 
        st.subheader("Contingency/Observed Table")
        st.text(contigency)

        row_len=len(df[att1].unique())
        col_len=len(df[att2].unique())
        row_sum = contigency.iloc[0:row_len,col_len].values
        exp = []
        for j in range(row_len):
            for val in contigency.iloc[row_len,0:col_len].values:
                exp.append(val * row_sum[j] / contigency.loc['All', 'All'])
        
        obs=[]
        for j in range(row_len):
            for val in contigency.iloc[j,0:col_len].values:
                obs.append(val)

        #Expected Table
        expArr=np.array(exp).reshape(row_len,col_len)
        st.subheader("expected Table")
        st.write(expArr)
        
        #Degree of Freedom
        st.subheader('Degree of Freedom')
        st.write("no of Rows:",row_len)
        st.write("no of Columns:",col_len)
        degreeOfFreedom=(row_len-1)*(col_len-1)
        st.write("(row-1)*(col-1)=",degreeOfFreedom)


        #((obs[i] - exp[i])^2/exp[i])
        st.subheader('(Obs[i]-exp[i])^2/exp[i]')
        objmexp=[]
        chiSquareValue=0
        for i in range(len(obs)):
          chiSquareValue+=((obs[i] - exp[i])**2/exp[i])
          objmexp.append((obs[i] - exp[i])**2/exp[i])
        objmexp=np.array(objmexp).reshape(row_len,col_len)
        st.write(objmexp)

        #chi Square Value
        st.subheader("conclusion of Chi Sqaure Test")
        st.write("Chi sqaure Value:",chiSquareValue)

        criticalValue = scipy.stats.chi2.ppf(1-.001, df = degreeOfFreedom)
        st.write("Critical Value:",criticalValue)

        if(criticalValue > chiSquareValue):
            st.write("chiSquare Value is less than critical Value so,They are independent")
        else:
            st.write("chiSquare Value is greater than critical Value so,They are Correlated")
        
        # covariance
        st.sidebar.subheader("Select Attr For Covaraince & Pearson")
        attr1= st.sidebar.selectbox("Select attr 1",colums)
        attr2= st.sidebar.selectbox("Select attr 2",colums)
        st.header(" covariance")
        data1=df[attr1].to_list()
        data2=df[attr2].to_list()
        
        xm =helper.mean(data1)
        ym=helper.mean(data2)
        n=len(data1)
        
        covariance=0.0
        for i in range(n):
            covariance += (data1[i]-xm)*(data2[i]-ym)/(n-1)
            
        st.write("The Covariance is :",covariance)

        # Pearson coefficient
        st.header("Pearson coefficient")

        stdD1=helper.stddeviation(data1)
        stdD2=helper.stddeviation(data2)

        pearson=(covariance/(stdD1*stdD2))
        st.write("The Pearson coefficient is :",pearson)

        #conclusion of perason test
        res=""
        if(pearson>0):
          res="Positively Correlated"
        elif(pearson<0):
          res="Negatively Correlated"
        else:
          res="Independent"
        st.write("conclusion:",res)

        #Normalization
        st.header("Normalization")
        st.subheader("(Decimal Scaling,Min Max,Z-Score)")
        st.text("")
         
        st.subheader("Select Attr For Normalization")
        att= st.selectbox("Select attr",colums)

        #decimal Scaling
        st.header("Decimal Scaling")
        data=df[attr1].to_list()
        n=len(data)
        denom=pow(10,len(str(max(data))))
    
        decimal_scaling=[]
        for val in data:
            decimal_scaling.append(val/denom)
        decimal_scaling.sort()
        st.text(decimal_scaling)
        #scatter plot of decimal scaling
        plt.locator_params(nbins = 10)
        plt.scatter(decimal_scaling,decimal_scaling, c ="green", s=5)
        plt.xlabel(att)
        plt.ylabel(att)
        # plt.rcParams['figure.figsize'] = [8, 4]
        st.write("Scatter Plot")
        st.pyplot(plt)
        plt.clf()

        #Min-Max Scaling
        st.header("2.Min-Max Normalization")
        xmin=min(data)
        xmax=max(data)
        lmin=0 #local min
        lmax=1 #local max
        minMax=[]
        if xmin==xmax:
            st.write("denominator became zero because min and max are same")
        else:
            for val in data:
                minMax.append((val-xmin)/(xmax-xmin)*(lmax-lmin)+lmin) 
        
        st.text(minMax)
        #scatter plot of decimal scaling
        plt.locator_params(nbins = 10)
        plt.scatter(minMax,minMax, c ="green", s=5)
        plt.xlabel(att)
        plt.ylabel(att)
        # plt.rcParams['figure.figsize'] = [8, 4]
        st.write("Scatter Plot")
        st.pyplot(plt)
        plt.clf()

        #Z-Score Scaling
        st.header("3.Z-Score Normalization")
        mean=helper.mean(data)
        stdD=helper.stddeviation(data)
        
        z_score=[]
        for val in data:
            z_score.append((val-mean)/stdD)
        st.text(z_score)

        #scatter plot of decimal scaling
        plt.locator_params(nbins = 10)
        plt.scatter(z_score,z_score, c ="green", s=5)
        plt.xlabel(att)
        plt.ylabel(att)
        # plt.rcParams['figure.figsize'] = [8, 4]
        st.write("Scatter Plot")
        st.pyplot(plt)
        plt.clf()
    if assignNo=="3":
        colums=df.columns

        targetAttr=st.sidebar.selectbox("Choose Target Attribute",colums)       
        st.header("Decision Tree")
        data=df
        features = list(colums)
        features.remove(targetAttr)

        def entropy(labels):
            entropy=0
            label_counts = Counter(labels)
            for label in label_counts:
                prob_of_label = label_counts[label] / len(labels)
                entropy -= prob_of_label * math.log2(prob_of_label)
            return entropy

        def information_gain(starting_labels, split_labels):
            info_gain = entropy(starting_labels)
            ans=0
            for branched_subset in split_labels:
                ans+=len(branched_subset) * entropy(branched_subset) / len(starting_labels)
            st.write("entropy:",ans)
            info_gain-=ans
            return info_gain

        def split(dataset, column):
            split_data = []
            col_vals = data[column].unique()
            for col_val in col_vals:
                split_data.append(dataset[dataset[column] == col_val])

            return(split_data)

        def find_best_split(dataset):
            best_gain = 0
            best_feature = 0
            st.subheader("Overall Entropy:")
            st.write(entropy(dataset[targetAttr]))
            for feature in features:
                split_data = split(dataset, feature)
                split_labels = [dataframe[targetAttr] for dataframe in split_data]
                st.subheader(feature)
                gain = information_gain(dataset[targetAttr], split_labels)
                st.write("Gain:",gain)
                if gain > best_gain:
                    best_gain, best_feature = gain, feature
            st.subheader("Highest Gain:")
            st.write(best_feature, best_gain)
            return best_feature, best_gain

        new_data = split(data, find_best_split(data)[0]) 
        # for i in new_data:
        #    st.write(i)

        features = list(colums)
        features.remove(targetAttr)
        x = df[features]
        y = df[targetAttr] # Target variable

        dataEncoder = preprocessing.LabelEncoder()
        encoded_x_data = x.apply(dataEncoder.fit_transform)

        st.header("1.Information Gain")
        # "leaves" (aka decision nodes) are where we get final output
        # root node is where the decision tree starts
        # Create Decision Tree classifer object
        decision_tree = DecisionTreeClassifier(criterion="entropy")
        # Train Decision Tree Classifer
        decision_tree = decision_tree.fit(encoded_x_data, y)
        
        #plot decision tree
        fig, ax = plt.subplots(figsize=(6, 6)) 
        #figsize value changes the size of plot
        tree.plot_tree(decision_tree,ax=ax,feature_names=features)
        plt.show()
        st.pyplot(plt)

        st.header("2.Gini Index")
        decision_tree = DecisionTreeClassifier(criterion="gini")
        # Train Decision Tree Classifer
        decision_tree = decision_tree.fit(encoded_x_data, y)
        
        fig, ax = plt.subplots(figsize=(6, 6)) 
        tree.plot_tree(decision_tree,ax=ax,feature_names=features)
        plt.show()
        st.pyplot(plt)

        X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth=2, random_state=1)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        c_matrix = confusion_matrix(y_test, y_pred)

        tp = c_matrix[1][1]
        tn = c_matrix[2][2]
        fp = c_matrix[1][2]
        fn = c_matrix[2][1]


        st.subheader("confusion Matrix:")
        st.write(c_matrix)

        # Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :
        st.write('Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :')

        
        st.write("Model Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
        # precision score
        val = metrics.precision_score(y_test, y_pred, average='macro')
        print('Precision score : ' + str(val))
        st.write('Precision score : ' + str(val))


        # Accuracy score
        val = metrics.accuracy_score(y_test, y_pred)
        st.write('Accuracy score : ' + str(val))

       
        st.header("Rule Base Classifier")
        # get the text representation
        text_representation = tree.export_text(clf,feature_names=features)
        st.text(text_representation)

        #Extract Code Rules
        st.subheader("Extract Code Rules")



        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
            st.text("def predict({}):".format(", ".join(feature_names)))

            def recurse(node, depth):
                indent = "    " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    st.text("{}if {} <= {}:".format(indent, name, np.round(threshold,2)))
                    recurse(tree_.children_left[node], depth + 1)
                    st.text("{}else:  # if {} > {}".format(indent, name, np.round(threshold,2)))
                    recurse(tree_.children_right[node], depth + 1)
                else:
                    st.text("{}return {}".format(indent, tree_.value[node]))

            recurse(0, 1)
        
        tree_to_code(decision_tree,features)

