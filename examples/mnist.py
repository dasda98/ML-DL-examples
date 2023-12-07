from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

"""

    EXERCISE 3.1
    
        Create classifier using KNN and MNIST database. Requires ~97% accuracy.

"""



if __name__ == '__main__':
    
    # Get data
    mnist = fetch_openml('mnist_784', as_frame=False, data_home='./scikit_learn_data')
    
    X, y = mnist.data, mnist.target
    
    # Split data by train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN Classifier
    knn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
    
    # param_grid = [
    #     {'weights': ['uniform', 'distance'],
    #      'n_neighbors': [3,4,5,6]
    #      }
    # ]
    # grid_search = GridSearchCV(knn_clf, param_grid)
    # grid_search.fit(X_train, y_train)
    
    """
    {'n_neighbors': 4, 'weights': 'distance'}
    0.9721964285714286
    """
    
    knn_clf.fit(X_train, y_train)
    
    accuracy = knn_clf.score(X_test, y_test)
    print(accuracy) # 0.9731428571428572
    