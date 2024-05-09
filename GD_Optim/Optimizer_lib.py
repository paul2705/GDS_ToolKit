from bayes_opt import BayesianOptimization
import numpy as np

# Function that queries your API
def query_api(x, y):
    # Dummy implementation, replace with actual API call
    return -((x - 1)**2 + (y - 2)**2) + np.random.normal(0, 0.1)
    # ret = float(input(f"Input Coordination {x,y} Value:"))
    # return ret

def bayesianOptimize(queryAPI, xBound, yBound, Iterations):
    # Define the bounds of your search space
    pbounds = {'x': (-10, 10), 'y': (-10, 10)}

    # Instantiate a BayesianOptimization object
    optimizer = BayesianOptimization(f=queryAPI, pbounds=pbounds, random_state=1)

    # Perform optimization
    optimizer.maximize(init_points=5, n_iter=Iterations)

    return optimizer.max

ret = bayesianOptimize(query_api, None, None, 20)
print("Optimal coordinates:", ret ['params'])
print("Optimal value:", ret ['target'])