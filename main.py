import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

### various toggles for debugging and testing ##
check_boot = 0

### Define functions ###########################
def bootstrap_sample(data, num_samples, pts_per_sample):
    # from data, take num_samples samples where each sample has pts_per_sample data points
    # n = num_samples
    boot_samples = np.zeros((pts_per_sample, num_samples))

    # test to clean up input
    if isinstance(data, pd.DataFrame):
        # redo this later to check if it's np.array and make it one if not
        data = data.to_numpy().flatten()
    if np.isnan(data).any():
        data = data[~np.isnan(data)]

    rng = np.random.default_rng()

    for i in range(0, num_samples):
        # print(rng.choice(data, (pts_per_sample, 1), replace=True))
        # print(np.shape(boot_samples[:,i]))
        boot_samples[:, i] = rng.choice(data, (pts_per_sample,), replace=True)

    return boot_samples

def compute_cost(x, y, w):
    # compute the cost for poisson distribution where b = 0; uses squared error cost function
    total_cost = 0
    f_wb = (w**x)*np.exp(-w)/sp.special.factorial((x))
    cost = (f_wb - y)**2
    total_cost = 1 / 2 / len(x) * np.sum(cost)

    return total_cost

def compute_gradient(x, y, w):
    # compute gradient of poisson distribution where b = 0
    dj_dw = 0
    m = len(x)
    f_wb = (w**x)*np.exp(-w)/sp.special.factorial((x))
    rhs = (1/sp.special.factorial((x))) * (x*w**(x-1) - (w**x)*np.exp(-w))
    dj_dw = (1/m)*(f_wb - y) * rhs
    dj_dw = np.sum(dj_dw)/x.shape[0]

    return dj_dw

def gradient_descent(x, y, w_in, cost_function, gradient_function, alpha, num_iters, tol):
    # x and y are the input training data
    # w_in is the initial guess for the parameter w (in this case, the lambda of the poisson distribution)
    # cost_function and gradient_function are defined above to compute cost and gradient
    # alpha is the learning rate, num_iters the max number of iterations to do
    # tol is the % change in cost_history relative to previous step to trigger exit of the loop, 0 < tol < 1
    # cost_history and w_history are for plotting later to make sure everything works right
        w_i = w_in
        cost_history = []
        w_history = []

        for i in range(num_iters):
            print(i)
            dj_dw = gradient_function(x, y, w_i)

            w_i = w_i - alpha * dj_dw

            if i < 100000:  # prevent resource exhaustion while recording cost evolution
                cost = cost_function(x, y, w_i)
                cost_history.append(cost)
                w_history.append(w_i)
            #if i > 2 and abs((cost_history[-1]-cost_history[-2])/cost_history[-2]) <= tol:
                #break

        print(f"Cost {float(cost_history[-1]):8.2f}   ")

        return w_i, cost_history, w_history


### Import data ################################
data_30kv = pd.read_csv(r'30kv.csv')
all_tools_data30kv = pd.concat([data_30kv['Pr15'],
                            data_30kv['Pr13'],
                            data_30kv['Pr12'],
                            data_30kv['Pr11'],
                            data_30kv['Pi1'],
                            data_30kv['Pi2']], axis=1)

data_27kv = pd.read_csv(r'27kv.csv')
all_tools_data27kv = pd.concat([data_27kv['Pr15'],
                            data_27kv['Pr13'],
                            data_27kv['Pr12'],
                            data_27kv['Pr11'],
                            data_27kv['Pi1'],
                            data_27kv['Pi2']], axis=1)
################################################

## Bootstrap a distribution of the number of arcs per quarter for num_samples samples of 13 weeks of arcing data
## Check convergence of the mean and stdev of this distribution for increasing num_samples
# if check_boot == 1:
#     num_samples = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
#     boot_means = np.zeros((20, len(num_samples)))
#     boot_stdevs = np.zeros((20, len(num_samples)))
#     for i in range(20):
#         for j in range(len(num_samples)):
#             temp_arcs = bootstrap_sample(all_tools_data, num_samples[j], 13)
#             temp_sums = np.sum(temp_arcs, axis=0)
#             boot_means[i,j] = np.mean(temp_sums)
#             boot_stdevs[i,j] = np.std(temp_sums)
#
#     fig2 = plt.boxplot(boot_means)
#     plt.show()
#     fig3 = plt.boxplot(boot_stdevs)
#     plt.show()


## Use previous results to bootstrap data with appropriate num_samples
num_samples = 2000
boot_qarcs_30 = bootstrap_sample(all_tools_data30kv, num_samples, 13)
boot_qrate_30 = np.sum(boot_qarcs_30, axis=0)
# plt.hist(boot_qrate_30, density=False, bins=(int(1+3.3*np.log(num_samples))))

boot_qarcs_27 = bootstrap_sample(all_tools_data27kv, num_samples, 13)
boot_qrate_27 = np.sum(boot_qarcs_27, axis=0)
plt.hist([boot_qrate_27,boot_qrate_30], label=['-27 kV', '-30 kV'], density=True, log=False, bins=(int(1+3.3*np.log(num_samples))))
plt.legend(loc='upper right')
plt.xlabel('Average arcs per quarter')
plt.ylabel('Percentage of samples with each arcing rate')
plt.show()



