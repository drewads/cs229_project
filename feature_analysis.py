import numpy as np

def main():
    weights = np.genfromtxt('log_reg_weights.csv', delimiter=",")
    red_weights = [weights[i] for i in range(0,len(weights),3)]
    green_weights = [weights[i] for i in range(1,len(weights),3)]
    blue_weights = [weights[i] for i in range(2,len(weights),3)]
    np.savetxt("red_log_reg_weights.csv", red_weights, delimiter=",")
    np.savetxt("green_log_reg_weights.csv", green_weights, delimiter=",")
    np.savetxt("blue_log_reg_weights.csv", blue_weights, delimiter=",")

if __name__ == '__main__':
    main()