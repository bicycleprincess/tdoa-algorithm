from calculator import V, Calculator
from convertions import dist

import numpy as np
import matplotlib.pyplot as plt


# define a field
L = 6

def establishScenario(n, dim):

    rn = L * np.random.rand(dim, n)
    #print rn
    bn = L * np.random.rand(dim, 1)

    doa = dist(rn, bn, 0)
    toa = doa / V

    noise = np.zeros(0)
    for i in range(len(doa)):
        noise = np.append(noise, np.random.randn()) * 1e-6

    toa = toa + noise

    ary = np.vstack((rn, toa))
    return ary.T, bn

if __name__ == '__main__':

    """
        (3, 2)
        (4, 2)
        (4, 3)
        (5, 3)
    """

    ary, bn = establishScenario(5, 2)
    a_shape = np.shape(ary)
    b_shape = np.shape(bn)

    rechner = Calculator(ary)
    est_ml = [rechner.ml()]
    est_tls = [rechner.tls()]
    est_ml_5 = [rechner.ml_5()]

    n, dim = np.shape(ary)

    if b_shape == (2,1):

        print('the unknown is locating at: ', bn[0][0], bn[1][0], '\n')

        if len(est_ml) == 2:

            x0, y0 = est_ml[0][0], est_ml[0][1]
            x1, y1 = est_ml[1][0], est_ml[1][1]
            print('the first estimation of maximun likelihoos is: ', x0, y0)
            print('the alternative estimation of maximun likelihoos is: ', x1, y1)
            plt.plot(x0, y0, 'o', color='g')
            plt.plot(x1, y1, 'o', color='g')

        elif len(est_ml) == 1:
            x0, y0 = est_ml[0][0], est_ml[0][1]
            print('the estimation of maximun likelihoos is: ', x0, y0)
            plt.plot(x0, y0, 'o', color='g')

        x_tls, y_tls = est_tls[0][0], est_tls[0][1]
        plt.plot(x_tls, y_tls, 'o', color='y')
        print('the estimation of least square is: ', x_tls, y_tls)

        plt.plot(bn[0], bn[1], 'x', color='r')
        plt.plot(ary[:, 0], ary[:, 1], '^')

    elif b_shape == (3,1):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(bn[0], bn[1], bn[2],'x', color='r')
        plt.plot(ary[:, 0], ary[:, 1], ary[:, 2], '^')
        print('the unknown is locating at: ', bn[0][0], bn[1][0], bn[2][0])
        
        if n == 4:

            if len(est_ml) == 2:

                x0, y0, z0 = est_ml[0][0], est_ml[0][1], est_ml[0][2]
                x1, y1, z1 = est_ml[1][0], est_ml[1][1], est_ml[1][2]
                print('the first estimation of maximun likelihoos is: ', x0, y0, z0)
                print('the alternative estimation of maximun likelihoos is: ', x1, y1, z1)
                ax.scatter(x0, y0, z0, color='g')
                ax.scatter(x1, y1, z1, color='g')

            elif len(est_ml) == 1:
                x0, y0, z0 = est_ml[0][0], est_ml[0][1], est_ml[0][2]
                print('the estimation of maximun likelihoos is: ', x0, y0, z0)
                ax.scatter(x0, y0, z0, color='g')

            x_ml_5, y_ml_5, z_ml_5 = est_ml_5[0][0], est_ml_5[0][1], est_ml_5[0][2]
            print('the estimation of high dimention maximun likelihoos is: ', x_ml_5, y_ml_5, z_ml_5)
            ax.scatter(x_ml_5, y_ml_5, z_ml_5, color='r')

        elif n == 5:

            if len(est_ml) == 2:

                x0, y0, z0 = est_ml[0][0], est_ml[0][1], est_ml[0][2]
                x1, y1, z1 = est_ml[1][0], est_ml[1][1], est_ml[1][2]
                print('the first estimation of maximun likelihoos is: ', x0, y0, z0)
                print('the alternative estimation of maximun likelihoos is: ', x1, y1, z1)
                ax.scatter(x0, y0, z0, color='g')
                ax.scatter(x1, y1, z1, color='g')

            elif len(est_ml) == 1:
                x0, y0, z0 = est_ml[0][0], est_ml[0][1], est_ml[0][2]
                print('the estimation of maximun likelihoos is: ', x0, y0, z0)
                ax.scatter(x0, y0, z0, color='g')

            x_ml_5, y_ml_5, z_ml_5 = est_ml_5[0][0], est_ml_5[0][1], est_ml_5[0][2]
            print('the estimation of high dimention maximun likelihoos is: ', x_ml_5, y_ml_5, z_ml_5)
            ax.scatter(x_ml_5, y_ml_5, z_ml_5, color='black')

    plt.show()
