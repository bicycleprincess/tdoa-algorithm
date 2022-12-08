import numpy as np

from convertions import list2string, dist
import settings
from constant import TEMPERTURE, COACH, NAMES, INFO, P4, P5


V = TEMPERTURE * 0.606 + 331.3


class Usher(object):

    def __init__(self, dictionary):

        self.length = len(dictionary)

    def check_Validity(self, s, doa):

        alist = []
        strings = s.split(' ')

        for string in strings:
            sub = []
            interrupt = False
            counter = 0
            for i in string:
                if i.isdigit() or i == '.' or i.startswith('-'):
                    counter += 1
                    sub.append(i)
                else:
                    interrupt = True
                    break

            for i in ('.', '-'):
                if sub.count(i) > 1:
                    first = sub.index(i)
                    second = sub[first+1:].index(i)
                    sub = sub[:(first+second+1)] 

            if not interrupt:
                try:
                    alist.append(float(list2string(sub)))
                except ValueError:
                    pass
            else:
                if counter != 0:
                    try:
                        rest = string[:counter]
                        alist.append(float(rest))
                    except ValueError:
                        try:
                            alist.append(float(list2string(sub)))
                        except ValueError:
                            pass

        alist.append(float(doa))
        settings.LOC.append(alist)
        return settings.LOC

    def construct_Arrary(self, l):

        DIM = 0

        for i in l:

            if len(i) == 1:
                i.extend(COACH)
            if len(i) == 4:
                DIM += 1

        if len(l) == 5:
            if DIM == 5:
                return np.asarray(l)
            elif DIM == 4:
                l = [x for x in l if len(x) == 4]
            elif DIM == 3:
                for i in range(len(l)):
                    if len(l[i]) == 3:
                        l[i].insert(2, 0.573)
            else:
                for i in range(len(l)):
                    if len(l[i]) == 4:
                        l[i].remove(l[i][3])
        elif len(l) == 4:
            if DIM < 4:
                for i in range(len(l)):
                    if len(l[i]) == 4:
                        l[i].remove(l[i][3])
            return np.asarray(l)
        elif len(l) == 3:
            if DIM <= 3:
                for i in range(len(l)):
                    if len(l[i]) == 4:
                        l[i].remove(l[i][3])
        return np.asarray(l)

    def non_collineation_check(self, ary):

        try:
            n, dim = np.shape(ary)[0], np.shape(ary)[1] - 1

            if dim == 2:
                p1 = ary[0, :]
                p2 = ary[1, :]
                p3 = ary[2, :]

                a = dist(p1, p2, 0)
                b = dist(p1, p3, 0)
                c = dist(p2, p3, 0)

                s = (a + b + c) / 2

                if (s*(s-a)*(s-b)*(s-c)) != 0.:
                    return np.chararray.tolist(ary)
                else:
                    return None
            return np.chararray.tolist(ary)
        except IndexError:
            pass


class Calculator(object):

    def __init__(self, ary):

        """Calculating for localisation

            np.shape(ary)

            3D: (5, 4), (4, 4)
            2D: (5, 3), (4, 3), (3, 4), (3, 3)

        """
        self.shape = np.shape(ary)
        self.ary = ary
        self.v = V
        self.n, self.dim = self.shape[0], self.shape[1] - 1
        self.group = ary[ary[:, self.dim].argsort()]

    def tls(self):

        """Total Least Square 2D and 3D.

        Parameters
        ----------
        rn : ndarray
             shape = (dim, n)
             rn is Reference nodes.
        rnr : ndarray
              shape = (dim, 1)
              rnr is Reference node of References,
              default as the first row of array
        toa : ndarray
              toa is Time of Arrival

        Return value
        ------------
        EST : list type (x, y, r) or (x, y, z, r)
              r is the distance from the unknown node to the rnr
        """
        rn = (self.group[1:, :self.dim]).T
        rnr = (self.group[0, :self.dim]).T
        toa = self.group[:, self.dim]
        tdoa = self.group[1:, self.dim] - self.group[0, self.dim]
        rdoa = tdoa * self.v
        rdoa_squared = rdoa * rdoa
        k1 = (np.sum(rn * rn, axis=0) - np.sum(rnr * rnr, axis=0))
        K = k1 - rdoa_squared
        A = np.hstack((rn.T-rnr.T, rdoa.reshape(np.shape(tdoa)[0], 1)))
        EST = 5e-1 * np.dot(np.linalg.pinv(A), K)
        return np.chararray.tolist(EST)

    def ml(self):

        """Maximum Likelihood 2D and 3D.

        Parameters
        ----------
        D : np.mat
            shape = (dim+1, n)
            P is the matrix about known positions and the toa infomation
        P : np.mat
            shape = (dim, n)
            P is the matrix about known positions merely
        toa : ndarray
              toa is Time of Arrival

        Return value
        ------------
        EST : list type (x, y) or (x, y, z)

        """
        D = np.mat(self.group)
        tdoa = (D[:, self.dim] - D[:, self.dim][0])[1:]
        rdoa = np.mat(tdoa * self.v)
        A = -(D[1:, :self.dim] - D[0, :self.dim]).I
        r_squared = rdoa.A * rdoa.A
        M = D[:, :self.dim]
        K = np.sum((np.multiply(M, M)), axis=1)
        B = (r_squared - K[1:] + K[0]) / 2
        E = A * rdoa
        F = A * B

        a = 1 - (E.T * E)
        b = 2 * (M[0] * E - F.T * E)
        c = 2 * (M[0] * F) - F.T * F - K[0]
        discr = b ** 2 - 4 * a * c

        if discr >= 0:
            root = np.sqrt(discr)
            for i in (root, -root):
                R0 = (i - b) / (2 * a)
                if R0 >= 0:
                    EST = E * R0 + F
                    return np.chararray.tolist((EST.A.squeeze()))

    def ml_5(self):

        """Maximum Likelihood 3D.
        """
        tdoa = self.group[:, self.dim] - self.group[:, self.dim][0]
        rdoa = tdoa * self.v
        D = np.hstack((self.group[:, :self.dim], rdoa.reshape((self.n, 1))))
        M = D[:, :self.dim]
        num = len(M)
        G = D[1:] - D[0]
        K = np.sum((np.multiply(M, M)), axis=1)
        r_squared = rdoa[1:] * rdoa[1:]
        h = 5e-1 * (r_squared - K[1:] + K[0])
        Q = np.mat((5e-1 * np.eye(num-1)) + 5e-1)
        first = np.dot(np.linalg.pinv(-G), h)
        R0 = first[-1]
        Y = np.mat(np.diag((rdoa[1:] + R0) * self.v ** 2))
        try:
            second = ((-G.T * Y.I * -G).I * (-G.T * Y.I) * np.mat(h).T).A.squeeze()
            return np.chararray.tolist(second)
        except np.linalg.LinAlgError:
            pass


def afsk_estimator(robot_nanme):

    """Handling the demodulated AFSK information

    """
    usher = Usher(settings.POSITIONS)
    for robot in settings.POSITIONS.keys():
        if robot != robot_nanme:
            try:
                info = settings.POSITIONS[robot]['position']
                doa = settings.POSITIONS[robot]['doa']
                ary = usher.check_Validity(info, doa)
            except KeyError:
                pass
            except TypeError:
                pass
    try:
        if len(ary) >= 3:
            ary = usher.construct_Arrary(ary)
            new_array = usher.non_collineation_check(ary)
            return new_array
    except UnboundLocalError:
        pass


def cal_toa(ta1, ta3, tb1, tb3):

    return ((abs(V * 0.5 * ((tb3-tb1)-(ta3-ta1))/44100.)) * 1e-2) / V


def cal_for_myself(number, etoa, extra=None):

    for i, info in INFO.iteritems():
        if number == info[0]:
            my_name = i

    if my_name in etoa.keys():
        for name in etoa.keys():

            if name == my_name:
                my_etoa = etoa[name]['etoa']

                if len(my_etoa) > 6:
                    if (int(my_etoa[-1]) - int(my_etoa[0])) / 44100. > 60.:
                        my_etoa.pop()
                    return (my_name, number, my_etoa), etoa

                elif len(my_etoa) >= 4 and len(my_etoa) >= number:
                    if int(my_etoa[0]) / 44100. > 6:
                        my_etoa.insert(0, '0')
                    return (my_name, number, my_etoa), etoa
                else:
                    return None, None
    else:
        if extra is not None and len(extra) >= 4:

            if len(extra) > 6:

                if (int(extra[-1]) - int(extra[0])) / 44100. > 60.:
                    extra.pop()

            return (my_name, number, extra), etoa
        else:
            return None, None


def estimator(myself, etoa):

    if myself is not None:
        my_name, my_id, my_etoa = myself[0], myself[1], myself[2]
        positions = []
        for key in etoa.keys():

            if key != my_name:
                l = []
                try:
                    if int(etoa[key]['etoa'][-1]) > 60 * 44100 * 2:
                        etoa[key]['etoa'].pop()

                    if my_etoa[NAMES.index(key)] != '0':
                        #print 'ok'
                        ta1 = my_etoa[my_id - 1]
                        ta3 = my_etoa[NAMES.index(key)]
                        tb1 = etoa[key]['etoa'][my_id - 1]
                        tb3 = etoa[key]['etoa'][NAMES.index(key)]
                        doa = cal_toa(int(ta1), int(ta3), int(tb1), int(tb3))

                        if doa < 10.8 / V:
                            l.append(etoa[key]['position'][0])
                            l.append(etoa[key]['position'][1])
                            l.append(doa)
                            positions.append(l)
                            settings.POSITIONS[key]['doa'] = doa

                except IndexError:
                    pass
        return positions
    else:
        return None


def check(x, y):

    if abs(x) < 5 and abs(y) < 4:
        return x, y
    else:
        return None


def cal(alist):

    if alist is not None and len(alist) >= 3:
        shape = len(alist)
        ary = np.asarray(alist)
        #print 'ary:', ary
        computer = Calculator(ary)
        rst = []

        x, y = computer.tls()[0], computer.tls()[1]
        if check(x, y):
            rst.append((x, y))
        else:
            if shape == 4:
                for i in P4:
                    computer = Calculator(np.asarray([alist[i[0]], alist[i[1]], alist[i[2]]]))
                    x, y = computer.tls()[0], computer.tls()[1]
                    if check(x, y):
                        rst.append((x, y))
            elif shape == 5:
                for i in P5:
                    if len(i) == 4:
                        computer = Calculator(np.asarray([alist[i[0]], alist[i[1]], alist[i[2]], alist[i[3]]]))
                        x, y = computer.tls()[0], computer.tls()[1]
                        if check(x, y):
                            rst.append((x, y))
                        else:
                            for i in P5:
                                if len(i) == 3:
                                    computer = Calculator(np.asarray([alist[i[0]], alist[i[1]], alist[i[2]]]))
                                    x, y = computer.tls()[0], computer.tls()[1]
                                    if check(x, y):
                                        rst.append((x, y))

        x, y = computer.ml_5()[0], computer.ml_5()[1]
        if check(x, y):
            rst.append((x, y))
        else:
            if shape >= 4:
                for i in P4:
                    computer = Calculator(np.asarray([alist[i[0]], alist[i[1]], alist[i[2]]]))
                    x, y = computer.ml_5()[0], computer.ml_5()[1]
                    if check(x, y):
                        rst.append((x, y))
            elif shape == 5:
                for i in P5:
                    if len(i) == 4:
                        computer = Calculator(np.asarray([alist[i[0]], alist[i[1]], alist[i[2]], alist[i[3]]]))
                        x, y = computer.ml_5()[0], computer.ml_5()[1]
                        if check(x, y):
                            rst.append((x, y))
                        else:
                            for i in P5:
                                if len(i) == 3:
                                    computer = Calculator(np.asarray([alist[i[0]], alist[i[1]], alist[i[2]]]))
                                    x, y = computer.ml_5()[0], computer.ml_5()[1]
                                    if check(x, y):
                                        rst.append((x, y))
        return rst
