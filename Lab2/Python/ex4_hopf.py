""" Hopf oscillator """
import numpy as np

class HopfParameters(object):
    """ Hopf Parameters """

    def __init__(self, mu, omega):
        super(HopfParameters, self).__init__()
        self.mu = mu
        self.omega = omega
        return

    def __str__(self):
        return self.msg()

    def msg(self):
        """ Message """
        return "mu: {}, omega: {}".format(self.mu, self.omega)

    def check(self):
        """ Check parameters """
        assert self.mu >= 0, "Mu must be positive"
        assert self.omega >= 0, "Omega must be positive"
        return


class CoupledHopfParameters(HopfParameters):
    """ Coupled Hopf Parameters """

    def __init__(self, mu, omega, k):
        super(CoupledHopfParameters, self).__init__(mu, omega)
        self.k = k
        return

    def msg(self):
        """ Message """
        return "mu: {}, omega: {}, k: {}".format(self.mu, self.omega, self.k)

    def check(self):
        """ Check parameters """
        assert self.mu >= 0, "Mu must be positive"
        assert self.omega >= 0, "Omega must be positive"
        assert self.k >= 0, "K must be positive"
        return


def hopf_equation(x, _=None, params=HopfParameters(mu=1., omega=1.0)):
    """ Hopf oscillator equation """
    mu = params.mu
    omega = params.omega
    if np.size(x)<2:
        mu=mu
    r = np.sqrt(x[0]**2+x[1]**2)
    xdot = (mu-r**2)*x[0]-omega*x[1]
    ydot = (mu-r**2)*x[1]+omega*x[0]
    return [xdot, ydot]


def coupled_hopf_equation(x, _=None, params=None):
    """ Coupled Hopf oscillator equation """
    if params is None:
        params = CoupledHopfParameters(
            mu=[1., 1.],
            omega=[1.0, 1.2],
            k=[-0.5, -0.5]
        )
    mu = params.mu
    omega = params.omega
    k = params.k
    indep_params = HopfParameters(mu[0], omega[0])
    indep = hopf_equation(x[0:2], indep_params)
    
    dep_params = HopfParameters(mu[1], omega[1])
    dep = hopf_equation(x[2:], dep_params)
    dep = np.array(dep) - k[0]*x[3]
    
    return [indep[0], indep[1], dep[0], dep[1]]

