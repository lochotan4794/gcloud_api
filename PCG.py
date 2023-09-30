import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import scipy.io

class PARAM:
     def __init__(self):
        self._Tmax = 0
        self._TOL = 0
        self._verbose = True
        self._pureFW = 1 # to not use away steps...
       
     # function to get value of _age
     def get_Tmax(self):
         return self._Tmax
       
     # function to set value of _age
     def set_Tmax(self, a):
         self._Tmax = a
  
     # function to delete _age attribute
     def del_Tmax(self):
         del self._Tmax

     def get_TOL(self):
         return self._TOL
       
     # function to set value of _age
     def set_TOL(self, a):
         self._TOL = a
  
     # function to delete _age attribute
     def del_TOL(self):
         del self._TOL

     def get_verbose(self):
         return self._verbose
       
     # function to set value of _age
     def set_verbose(self, a):
         self._verbose = a
  
     # function to delete _age attribute
     def del_verbose(self):
         del self._verbose

     def get_pureFW(self):
         return self._pureFW
       
     # function to set value of _age
     def set_pureFW(self, a):
         self._pureFW = a
  
     # function to delete _age attribute
     def del_pureFW(self):
         del self._pureFW
     
     Tmax = property(get_Tmax, set_Tmax, del_Tmax) 
     TOL = property(get_TOL, set_TOL, del_TOL) 
     verbose = property(get_verbose, set_verbose, del_verbose) 
     pureFW = property(get_verbose, set_pureFW, del_pureFW) 
       
  
opts = PARAM()
  
def find(s):
    # print(s)
    index = []
    ls = s.tolist()
    for l in ls:
        index.append(l[1])
    return index

def away_step(grad, S, I_active):
    s = np.dot(grad.T, S[:,I_active])
    idx = np.argmax(s)
    id = I_active[idx]
    return id

def cost_fun(y, A, b):
    by = np.dot(b.T, y)
    ay = np.dot(A, y)
    cost =.5 * np.dot(y.T, ay) + by
    return cost


class KeyStore:

    def __init__(self):
        self.store = []

    def isKey(self, key):
        for item in self.store:
            k, value = item
            if np.array_equal(k, key):
                return True
        return False

    def set_key(self, key, value):
        self.store.append((key, value))

    def get_value(self, key):
        for item in self.store:
            k, value = item
            if np.array_equal(k, key):
                return value
        assert("not found key!")

    def delete(self, key):
        idx = -1
        for i in range(len(self.store)):
            k, value = self.store[i]
            if np.array_equal(k, key):
                idx = i
        if idx != -1:
            self.store.pop(idx)

def PFW(x_0, S_0, alpha_0, A, b, fun_optim, cost_fun, ids, opts):
    it   = 0
    minf = 0
    minx = []

    res = {}

    x_t         = x_0
    S_t         = S_0
    alpha_t     = alpha_0

    eps = 1e-16

    mapping = KeyStore()

    max_index = S_t.shape[1] - 1 # keep track of size of S_t
    for index in range(0, max_index):
        mapping.set_key(np.expand_dims(S_t[:,index], 1), index)
    I_active = find(np.argwhere(alpha_t > 0))

    fvalues = {}
    gap_values = {}
    number_drop = 0 # counting drop steps (max stepsize for away step)

    print('running pairwise FW, for at most {} iterations\n'.format(opts.Tmax))

    # optimization: 

    while it <= opts.Tmax:
        it = it + 1

        # gradient:
        grad        = np.dot(A, x_t) + b

        # cost function:
        f_t = cost_fun(x_t, A, b)

        # towards direction search:
        s_FW     = fun_optim( grad, ids ) # the linear minimization oracle
        d_FW     = s_FW - x_t

        # duality gap:
        gap = np.dot(- d_FW.T, grad)

        fvalues[it-1] = f_t
        gap_values[it-1] = gap
        
        if opts.verbose:
            print('it = {} -  f = {} - gap={}\n'.format(it, f_t, gap))

        if gap < opts.TOL:
            print('end of PFW: reach small duality gap (gap={})\n'.format(gap))
            break
        
        # away direction search:
        if S_t.shape[1] > 0:
            id_A   = away_step(grad, S_t, I_active)
            v_A    = np.expand_dims(S_t[:, id_A], 1)
            #d_A    = x_t - v_A
            alpha_max = alpha_t[0, id_A]
        else:
            print('error: empty support set at step (it={})\n'.format(it))

            # construct pair direction (between towards and away):

        d = s_FW - v_A #was for away: d_A
        max_step = alpha_max # was for away: alpha_max / (1 - alpha_max)
        
        # line search: (same as away, but different d)

        step = -  np.dot(grad.T , d) / np.dot(np.dot( d.T, A), d)[0, 0]

        step = max(0, min(step, max_step ))

        if step < -eps:
            # not a descent direction???
            print('ERROR -- not descent direction???')

        # doing steps and updating active set:
        
        # away part of the step step:
        # alpha_t = (1+step)*alpha_t % note that inactive should stay at 0;
        if abs(step - max_step) < 10*eps:
            # drop step:
            number_drop = number_drop+1
            alpha_t[0, id_A] = 0
            I_active.remove(id_A)# remove from active set
            #TODO: could possibly also remove it from S_t
        else:
            alpha_t[0,id_A] = alpha_t[0, id_A] - step
            
        # towards part of the step:
        # alpha_t = (1-step)*alpha_t
            
        # is this s_FW a new vertex?
        h = s_FW
        if not mapping.isKey(h):
            # we need to add vertex in S_t:
            max_index = max_index + 1
            mapping.set_key(h, max_index)
            S_t = np.hstack((S_t, s_FW))
            id_FW = max_index
            alpha_t = np.append(alpha_t, np.ones(shape=(1,1)) * step, axis=1) # this increase size of alpha_t btw
            I_active.append(id_FW)
        else:
            id_FW = mapping.get_value(h)
            if alpha_t[0, id_FW] < eps:
                # we already had atom in 'correction poytope', but it was not
                # active, so now track it as active:
                I_active.append(id_FW)
            alpha_t[0, id_FW] = alpha_t[0, id_FW] + step
            
        # exceptional case: stepsize of 1, this collapses the active set!
        if step > 1-eps:
            I_active = [id_FW]
        
        x_t = x_t + step * d
    
    res["primal"] = fvalues
    res["gap"] = gap_values
    res["number_drop"] = number_drop
    res["S_t"] = S_t
    res["alpha_t"] = alpha_t
    res["x_t"] = x_t

    return x_t,f_t,res


def solver_image(x):
    idx  = np.argmin(x)
    y    = np.zeros(shape=x.shape)
    y[idx] = 1
    return y

def solver_images(x, ids):
    y   = np.zeros(shape=x.shape)
    for i in range(0 , ids.shape[0]):
        y[ids[i,0] : ids[i, 1] + 1] = solver_image(x[ids[i,0] : ids[i, 1] + 1])
    return y


def get_first_indice(arr):
    ind = []
    unique_set = set(arr.tolist())
    for i in unique_set:
        idx = np.argwhere(arr == i)
        ind.append(min(idx)[0])
    return ind


def get_last_indice(arr):
    ind = []
    unique_set = set(arr.tolist())
    for i in unique_set:
        idx = np.argwhere(arr == i)
        ind.append(max(idx)[0])
    return ind

def init_images(var_index):
    
    N = var_index.shape[0]
    a, b, c = np.unique(var_index[:, 0:2], return_index=True, return_inverse=True, axis=0)
    ib = np.expand_dims(np.array(get_first_indice(c)), 1)
    ie = np.expand_dims(np.array(get_last_indice(c)), 1)
    ids = np.hstack((ib, ie))

    fun_optim = solver_images

    n_imgs = ids.shape[0]
    boxes_per_img = int(N / n_imgs)
    x_0     = np.ones((N,1)) / boxes_per_img
    S_0     = np.zeros((N, boxes_per_img))
    alpha_0 = np.ones((1, boxes_per_img)) / boxes_per_img
    for i in  range(1, boxes_per_img):
        index = ids[:,0] + (i-1) * (N + 1)
        for j in index.tolist():
            row = int(j / N)
            col = j - N * row 
            S_0[col, row] = 1
        # print(S_0)
    return x_0, S_0, alpha_0, ids