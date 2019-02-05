from multiprocessing import Pool
import numba as nb
import numpy as np
import os


class COA:
    def __init__(self, lower_upper_boundaries, n_packs=20, n_coyotes=5, workers=os.cpu_count()):
        if n_coyotes < 3:
            raise Exception("At least 3 coyotes per pack must be used")

        self.lower_upper_boundaries = lower_upper_boundaries
        self.n_packs = n_packs
        self.n_coyotes = n_coyotes
        self.workers = workers

        self.lower_boundaries = self.lower_upper_boundaries[0]
        self.upper_boundaries = self.lower_upper_boundaries[1]
        self.problem_dimensions = self.lower_upper_boundaries.shape[1]
        self.ps = 1 / self.problem_dimensions

    def fit(self, func, n_feval_max):
        #executor = Pool(self.workers)

        p_leave = 0.005*(self.n_coyotes ** 2)
        pop_total = self.n_packs * self.n_coyotes
        coyotes = np.tile(self.lower_boundaries, [pop_total, 1]) + np.random.rand(pop_total, self.problem_dimensions) * \
                  np.tile(self.upper_boundaries, [pop_total, 1]) - np.tile(self.lower_boundaries, [pop_total, 1])
        ages = np.zeros(pop_total)
        packs = np.random.permutation(pop_total).reshape(self.n_packs, self.n_coyotes)

        costs = np.array([func(coyotes[cost, :]) for cost in range(pop_total)])

        n_feval = pop_total
        global_best_cost = min(costs)
        ibest = costs.argmin()
        global_best_params = coyotes[ibest, :]

        year = 1
        while n_feval < n_feval_max:
            try:
                year += 1
                args_list = [[func, coyotes[packs[p, :], :], costs[packs[p, :]], ages[packs[p, :]], p] for p in range(self.n_packs)]
                #results = executor.imap_unordered(self._eval_pack, args_list)
                results = [self._eval_pack(arg[0], arg[1], arg[2], arg[3], arg[4]) for arg in args_list]

                for result in results:
                    coyotes_aux, costs_aux, ages_aux, n_feval_ret, pack_id = result
                    coyotes[packs[pack_id], :] = coyotes_aux
                    costs[packs[pack_id]] = costs_aux
                    ages[packs[pack_id]] = ages_aux
                    n_feval += n_feval_ret

                if self.n_packs > 1:
                    if np.random.rand() < p_leave:
                        rp = np.random.permutation(self.n_packs)[:2]
                        rc = np.array([np.random.randint(self.n_coyotes), np.random.randint(self.n_coyotes)])
                        aux = packs[rp[0], rc[0]]
                        packs[rp[0], rc[0]] = packs[rp[1], rc[1]]
                        packs[rp[1], rc[1]] = aux
                ages += 1
                global_best_cost = costs.min()
                global_best_params = coyotes[costs.argmin()]
            except KeyboardInterrupt:
                raise
                #executor.terminate()
                #executor.join()
        #executor.terminate()
        #executor.join()
        return global_best_cost, global_best_params

    #@nb.jit(nb.int32(nb.types.pyfunc_type))
    def _eval_pack(self, func, coyotes_aux, costs_aux, ages_aux, pack_id):

        ind = np.argsort(costs_aux)
        costs_aux = costs_aux[ind]
        coyotes_aux = coyotes_aux[ind, :]
        ages_aux = ages_aux[ind]
        c_alpha = coyotes_aux[0, :]

        n_feval = 30
        """tendency = np.median(coyotes_aux, 0)
        new_coyotes = np.zeros((self.n_coyotes, self.problem_dimensions))
        t = time.time()
        
        for c in range(self.n_coyotes):
            rc1 = c
            while rc1 == c:
                rc1 = np.random.randint(self.n_coyotes)
            rc2 = c
            while rc2 == c or rc2 == rc1:
                rc2 = np.random.randint(self.n_coyotes)

            new_coyotes[c, :] = coyotes_aux[c, :] + np.random.rand() * (c_alpha - coyotes_aux[rc1, :]) + \
                                np.random.rand() * (tendency - coyotes_aux[rc2, :])

            for abc in range(self.problem_dimensions):
                new_coyotes[c, abc] = max(
                    [min([new_coyotes[c, abc], self.upper_boundaries[abc]]), self.lower_boundaries[abc]])

            new_cost = func(new_coyotes[c, :])
            n_feval += 1

            if new_cost < costs_aux[c]:
                costs_aux[c] = new_cost
                coyotes_aux[c, :] = new_coyotes[c, :]

        parents = np.random.permutation(self.n_coyotes)[:2]
        prob1 = (1 - self.ps) / 2
        prob2 = prob1
        pdr = np.random.permutation(self.problem_dimensions)
        p1 = np.zeros(self.problem_dimensions)
        p2 = np.zeros(self.problem_dimensions)
        p1[pdr[0]] = 1
        p2[pdr[1]] = 1
        r = np.random.rand(self.problem_dimensions - 2)
        p1[pdr[2:]] = r < prob1
        p2[pdr[2:]] = r > 1 - prob2

        n = np.logical_not(np.logical_or(p1, p2))

        pup = p1 * coyotes_aux[parents[0], :] + \
              p2 * coyotes_aux[parents[1], :] + \
              n * (self.lower_boundaries + np.random.rand(1, self.problem_dimensions) * (self.upper_boundaries - self.lower_boundaries))
        pup_cost = func(pup[0])
        n_feval += 1

        worst = np.flatnonzero(costs_aux > pup_cost)

        if len(worst) > 0:
            older = np.argsort(ages_aux[worst])
            which = worst[older[::-1]]
            coyotes_aux[which[0], :] = pup
            costs_aux[which[0]] = pup_cost
            ages_aux[which[0]] = 0
        print(time.time()-t)"""
        return coyotes_aux, costs_aux, ages_aux, n_feval, pack_id


def fobj(x):
    return np.sum(x**2)

if __name__ == "__main__":

    import time
    from tqdm import tqdm
    d = 30
    lu = np.zeros((2, d))
    lu[0, :] = -1
    lu[1, :] = 1
    nfeval = 1000*d
    n_packs = 20
    n_coy = 5
    t = time.time()
    y = np.zeros((1, 100))
    coa = COA(lu,n_packs=n_packs,n_coyotes=n_coy)
    for i in tqdm(range(100)):
        mini, par = coa.fit(fobj, nfeval)
        y[0, i] = mini
        #print(par)
        #print(time.time()-t)
        t = time.time()
    print([np.min(y), np.mean(y), np.median(y), np.max(y), np.std(y)])


