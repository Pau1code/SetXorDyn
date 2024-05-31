import os
import copy
import math
import mmh3
import pickle
import random
import numpy as np

from utils import *
from math import exp, pow, log

import os
import pickle

from utils import *
from math import exp, pow, log
# from learned_sx_estimator.network import *


class SetXor:
    def __init__(self, dict_dataset, m, w, epsilon, output, random_response, seed, lamda_ratio=None, delete_dataset=None):
        self.dict_dataset = dict_dataset
        self.m_size = m
        self.w_size = w
        self.lamb = 1 / m if lamda_ratio==None else lamda_ratio / m
        self.epsilon = epsilon
        self.output = output
        self.seed = seed
        self.random_response = 1
        self.set_xor_seed = [0] * m
        
        if delete_dataset == None:
            self.delete_dataset = {}
            for user in self.dict_dataset:
                self.delete_dataset[user] = {}
                self.delete_dataset[user]['elements']=[]
        else:
            self.delete_dataset = delete_dataset
        self.repeatTimes = 1 if delete_dataset==None else 10

        # self.alpha = 0.1
        self.alpha = 1 / (1+math.exp(epsilon))


    def init_set_xor_seed(self, m):
        for i in range(m):
            self.set_xor_seed[i] = random.randint(0, 2 ** 32)
            
    def __arriveElement(self, item, row, repeattimes, setxor_sketch):
        poisoon_var = 1
        # item_trans = mmh3.hash(str(item), signed=False, seed=self.seed)
        # uni = item_trans / pow(2, 32)
        # poisoon_var = self.poisson(uni)
        if poisoon_var == 0: return 
        
        temp_pair = [0] * 2
        for p in range(repeattimes):
            for k in range(1, poisoon_var+1):
                temp_pair[0] = item
                temp_pair[1] = k
                
                temp_x_pair = [temp_pair[0], temp_pair[1], row]
                # temp_x_pair = [temp_pair[0], temp_pair[1], random.randint(1,100000)]
                
                # i ← H(e,k)
                temp_i = mmh3.hash(str(temp_pair), signed=False, seed=self.seed + 1)
                temp_i = bin(temp_i)[2:]
                index_i = int(temp_i, 2) % self.m_size
                
                index_j = self.compute_index_value(temp_pair, index_i)
                
                # temp_x = mmh3.hash(str(temp_x_pair), signed=False, seed=self.seed + 2)
                # temp_x = bin(temp_x)[2:]
                # x = int(temp_x, 2) % 2
                
                setxor_sketch[index_i][index_j] ^= 1
            row += 1

    def build_sketch(self):
        self.dict_setxor_sketch = dict()
        self.init_set_xor_seed(self.m_size)
        
        
        for user in self.dict_dataset:
            setxor_sketch = [[0] * self.w_size for _ in range(self.m_size)]
            
            # arrive
            user_dict = self.dict_dataset[user]
            # print(len(user_dict['elements']))
            for i in range(len(user_dict['elements'])):
                item = user_dict['elements'][i]
                row = copy.deepcopy(user_dict['index'][i])
                repeattimes = user_dict['repeattimes'][i]
                self.__arriveElement(item, row, repeattimes, setxor_sketch)
            
            # delete
            user_dict = self.delete_dataset[user]
            # print(len(user_dict['elements']))
            for i in range(len(user_dict['elements'])):
                item = user_dict['elements'][i]
                row = copy.deepcopy(user_dict['index'][i])
                repeattimes = user_dict['repeattimes'][i]
                self.__arriveElement(item, row, repeattimes, setxor_sketch)

            # pertube
            if self.random_response:
                setxor_sketch = self.perturbsketch(setxor_sketch)

            self.dict_setxor_sketch[user] = setxor_sketch

    # lamb取值和后面对应，而且这里泊松有没有必要？
    def poisson(self, u):
        x = 0
        p = exp(-(self.m_size * self.lamb))
        s = p
        for i in range(100):
            if u <= s:
                break
            x += 1
            p = p * (self.m_size * self.lamb) / float(x)
            s += p
        return x

    def compute_index_value(self, temp_pair, index_i):
        binary_item = mmh3.hash(str(temp_pair), signed=False, seed=self.set_xor_seed[index_i])
        binary_item = bin(binary_item)[2:]

        index_j = 0
        # trailing zeros
        rvs_binary_item = binary_item[::-1]
        for bit in rvs_binary_item:
            if bit == '0':
                index_j += 1
            else:
                break
        return index_j

    def perturbsketch(self, setxor_sketch,):
        for i in range(self.m_size):
            for j in range(self.w_size):
                temp = random.random()
                if temp <= 1 - self.alpha:
                    pertur_y = 1
                else:
                    pertur_y = 0

                setxor_sketch[i][j] ^= pertur_y

        return setxor_sketch
    

    def phi_func(self, n, w, alpha, lamb):
        z = w / 2
        for i in range(w - 1):
            z -= pow((1.0 - 2.0 * alpha), 2) * exp(-2.0 * n * lamb / pow(2.0, (i + 1))) / 2

        z -= pow((1.0 - 2.0 * alpha), 2) * exp(-2.0 * n * lamb / pow(2.0, (w - 1))) / 2

        return z

    def phi_func_derived(self, n, w, alpha, lamb):
        z = 0.0
        for i in range(w - 1):
            z += lamb * pow((1.0 - 2.0 * alpha), 2) * exp(
                -2.0 * n * lamb / pow(2.0, i + 1)) / pow(2.0, i + 1)

        z += lamb * pow((1.0 - 2.0 * alpha), 2) * exp(
            -2.0 * n * lamb / pow(2.0, w - 1)) / pow(2.0, w - 1)

        return z


    # TODO when n<=3m without combine LPC, refer to
    # Half-Xor 3.3.4 Finial Combined NDV Estimation
    def phi_func_estimator(self, merge_sketch, m, w, alpha, lamb):
        count = 0
        n = 100
        error = 1e-5
        for i in range(m):
            for j in range(w):
                count += merge_sketch[i][j]

        v = count / m

        while (self.phi_func(n, w, alpha, lamb) - v) * (self.phi_func(n+1, w, alpha, lamb) - v) > error:
            n = n - (self.phi_func(n, w, alpha, lamb) - v) / self.phi_func_derived(n, w, alpha, lamb)

        return n

    # def binaryfind(self, left, right):
    #     error = 1e-6
    #     middle = (left + right) / 2
    #     if self.phi_func(middle, self.w_size, self.alpha, self.lamb) == 0:
    #         return middle
    #     while abs(self.phi_func(middle, self.w_size, self.alpha, self.lamb)) > error:
    #         middle = (left + right) / 2
    #         if self.phi_func(left, self.w_size, self.alpha, self.lamb) * self.phi_func(middle,  self.w_size, self.alpha, self.lamb) <= 0:
    #             right = middle
    #         else:
    #             left = middle
    #     return middle

    def IVW_estimate(self, merge_sketch, m, w, alpha, lamb, esti):
        n = [0] * 32
        var = [0] * 32
        n_f = 0
        denomi = 0
        for j in range(w):
            z = 0
            for i in range(m):
                if merge_sketch[i][j] == 1:
                    z += 1
            if j < (w - 1):
                p = 1.0 / pow(2.0, j + 1)
            else:
                p = 1.0 / pow(2.0, j)
            if z < int(0.5 * m):
                n[j] = -log((1.0 - 2.0 * z / m) / pow((1.0 - 2.0 * alpha), 2)) / (2.0 * lamb * p)
                '''ZeroDivisionError: float division by zero'''
                deno = 4.0 * m * pow((lamb * p), 2) * pow((1.0 - 2.0 * alpha), 4) * exp(-4.0 * esti * lamb * p)
                if deno == 0: var[j] = float('inf')
                else: var[j] = (1.0 - pow((1.0 - 2.0 * alpha), 4) * exp(-4.0 * esti * lamb * p)) / deno
                denomi += 1.0 / var[j]
            else:
                n[j] = -1.0
                var[j] = -1.0

        for j in range(w):
            if n[j] >= 0:
                n_f += (n[j] / var[j]) / denomi

        return n_f

    def add_lap_noise(self, data):
        lap_noise = np.random.laplace(0, 1, len(data))
        return lap_noise + data
    
    def get_merge(self, M, d):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = np.array([[0] * self.w_size for _ in range(self.m_size)])
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]
        lst_result = [sum(setxor_sketch_merge[:][j]) / self.m_size for j in range(self.w_size)]
        lst_result.append(M)
        lst_result.append(d)
        return lst_result

    def estimate_intersection(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = [[0] * self.w_size for _ in range(self.m_size)]
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]

            actual_union, actual_intersection = compute_difference(lst_A, lst_B)

            estimated_difference = self.phi_func_estimator(setxor_sketch_merge, self.m_size,
                                                           self.w_size, self.alpha, self.lamb)

            # print(estimated_difference_IVW)
            A_hat = len(self.dict_dataset['A']['elements']) + np.random.laplace(0, self.epsilon)
            B_hat = len(self.dict_dataset['B']['elements']) + np.random.laplace(0, self.epsilon)

            estimated_intersection = 0.5 * (A_hat + B_hat - estimated_difference)
            # print("actual_intersection:{} | estimated_intersection:{}".format(actual_intersection, estimated_intersection))
            estimated_union = 0.5 * (A_hat + B_hat + estimated_difference)
            # print(estimated_union)

            # lst_result.append([actual_intersection, estimated_intersection])

        # foutput = open(os.path.join(self.output, 'setxor.out'), 'wb')
        # pickle.dump(lst_result, foutput)
        # foutput.close()

        return [estimated_intersection]
    
    def estimate_union(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = [[0] * self.w_size for _ in range(self.m_size)]
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]

            actual_union, actual_intersection = compute_difference(lst_A, lst_B)

            estimated_difference = self.phi_func_estimator(setxor_sketch_merge, self.m_size,
                                                           self.w_size, self.alpha, self.lamb)

            # print(estimated_difference_IVW)
            A_hat = len(self.dict_dataset['A']['elements']) + np.random.laplace(0, self.epsilon)
            B_hat = len(self.dict_dataset['B']['elements']) + np.random.laplace(0, self.epsilon)

            estimated_intersection = 0.5 * (A_hat + B_hat - estimated_difference)
            # print("actual_intersection:{} | estimated_intersection:{}".format(actual_intersection, estimated_intersection))
            estimated_union = 0.5 * (A_hat + B_hat + estimated_difference)
            # print(estimated_union)

            # lst_result.append([actual_intersection, estimated_intersection])

        # foutput = open(os.path.join(self.output, 'setxor.out'), 'wb')
        # pickle.dump(lst_result, foutput)
        # foutput.close()

        return [estimated_union]

    def estimated_difference_IVW(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = [[0] * self.w_size for _ in range(self.m_size)]
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]

            actual_union, actual_intersection = compute_difference(lst_A, lst_B)

            estimated_difference = self.phi_func_estimator(setxor_sketch_merge, self.m_size,
                                                           self.w_size, self.alpha, self.lamb)
            # print(estimated_difference)
            estimated_diff_IVW = self.IVW_estimate(setxor_sketch_merge, self.m_size, self.w_size, self.alpha,
                                                         self.lamb, estimated_difference)
        return [estimated_diff_IVW]

    def estimate_intersection_IVW(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = [[0] * self.w_size for _ in range(self.m_size)]
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]

            actual_union, actual_intersection = compute_difference(lst_A, lst_B)

            estimated_difference = self.phi_func_estimator(setxor_sketch_merge, self.m_size,
                                                           self.w_size, self.alpha, self.lamb)
            # print(estimated_difference)
            estimated_difference_IVW = self.IVW_estimate(setxor_sketch_merge, self.m_size, self.w_size, self.alpha,
                                                         self.lamb, estimated_difference)

            # print(estimated_difference_IVW)
            A_hat = len(self.dict_dataset['A']['elements']) + np.random.laplace(0, self.epsilon)
            B_hat = len(self.dict_dataset['B']['elements']) + np.random.laplace(0, self.epsilon)

            estimated_intersection = 0.5 * (A_hat + B_hat - estimated_difference)
            estimate_intersection_IVW = 0.5 * (A_hat + B_hat - estimated_difference_IVW)
            # print("actual_intersection:{} | estimated_intersection:{}".format(actual_intersection, estimated_intersection))
            estimated_union = 0.5 * (A_hat + B_hat + estimated_difference)
            # print(estimated_union)

            # lst_result.append([actual_intersection, estimated_intersection])

        # foutput = open(os.path.join(self.output, 'setxor.out'), 'wb')
        # pickle.dump(lst_result, foutput)
        # foutput.close()

        return [estimate_intersection_IVW]
    
    def estimate_union_IVW(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = [[0] * self.w_size for _ in range(self.m_size)]
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]

            actual_union, actual_intersection = compute_difference(lst_A, lst_B)

            estimated_difference = self.phi_func_estimator(setxor_sketch_merge, self.m_size,
                                                           self.w_size, self.alpha, self.lamb)
            # print(estimated_difference)
            estimated_difference_IVW = self.IVW_estimate(setxor_sketch_merge, self.m_size, self.w_size, self.alpha,
                                                         self.lamb, estimated_difference)

            # print(estimated_difference_IVW)
            A_hat = len(self.dict_dataset['A']['elements']) + np.random.laplace(0, self.epsilon)
            B_hat = len(self.dict_dataset['B']['elements']) + np.random.laplace(0, self.epsilon)

            estimated_intersection = 0.5 * (A_hat + B_hat - estimated_difference)
            estimate_intersection_IVW = 0.5 * (A_hat + B_hat - estimated_difference_IVW)
            # print("actual_intersection:{} | estimated_intersection:{}".format(actual_intersection, estimated_intersection))
            estimated_union = 0.5 * (A_hat + B_hat + estimated_difference)
            # print(estimated_union)

            # lst_result.append([actual_intersection, estimated_intersection])

        # foutput = open(os.path.join(self.output, 'setxor.out'), 'wb')
        # pickle.dump(lst_result, foutput)
        # foutput.close()

        return [estimated_union]
    
    def estimate_difference_MLP(self, M, path='.', verold=False):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = np.array([[0] * self.w_size for _ in range(self.m_size)])
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]

            lst_result = [sum(setxor_sketch_merge[:, j]) / self.m_size for j in range(self.w_size)]
            lst_result.append(M)
            if verold:model = torch.load(f'{path}/learned_sx_estimator/networkRes/ver1/net_{int(M)}.pt')
            else:model = torch.load(f'{path}/learned_sx_estimator/networkRes/net_{int(M)}.pt')
            diff = [model(torch.tensor(lst_result[0:32], dtype = torch.float32)).item()]

            return diff
    
    def estimate_intersection_MLP(self, M, path='.', verold=False):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            setxor_sketch_A = self.dict_setxor_sketch[user_A]
            setxor_sketch_B = self.dict_setxor_sketch[user_B]

            setxor_sketch_merge = np.array([[0] * self.w_size for _ in range(self.m_size)])
            for i in range(self.m_size):
                for j in range(self.w_size):
                    setxor_sketch_merge[i][j] = setxor_sketch_A[i][j] ^ setxor_sketch_B[i][j]

            lst_result = [sum(setxor_sketch_merge[:, j]) / self.m_size for j in range(self.w_size)]
            lst_result.append(M)
            if verold:model = torch.load(f'{path}/learned_sx_estimator/networkRes/ver1/net_{int(M)}.pt')
            else:model = torch.load(f'{path}/learned_sx_estimator/networkRes/net_{int(M)}.pt')
            diff = model(torch.tensor(lst_result[0:32], dtype = torch.float32)).item()

            # print(estimated_difference_IVW)
            A_hat = len(lst_A) + np.random.laplace(0, self.epsilon, 1)
            B_hat = len(lst_B) + np.random.laplace(0, self.epsilon, 1)
            
            estimate_intersection_MLP = 0.5 * (A_hat + B_hat - diff)
            return estimate_intersection_MLP
