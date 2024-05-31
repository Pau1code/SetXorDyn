import time
import random
import argparse

from utils import *
from loader import Dataloader

from setxor import SetXor
from setxor_Dyn import SetXor_Dyn
from baseline.cl import CL
from baseline.fm import FM
from baseline.ll import LL
from baseline.sfm import SFM
from baseline.hll import HyperLogLog


def get_args():
    parser = argparse.ArgumentParser(description="sketch method for estimating intersection cardinalities")
    parser.add_argument('--method', default='SetXorDyn', type=str, help='method name: SetXorDyn/SetXorDyn_IVW/SFM/HLL/FM/CL/LL')
    parser.add_argument('--dataset', default='synthetic', type=str, help='dataset path or synthetic')
    parser.add_argument('--intersection', default=100000000, type=int, help='set intersection cardinality')
    parser.add_argument('--difference', default=100000000, type=int, help='set difference cardinality')
    parser.add_argument('--ratio', default=0.5, type=float, help='skewness ratio used to control cardinalities of two sets')
    parser.add_argument('--exp_rounds', default=10, type=int, help='the number of experimental rounds')
    parser.add_argument('--output', default='result/', type=str, help='output directory')
    parser.add_argument('--epsilon', default=1, type=int, help='privacy budget')
    parser.add_argument('--counter', default=32, type=int, help='counter size')

    # FM sketch param
    parser.add_argument('--fm_Msize', default=4096, type=int, help='the number of FM sketch rows')
    parser.add_argument('--fm_Wsize', default=32, type=int, help='the number of FM sketch columns')
    
    # HLL sketch param
    parser.add_argument('--hll_Msize', default=4096, type=int, help='the number of HLL sketch rows')

    # SFM param
    parser.add_argument('--sfm_Msize', default=4096, type=int, help='m of SFM sketch')
    parser.add_argument('--sfm_Wsize', default=32, type=int, help='w of SFM sketch')
    parser.add_argument('--merge_method', default='deterministic', type=str, help='the merge method of SFM deterministic/random')

    # SetXor/SetXor_IVW/SetXor_Dyn/SetXor_IVW_Dyn param
    parser.add_argument('--setxor_Msize', default=4096, type=int, help='m of SetXorDyn/SetXorDyn_IVW sketch')
    parser.add_argument('--setxor_Wsize', default=32, type=int, help='w of SetXorDyn/SetXorDyn_IVW sketch')
    parser.add_argument('--random_response', action="store_true", help='whether use random response')
    
    # Cascading_Legions param
    parser.add_argument('--cl_Msize', default=4096, type=int, help='m of Cascading_Legions')
    parser.add_argument('--cl_l', default=32, type=int, help='l of Cascading_Legions')
    
    # Liquid_Legions param
    parser.add_argument('--ll_Msize', default=4096, type=int, help='m of Liquid_Legions')
    parser.add_argument('--ll_a', default=10, type=int, help='a of Liquid_Legions')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    exp_rounds = args.exp_rounds
    lst_all_results = list()
    all_time_list = list()

    for r in range(exp_rounds):
        seed = random.randint(1, 2**32-1)
        dataloader = Dataloader(args.dataset, args.intersection, args.difference, args.ratio, seed, maxrepeattimes=10)
        dict_dataset = dataloader.load_dataset()
        print('{}th exp_rounds dataset generation finished!'.format(r))
        start_time = time.time()
        if args.method == 'SetXorDyn':
            setxor = SetXor_Dyn(dict_dataset, args.setxor_Msize, args.setxor_Wsize, args.epsilon, args.output, args.random_response, seed)
            setxor.build_sketch()
            result = setxor.estimate_intersection()
            lst_all_results.append(result[0])
            
        if args.method == 'SetXorDyn_IVW':
            setxor = SetXor_Dyn(dict_dataset, args.setxor_Msize, args.setxor_Wsize, args.epsilon, args.output, args.random_response, seed)
            setxor.build_sketch()
            result = setxor.estimate_intersection_IVW()
            lst_all_results.append(result[0])

        if args.method == "SFM":
            sfm = SFM(dict_dataset, int(args.sfm_Msize/args.counter), args.sfm_Wsize, args.epsilon, args.merge_method, seed)
            sfm.build_fm_sketch()
            result = sfm.estimation_itersection_cardinality()
            lst_all_results.append(result[0])

        if args.method == "FM":
            fm = FM(dict_dataset, int(args.fm_Msize/args.counter), args.fm_Wsize, seed)
            fm.build_sketch()
            result = fm.estimate_intersection()
            lst_all_results.append(result[0])

        if args.method == "HLL":
            hll = HyperLogLog(dict_dataset, int(args.hll_Msize/args.counter), seed)
            hll.build_sketch()
            result = hll.estimate_intersection()
            lst_all_results.append(result[0])
            
        if args.method == "CL":
            cl = CL(dict_dataset, int(args.cl_Msize/args.counter), args.cl_l, args.epsilon, seed)
            cl.build_sketch()
            result = cl.estimation_intersection_cardinality()
            lst_all_results.append(result[0])
            
        if args.method == "LL":
            ll = LL(dict_dataset, args.ll_a, int(args.ll_Msize/args.counter), args.epsilon, seed)
            ll.build_sketch()
            result = ll.estimation_intersection_cardinality()
            lst_all_results.append(result[0])
        
        end_time = time.time()
        each_round_time = end_time - start_time
        all_time_list.append(each_round_time)

    print('time:', sum(all_time_list)/len(all_time_list))
    AARE = compute_aare(lst_all_results, args.intersection)
    print("The value of AARE: {}%".format(AARE * 100))