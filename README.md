# Private Sketches for Securely Estimating Set Intersection Cardinality
This repository contains our implementation of SetXor and SetXorDyn, which provides the intersection cardinality estimation of static and streaming sets respectively, while satisfying Local Differential Privacy (LDP). Moreover, we offer enhanced models, specifically SetXor-IVW and SetXorDyn-IVW, which utilize inverse variance weighting to enhance estimation precision. The implementation is based on [MurmurHash3](https://dl.acm.org/doi/abs/10.5555/3295222.3295407). Both of our SetXor and SetXorDyn construct a compact bit matrix and performs bitwise XOR operations when adding elements or merging sketches. Additionally, they incorporate a random response mechanism to satisfy Differential Privacy (DP) and estimates the intersection cardinality through the estimation of the difference set cardinality. Moreover, we conducted extensive experiments to compare the performance of our SetXor with the method [SFM](https://arxiv.org/pdf/2302.02056.pdf) and the state-of-the-art method [LL](https://research.google/pubs/pub49177/). 

### Datasets
Our experiments encompass both synthetic datasets and real-world datasets, including network origin-destination traffic datasets and the webpage visits dataset in the static case, and the network traffic datasets and the BitcoinHeist dataset in the streaming case, to evaluate different methods. For the synthetic datasets, we randomly generate unique data from a 32-bit integer space, which is used to construct both static and streaming datasets. As for the network origin-destination traffic dataset, we utilize the AbileneTM dataset from the first week, which can be accessed through the following source: 
```url
https://www.cs.utexas.edu/~yzhang/research/AbileneTM/
```
For the webpage visits dataset, we estimate the common users who visit two different categories’ webpage, which can be referred through: 
```url
https://doi.org/10.24432/C5390X
```
And for the BitcoinHeist dataset, we construct sketches to record the Bitcoin transaction records, which can be obtained through:
```url
https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset
```
For the network traffic dataset, we utilize network flows captured from July 3 to July 7, 2017, which can be referred through: 
```url
https://www.unb.ca/cic/datasets/ids-2017.html
```

### Methods implemented
|   Method   |                         Description                          |               Reference                |
| :--------: | :----------------------------------------------------------: | :------------------------------------: |
|   SetXor   |                  the original SetXor sketch                  |         [setxor.py](setxor.py)         |
| SetXor-IVW | improved method for SetXor, providing more accurate estimation |         [setxor.py](setxor.py)         |
|   SetXorDyn   |                  the original SetXorDyn sketch                  |         [setxor_Dyn.py](setxor_Dyn.py)         |
| SetXorDyn-IVW | improved method for SetXorDyn, providing more accurate estimation |         [setxor_Dyn.py](setxor_Dyn.py)         |
|  [SFM-Sym](https://arxiv.org/abs/2302.02056)   |                SFM with deterministic merging                | [sfm.py](./baseline/sfm.py) |
|  [SFM-Xor](https://arxiv.org/abs/2302.02056)   |                   SFM with random merging                    | [sfm.py](./baseline/sfm.py) |
|    [HLL](https://dmtcs.episciences.org/3545/pdf)     |                    the HyperLogLog sketch                    | [hll.py](./baseline/hll.py) |
|     [FM](https://www.sciencedirect.com/science/article/pii/0022000085900418)     |                  the Flajolet-Martin sketch                  |  [fm.py](./baseline/fm.py)  |
|     [CL](https://research.google/pubs/pub49177/)     |                  the CascadingLegions                  |  [cl.py](./baseline/cl.py)  |
|     [LL](https://research.google/pubs/pub49177/)     |                  the LiquidLegions                  |  [ll.py](./baseline/ll.py)  |

We evaluate the performance of the eight methods mentioned above in estimating intersection cardinality in [`main_static.py`](main_static.py) and [`main_streaming.py`](main_streaming.py), used for static and streaming sets, respectively. Among them, HLL and FM do not incorporate privacy protection mechanisms. You can execute [`main_static.py`](main_static.py) and [`main_streaming.py`](main_streaming.py) with the following parameters:
| Parameters     | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| --method       | method name: SetXor/SetXor_IVW/SetXor_Dyn/SetXor_IVW_Dyn/SFM/HLL/FM                   |
| --dataset      | dataset path or synthetic to generate dataset               |
| --intersection | set intersection cardinality                                |
| --difference   | set difference cardinality                                  |
| --ratio        | skewness ratio used to control cardinalities of two sets    |
| --exp_rounds   | the number of experimental rounds                           |
| --hll_Msize    | the number of HLL sketch rows                               |
| --fm_Msize     | the number of FM sketch rows                                |
| --fm_Wsize     | the number of FM sketch columns                             |
| --sfm_Msize    | m of SFM sketch                                             |
| --sfm_Wsize    | w of SFM sketch                                             |
| --sfm_epsilon  | privacy budget of SFM                                       |
| --merge_method | the merge method of SFM deterministic/random                |
| --setxor_Msize | m of SetXor/SetXor_IVW/SetXor_Dyn/SetXor_IVW_Dyn sketch                               |
| --setxor_Wsize | w of SetXor/SetXor_IVW/SetXor_Dyn/SetXor_IVW_Dyn sketch                               |
| --cl_Msize | m of CascadingLegions                               |
| --cl_l | l of CascadingLegions                               |
| --ll_Msize | m of LiquidLegions                               |
| --ll_a | a of LiquidLegions                               |
| --epsilon      | the privacy budget |

### Quickstart
To install the requirements used in the experiment, simply run:
```bash
pip install -r requirements.txt
```

### Example
Use `main_static.py` to test a method in the static case, and use `main_streaming.py` to test a method in the streaming case.
When the intersection cardinality is 100000 and the difference cardinality is 10000 in the static case, use SetXor to estimate the intersection cardinality:
```bash
python main_static.py --method SetXor --intersection 100000 --difference 10000
```
When the intersection cardinality is 100000 and the difference cardinality is 10000 in the static case, use SetXor-IVW to estimate the intersection cardinality with a privacy budget of 2:
```bash
python main_static.py --method SetXor_IVW --intersection 100000 --difference 10000 --epsilon 2
```
When the intersection cardinality is 100000 and the difference cardinality is 10000 in the streaming case, use SetXorDyn to estimate the intersection cardinality:
```bash
python main_streaming.py --method SetXorDyn --intersection 100000 --difference 10000
```
When the intersection cardinality is 100000 and the difference cardinality is 10000 in the streaming case, use SetXorDyn-IVW to estimate the intersection cardinality with a privacy budget of 2:
```bash
python main_streaming.py --method SetXorDyn_IVW --intersection 100000 --difference 10000 --epsilon 2
```
When the intersection cardinality is 100000 and the difference cardinality is 10000 in the streaming case, use SFM-Sym to estimate the intersection cardinality:
```bash
python main_streaming.py --method SFM --intersection 100000 --difference 10000 --merge_method random
```
When the intersection cardinality is 100000 and the difference cardinality is 10000 in the streaming case, use SFM-Xor to estimate the intersection cardinality:
```bash
python main_streaming.py --method SFM --intersection 100000 --difference 10000 --merge_method deterministic
```
When the intersection cardinality is 1000000 and the difference cardinality is 100000 in the static case, use HLL to estimate the intersection cardinality with the sketch having 1024 rows:
```bash
python main_static.py --method HLL --intersection 1000000 --difference 100000 --hll_Msize 1024
```
When the intersection cardinality is 1000000 and the difference cardinality is 100000 in the static case, use FM to estimate the intersection cardinality:
```bash
python main_static.py --method FM --intersection 1000000 --difference 100000
```
Use CascadingLegions to estimate the intersection cardinality:
```bash
python main_streaming.py --method CL --intersection 1000000 --difference 100000
```
Use LiquidLegions to estimate the intersection cardinality:
```bash
python main_streaming.py --method LL --intersection 1000000 --difference 100000
```