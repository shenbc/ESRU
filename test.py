import random
import numpy as np

test_list = [0,0.1,0,0.25,0.65]

def rounding(test_list):
    num = len(test_list)
    res_list = [0] * num
    res_pos = 0
    remain_prob = 1
    for res_idx in range(num):
        if test_list[res_pos] == 0:
            res_pos = res_idx
        elif test_list[res_idx] == 0:
            continue
        else:
            r = random.random()
            if r >= test_list[res_pos] / remain_prob:
                remain_prob -= test_list[res_pos]
                res_pos = res_idx
            else:
                break

    res_list[res_pos] = 1
    return res_list

# print(rounding(test_list))

res_check_list = np.zeros([len(test_list)])

repeat_num = 10000

for _ in range(repeat_num):
    res_check_list += rounding(test_list)
    # print(rounding(test_list))
    # break

print(res_check_list / repeat_num)