import pulp
import json
import random
import time
import copy
import numpy as np
from alg1 import ALG1SIM
from alg1_comp import ALG1COMP


def gamma_preset():
    """
    gamma列表前1/2是已有流量，删掉dynamic_precent1% (DYNAMIC_PCT1)
    后1/2为新到的未分配的，保留dynamic_precent2% (DYNAMIC_PCT2)
    :return: None，因为改的GAMMA是全局变量
    """
    global GAMMA
    dynamic1 = int(len(GAMMA) * 0.5 * DYNAMIC_PCT1)
    dynamic2 = int(len(GAMMA) * 0.5) - int(len(GAMMA) * 0.5 * DYNAMIC_PCT2)
    GAMMA_temp = GAMMA[dynamic1:-dynamic2]  # 前dynamic1个flow扔，从0.5开始要dynamic2个flow
    for i in range(len(GAMMA_temp)):
        GAMMA_temp[i][0] = i
    GAMMA = GAMMA_temp


if __name__ == '__main__':
    M = []  # NF，如[[0, 0], [1, 0], [2, 0]]，id、服务类别
    C = 300000  # NF服务能力333000
    GAMMA = []  # 流，如[[0, 0, -1, 70], [1, 0, -1, 80]]，id、从（租户）、到（nf）、流量，其中前1/2是已有流量，后1/2为新到的未分配的
    T = []  # tenant
    U = 2000  # 更新时间限制
    u = 0.2  # 一次所需时间，毫秒，不加约束时20957 = flow_redicted
    DYNAMIC_PCT1 = 0.2  # 已有的流结束的比例
    DYNAMIC_PCT2 = 0.2  # 新到的未分配的流的比例
    binary = False
    CYCLE = 1
    f = None

    # topo_name = ['./topo/topo_n30_t10_f24000_1.json']
    topo_name = ['./topo/topo_n30_t10_f2000_1.json', './topo/topo_n30_t10_f4000_1.json',
                 './topo/topo_n30_t10_f6000_1.json', './topo/topo_n30_t10_f8000_1.json',
                 './topo/topo_n30_t10_f10000_1.json', './topo/topo_n30_t10_f12000_1.json',
                 './topo/topo_n30_t10_f14000_1.json', './topo/topo_n30_t10_f16000_1.json',
                 './topo/topo_n30_t10_f18000_1.json', './topo/topo_n30_t10_f20000_1.json',
                 './topo/topo_n30_t10_f22000_1.json', './topo/topo_n30_t10_f24000_1.json']
    CYCLE_TOPO = len(topo_name)
    res = []

    for topo_id in range(CYCLE_TOPO):
        print('======', topo_name[topo_id], '======')
        with open(topo_name[topo_id]) as json_file:
            data = json.load(json_file)
            M = []
            GAMMA = []
            T = []
            for i in range(len(data['nf_list'])):
                M.append([data['nf_list'][i]['id'], data['nf_list'][i]['type']])
            for i in range(len(data['flow_list'])):
                GAMMA.append(
                    [data['flow_list'][i]['id'], data['flow_list'][i]['fromwhere'], data['flow_list'][i]['towhere'],
                     data['flow_list'][i]['traffic']])
            for i in range(len(data['tenant_list'])):
                T.append(data['tenant_list'][i]['id'])
        gamma_preset()

        # ALG1
        filename = 'ALG1SIM_n' + str(len(M)) + '_t' + str(len(T)) + '_f' + str(len(GAMMA)) + '_' + str(binary)
        f = open('log/' + filename + '.txt', 'w')
        for i in range(CYCLE):
            print('now in cycle: ', i, '\n')
            f.write('now in cycle: ' + str(i) + str('\n'))
            ALG1SIM(M, C, GAMMA, T, U, u, binary, f)
        f.close()

        # ALG1COMP
        filename = 'ALG1COMP_n' + str(len(M)) + '_t' + str(len(T)) + '_f' + str(len(GAMMA)) + '_' + str(binary)
        f = open('log/' + filename + '.txt', 'w')
        for i in range(CYCLE):
            print('now in cycle: ', i, '\n')
            f.write('now in cycle: ' + str(i) + str('\n'))
            ALG1COMP(M, C, GAMMA, T, U, u, f)
        f.close()
