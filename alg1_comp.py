import copy

import pulp
import random
import time
import numpy as np
from cal_func import cal_res


def alg_IP(M, C, GAMMA, T, U, u, binary):
    """
    对比算法，不考虑时间约束U，0-1的整数规划
    """
    MyLP = pulp.LpProblem(name="alg1sim", sense=pulp.LpMinimize)

    # 初始化变量
    beta_n_gamma = []
    for m in M:
        for gamma in GAMMA:
            if gamma[2] == m[0]:
                beta_n_gamma.append(1)
            else:
                beta_n_gamma.append(0)

    variables_y_m_gamma = []
    if binary:
        for m in M:
            for gamma in GAMMA:
                variables_y_m_gamma.append(pulp.LpVariable(
                    'y_m' + str(m[0]) + '_gamma' + str(gamma[0]), lowBound=0, upBound=1, cat=pulp.LpBinary))
    else:
        for m in M:
            for gamma in GAMMA:
                variables_y_m_gamma.append(pulp.LpVariable(
                    'y_m' + str(m[0]) + '_gamma' + str(gamma[0]), lowBound=0, upBound=1, cat=pulp.LpContinuous))

    variables_z_m = []
    if binary:
        for i in range(len(M)):
            variables_z_m.append(pulp.LpVariable(
                'z_m' + str(i), lowBound=0, upBound=1, cat=pulp.LpBinary))
    else:
        for i in range(len(M)):
            variables_z_m.append(pulp.LpVariable(
                'z_m' + str(i), lowBound=0, upBound=1, cat=pulp.LpContinuous))
    MyLP += sum(variables_z_m), "min total z"

    # 等式1
    for gamma in GAMMA:
        tempsum_y_m_gamma = 0
        for m in M:
            tempsum_y_m_gamma += variables_y_m_gamma[m[0] * len(GAMMA) + gamma[0]]
        MyLP += tempsum_y_m_gamma == 1

    # 等式2
    for gamma in GAMMA:
        for m in M:
            MyLP += variables_y_m_gamma[m[0] * len(GAMMA) + gamma[0]] <= variables_z_m[m[0]]

    # 等式3
    for m in M:
        tempsum_y_m_gamma = 0
        for gamma in GAMMA:
            tempsum_y_m_gamma += variables_y_m_gamma[m[0] * len(GAMMA) + gamma[0]] * gamma[3]
        MyLP += tempsum_y_m_gamma <= C * variables_z_m[m[0]]
        # 修正前 MyLP += tempsum_y_m_gamma <= C

    # 等式4
    # tempsum_update_time = 0
    # for gamma in GAMMA:
    #     for m in M:
    #         tempsum_update_time += beta_n_gamma[m[0] * len(GAMMA) + gamma[0]] * (
    #                 1 - variables_y_m_gamma[m[0] * len(GAMMA) + gamma[0]])
    # MyLP += (tempsum_update_time <= U/u)

    MyLP.solve()
    # pulp.LpSolverDefault.msg = 1
    # MyLP.solve(pulp.PULP_CBC_CMD(msg=True))
    objective = pulp.value(MyLP.objective)
    print(pulp.LpStatus[MyLP.status])
    # assert (pulp.LpStatus[MyLP.status] == 'Optimal')
    if pulp.LpStatus[MyLP.status] != 'Optimal':
        print('!!!!!!not Optimal!!!!!!')
        print(pulp.LpStatus[MyLP.status],'\n\n\n')

    # 结果
    temp_beta_n_gamma = np.array(beta_n_gamma)
    beta_n_gamma = np.reshape(temp_beta_n_gamma, (len(M), len(GAMMA)))

    y_m_gamma = np.zeros([len(M), len(GAMMA)])
    for i in range(len(y_m_gamma)):
        for j in range(len(y_m_gamma[i])):
            y_m_gamma[i][j] = variables_y_m_gamma[i * len(GAMMA) + j].value()

    z_m = np.zeros(len(M))
    for i in range(len(M)):
        z_m[i] = variables_z_m[i].value()

    return beta_n_gamma, y_m_gamma, z_m, objective


def alg1sim_round(M, GAMMA, y_m_gamma, z_m, binary):
    # hat_z_m = []
    # hat_y_m_gama = []
    hat_z_m = np.zeros(len(M))
    hat_y_m_gama = np.zeros([len(M), len(GAMMA)])
    random.seed(time.time())
    if binary == False:
        for m in M:
            if random.random() < z_m[m[0]]:  # 遍历每个变量z_m，并以概率z_m将其设置为1
                # hat_z_m.append(1)
                hat_z_m[m[0]] = 1
            else:
                # hat_z_m.append(0)
                hat_z_m[m[0]] = 0
        for m in M:
            for gamma in GAMMA:
                # assert m[0]*len(GAMMA)+gamma[0] == the last pos in list
                if hat_z_m[m[0]] == 0:
                    # hat_y_m_gama.append(0)
                    hat_y_m_gama[m[0]][gamma[0]] = 0
                else:
                    if random.random() < (y_m_gamma[m[0]][gamma[0]] * 1.0 / z_m[m[0]]):
                        hat_y_m_gama[m[0]][gamma[0]] = 1
                    else:
                        hat_y_m_gama[m[0]][gamma[0]] = 0

    objective_hat = sum(hat_z_m)

    return hat_y_m_gama, hat_z_m, objective_hat


def alg_SJF(M, C, GAMMA):
    """
    SJF对比算法，为具有最小流量需求的请求选择负担最小的NF【不变目前分配，不影响更新时间约束，但会有badput】
    :return: 分配后所有的流 new_gamma
    """

    # 计算当前NF负载，和新到流new_gamma
    m_load = np.zeros(len(M))  # 当前NF负载
    new_gamma = copy.deepcopy(GAMMA)  # 输出的流分配结果
    for gamma in GAMMA:
        if gamma[2] != -1:
            m_load[gamma[2]] += gamma[3]

    # 对流量排序，并把最小流量分给负载最轻的NF
    new_gamma.sort(key=lambda x: x[-1])
    for gamma in new_gamma:
        if gamma[2] != -1:  # 如果流已经被分配则不考虑
            continue

        lowest_nf_id = np.argmin(m_load)
        if m_load[lowest_nf_id] + gamma[3] <= C:
            gamma[2] = lowest_nf_id
            m_load[lowest_nf_id] += gamma[3]
        else:
            break

    # new_gamma输出转化
    beta_n_gamma = []
    for m in M:
        for gamma in GAMMA:
            if gamma[2] == m[0]:
                beta_n_gamma.append(1)
            else:
                beta_n_gamma.append(0)
    temp_beta_n_gamma = np.array(beta_n_gamma)
    beta_n_gamma = np.reshape(temp_beta_n_gamma, (len(M), len(GAMMA)))
    y_m_gama = np.zeros([len(M), len(GAMMA)])
    z_m = np.zeros(len(M))
    for i in range(len(M)):
        for j in range(len(GAMMA)):
            if new_gamma[j][2] == i:
                y_m_gama[i][j] = 1
                z_m[i] = 1
    objective = np.sum(z_m)
    flow_redict = 0  # 更改映射的流的条数
    for i in range(len(M)):
        for j in range(len(GAMMA)):
            flow_redict += (int)(beta_n_gamma[i][j] * (1 - y_m_gama[i][j]))

    return beta_n_gamma, y_m_gama, z_m, objective, m_load, flow_redict


def realloc_gamma(new_gamma, m_load, m_idx, C, rec_pct, ext_pct):
    """
    将分配给NF[m_idx]的流分给其他负载大于rec_pct的NF
    :param new_gamma:
    :param m_idx:
    :return:
    """
    # 找到分配给NF[m_idx]的流
    for gamma in new_gamma:
        if gamma[2] == m_idx:
            # 当前找到了负载过低的NF上的一条流
            # 接下来遍历其他NF[m_idx2]，当这个NF负载大于最低阈值时执行分配
            for m_idx2 in range(len(m_load)):
                if m_idx2 == m_idx:
                    continue
                if m_load[m_idx2] + gamma[3] <= ext_pct * C:
                    gamma[2] = m_idx2
                    m_load[m_idx2] += gamma[3]
                    m_load[m_idx] -= gamma[3]
                    break

    return new_gamma, m_load


def alg_RB(M, C, GAMMA, rec_pct=0.2, ext_pct=0.9):
    # Rule-based，新到的一个流依次找NF，当放置后NF容量不超过ext_pct = 90%（或者过载）时，则放置
    # 找不到则新开一个NF
    # 遍历所有NF，遇到小于rec_pct = 20%的就将其流量分配给别人
    # 计算当前NF负载，和新到流new_gamma

    m_load = np.zeros(len(M))  # 当前NF负载
    new_gamma = copy.deepcopy(GAMMA)  # 输出的流分配结果
    for gamma in GAMMA:
        if gamma[2] != -1:
            m_load[gamma[2]] += gamma[3]

    # 遍历所有流，当流没有被分配时，按顺序分配到第一个可用NF
    for gamma in new_gamma:
        if gamma[2] == -1:
            for m_idx in range(len(m_load)):
                if m_load[m_idx]+gamma[3]<=C:
                    m_load[m_idx]+=gamma[3]
                    gamma[2] = m_idx
                    break


    # 遍历所有NF，遇到小于rec_pct = 20%的就将其流量分配给别人
    for m_idx in range(len(m_load)):
        if m_load[m_idx] <= (C * rec_pct):
            # 对于每个负载过低的NF，执行下面函数
            # print('\nreallocing')
            # print(m_idx)
            # print(m_load[m_idx])
            new_gamma, m_load = realloc_gamma(new_gamma, m_load, m_idx, C, rec_pct, ext_pct)

    # new_gamma输出转化
    beta_n_gamma = []
    for m in M:
        for gamma in GAMMA:
            if gamma[2] == m[0]:
                beta_n_gamma.append(1)
            else:
                beta_n_gamma.append(0)
    temp_beta_n_gamma = np.array(beta_n_gamma)
    beta_n_gamma = np.reshape(temp_beta_n_gamma, (len(M), len(GAMMA)))
    y_m_gama = np.zeros([len(M), len(GAMMA)])
    z_m = np.zeros(len(M))
    for i in range(len(M)):
        for j in range(len(GAMMA)):
            if new_gamma[j][2] == i:
                y_m_gama[i][j] = 1
                z_m[i] = 1
    objective = np.sum(z_m)
    flow_redict = 0  # 更改映射的流的条数
    for i in range(len(M)):
        for j in range(len(GAMMA)):
            flow_redict += (int)(beta_n_gamma[i][j] * (1 - y_m_gama[i][j]))

    return beta_n_gamma, y_m_gama, z_m, objective, m_load, flow_redict


def ALG1COMP(M, C, GAMMA, T, U, u, f):
    f.write('M(NFs) : ' + str(len(M)) + '\n')
    f.write('GAMMA(flows) : ' + str(len(GAMMA)) + '\n')
    f.write('T(tenants) : ' + str(T) + '\n')
    f.write('C(capicity) : ' + str(C) + '\n')
    f.write('U(time limit) : ' + str(U) + '\n')
    f.write('u(time cost) : ' + str(u) + '\n')

    # 对比算法，不考虑时间约束U，0-1的整数规划（实际是小数规划 + rounding） ################################
    beta_n_gamma, y_m_gamma_compIP, z_m_compIP, objective_compIP = alg_IP(M, C, GAMMA, T, U, u, binary=False)
    hat_y_m_gama_compIP, hat_z_m_compIP, objective_compIP2 = alg1sim_round(M, GAMMA, y_m_gamma_compIP, z_m_compIP,
                                                                           binary=False)
    objective, flow_redict, throughtput, badput, disobayC = cal_res(M, C, GAMMA, hat_y_m_gama_compIP, binary=True)
    print('Using alg_IP\n', objective, '\t', flow_redict, '\t', throughtput, '\t', badput, '\t', disobayC, '\n')
    f.write('Using alg_IP\n' + str(objective) + '\t' + str(flow_redict) + '\t' + str(throughtput) + '\t' + str(
        badput) + '\t' + str(disobayC) + '\n')

    # print('\nafter using alg_IP\n')
    # f.write('\nafter using alg_IP\n')
    # print('hat_y_m_gama_compIP\n', hat_y_m_gama_compIP)
    # f.write('hat_y_m_gama_compIP\n' + str(hat_y_m_gama_compIP) + '\n')
    # print('hat_z_m_compIP\n', hat_z_m_compIP)
    # f.write('hat_z_m_compIP\n' + str(hat_z_m_compIP) + '\n')
    # print("objective_compIP2: ", objective_compIP2)
    # f.write("objective_compIP2: " + str(objective_compIP2) + '\n')
    # print("flow_redict: ", flow_redict)
    # f.write("flow_redict: " + str(flow_redict) + '\n')

    # 对比算法，最短作业优先 ################################
    beta_n_gamma, y_m_gama_SJF, z_m_SJF, objective_SJF, m_load_SJF, flow_redict_SJF = alg_SJF(M, C, GAMMA)
    objective, flow_redict, throughtput, badput, disobayC = cal_res(M, C, GAMMA, y_m_gama_SJF, binary=True)
    print('Using alg_SJF\n', objective, '\t', flow_redict, '\t', throughtput, '\t', badput, '\t', disobayC, '\n')
    f.write('Using alg_SJF\n' + str(objective) + '\t' + str(flow_redict) + '\t' + str(throughtput) + '\t' + str(
        badput) + '\t' + str(disobayC) + '\n')

    # print('\nafter using alg1sim_SJF\n')
    # f.write('\nafter using alg1sim_SJF\n')
    # print('y_m_gama_SJF\n', y_m_gama_SJF)
    # f.write('y_m_gama_SJF\n' + str(y_m_gama_SJF) + '\n')
    # print('z_m_SJF\n', z_m_SJF)
    # f.write('z_m_SJF\n' + str(z_m_SJF) + '\n')
    # print("objective_SJF: ", objective_SJF)
    # f.write("objective_SJF: " + str(objective_SJF) + '\n')
    # print("m_load_SJF: ", m_load_SJF)
    # f.write("m_load_SJF: " + str(m_load_SJF) + '\n')
    # print("flow_redict_SJF: ", flow_redict_SJF)
    # f.write("flow_redict_SJF: " + str(flow_redict_SJF) + '\n')

    # 对比算法，RB ################################
    beta_n_gamma, y_m_gama_RB, z_m_RB, objective_RB, m_load_RB, flow_redict_RB = alg_RB(M, C, GAMMA, rec_pct=0.4, ext_pct=0.95)
    objective, flow_redict, throughtput, badput, disobayC = cal_res(M, C, GAMMA, y_m_gama_RB, binary=True)
    print('Using alg_RB\n', objective, '\t', flow_redict, '\t', throughtput, '\t', badput, '\t', disobayC, '\n')
    f.write('Using alg_RB\n' + str(objective) + '\t' + str(flow_redict) + '\t' + str(throughtput) + '\t' + str(
        badput) + '\t' + str(disobayC) + '\n')

    # print('\nafter using alg1sim_RB\n')
    # f.write('\nafter using alg1sim_RB\n')
    # print('y_m_gama_RB\n', y_m_gama_RB)
    # f.write('y_m_gama_RB\n' + str(y_m_gama_RB) + '\n')
    # print('z_m_RB\n', z_m_RB)
    # f.write('z_m_RB\n' + str(z_m_RB) + '\n')
    # print("objective_RB: ", objective_RB)
    # f.write("objective_RB: " + str(objective_RB) + '\n')
    # print("m_load_RB: ", m_load_RB)
    # f.write("m_load_RB: " + str(m_load_RB) + '\n')
    # print("flow_redict_RB: ", flow_redict_RB)
    # f.write("flow_redict_RB: " + str(flow_redict_RB) + '\n')
