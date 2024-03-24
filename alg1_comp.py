import pulp
import random
import time
import numpy as np


def alg_IP(M, C, GAMMA, T, U, u, f):
    """
    对比算法，不考虑时间约束U，0-1的整数规划
    :param M:
    :param C:
    :param GAMMA:
    :param T:
    :param U:
    :param u:
    :param binary:
    :param f:
    :return:
    """
    MyLP = pulp.LpProblem(name="alg_IP", sense=pulp.LpMinimize)

    # 初始化变量
    beta_n_gamma = []
    for m in M:
        for gamma in GAMMA:
            if gamma[2] == m[0]:
                beta_n_gamma.append(1)
            else:
                beta_n_gamma.append(0)

    variables_y_m_gamma = []

    for m in M:
        for gamma in GAMMA:
            variables_y_m_gamma.append(pulp.LpVariable(
                'y_m' + str(m[0]) + '_gamma' + str(gamma[0]), lowBound=0, upBound=1, cat=pulp.LpBinary))

    variables_z_m = []
    for i in range(len(M)):
        variables_z_m.append(pulp.LpVariable(
            'z_m' + str(i), lowBound=0, upBound=1, cat=pulp.LpBinary))
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
        MyLP += tempsum_y_m_gamma <= C

    MyLP.solve()
    objective = pulp.value(MyLP.objective)
    assert (pulp.LpStatus[MyLP.status] == 'Optimal')

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


def alg_SJF(M, C, GAMMA, T, f):
    """
    SJF对比算法，为具有最小流量需求的请求选择负担最小的NF【不变目前分配，不影响更新时间约束，但会有badput】
    :return: 分配后所有的流 new_gamma
    """

    # 计算当前NF负载，和新到流new_gamma
    m_load = np.zeros(len(M))  # 当前NF负载
    new_gamma = []  # 输出的流分配结果
    for gamma in GAMMA:
        if gamma[2] != -1:
            m_load[gamma[2]] += gamma[3]
        new_gamma.append(gamma)

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

    return new_gamma


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
            # 接下来遍历其他NF，当这个NF负载大于最低阈值时执行分配
            for m_idx2 in range(len(m_load)):
                if m_load[m_idx2] >= rec_pct * C and m_load[m_idx2] + gamma[3] <= ext_pct * C:
                    gamma[2] = m_idx2
                    m_load[m_idx2] += gamma[3]
                    m_load[m_idx] -= gamma[3]

    return new_gamma, m_load


def alg_RB(M, C, GAMMA, rec_pct=0.2, ext_pct=0.9):
    # Rule-based，新到的一个流依次找NF，当放置后NF容量不超过ext_pct = 90%（或者过载）时，则放置
    # 找不到则新开一个NF
    # 遍历所有NF，遇到小于rec_pct = 20%的就将其流量分配给别人
    # 计算当前NF负载，和新到流new_gamma

    m_load = np.zeros(len(M))  # 当前NF负载
    new_gamma = []  # 输出的流分配结果
    for gamma in GAMMA:
        if gamma[2] != -1:
            m_load[gamma[2]] += gamma[3]
        new_gamma.append(gamma)

    for gamma in new_gamma:
        if gamma[2] != -1:  # 如果流已经被分配则不考虑
            continue

        for m_idx in range(len(m_load)):
            if m_load[m_idx] + gamma[3] <= C * ext_pct:
                gamma[2] = m_idx
                m_load[m_idx] += gamma[3]

    # 遍历所有NF，遇到小于rec_pct = 20%的就将其流量分配给别人
    for m_idx in range(len(m_load)):
        if m_load[m_idx] <= (C * rec_pct):
            # 对于每个负载过低的NF，执行下面函数
            new_gamma, m_load = realloc_gamma(new_gamma, m_load, m_idx, C, rec_pct, ext_pct)

    return new_gamma, m_load


def ALG1COMP(M, C, GAMMA, T, U, u, f):
    f.write('M(NFs) : ' + str(M) + '\n')
    f.write('GAMMA(flows) : ' + str(GAMMA) + '\n')
    f.write('T(tenants) : ' + str(T) + '\n')
    f.write('C(capicity) : ' + str(C) + '\n')
    f.write('U(time limit) : ' + str(U) + '\n')
    f.write('u(time cost) : ' + str(u) + '\n')

    # 对比算法，不考虑时间约束U，0-1的整数规划 ################################
    beta_n_gamma, y_m_gamma_compIP, z_m_compIP, objective_compIP = alg_IP(M, C, GAMMA, T, U, u, f)
    # beta_n_gamma[i][j]  表示m i gamma j 是否为1，即原始gammaj是否被分配到了mi上，size=len(M)*len(GAMMA)
    # y_m_gamma[i][j]     表示m i gamma j 是否为1，即分配后gammaj是否被分配到了mi上，size=len(M)*len(GAMMA)
    # z_m[i]              表示m i 是否为1，即NF mi是否开启，size=len(M)
    # objective           表示目标函数值
    print('beta_n_gamma\n', beta_n_gamma)
    f.write('beta_n_gamma\n' + str(beta_n_gamma) + '\n')

    print('\nafter using alg_IP\n')
    f.write('\nafter using alg_IP\n')
    print('y_m_gamma\n', y_m_gamma_compIP)
    f.write('y_m_gamma\n' + str(y_m_gamma_compIP) + '\n')
    print('z_m\n', z_m_compIP)
    f.write('z_m\n' + str(z_m_compIP) + '\n')
    print('objective: ', objective_compIP)
    f.write("objective_hat: " + str(objective_compIP) + '\n')

    # 对比算法，最短作业优先
    gamma_SJF = alg_SJF(M, C, GAMMA, T, f)

    # 对比算法，RB
