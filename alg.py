import pulp
import json
import random
import time
import copy
import numpy as np


def alg1sim_ori():
    MyLP = pulp.LpProblem("alg1sim", sense=pulp.LpMinimize)

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
        MyLP += tempsum_y_m_gamma <= C

    # 等式4
    tempsum_update_time = 0
    for gamma in GAMMA:
        for m in M:
            tempsum_update_time += beta_n_gamma[m[0] * len(GAMMA) + gamma[0]] * (
                    1 - variables_y_m_gamma[m[0] * len(GAMMA) + gamma[0]]) * u
    MyLP += tempsum_update_time <= U

    MyLP.solve()
    objective = pulp.value(MyLP.objective)

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


def alg1sim_round(y_m_gamma, z_m, objective):
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
                    hat_y_m_gama[m[0]][gamma[0]]=0
                else:
                    if random.random() < (y_m_gamma[m[0]][gamma[0]] * 1.0 / z_m[m[0]]):
                        hat_y_m_gama[m[0]][gamma[0]]=1
                    else:
                        hat_y_m_gama[m[0]][gamma[0]]=0

    objective_hat = sum(hat_z_m)


    return hat_y_m_gama, hat_z_m, objective_hat

def try_assign_gamma_with_sat3(hat_y_gama_sat1, new_gamma_idx):
    '''
    尝试增加一个gamma[idx]分给当前m。其中已分给当前m的流的情况是hat_y_m_gama_sat1
    :param hat_y_gama_sat1: size=1*len(GAMMA)的0-1行，表示当前分配给当前m的所有gamma方案
    :param new_gamma_idx: 一个企图分给当前m的新gamma
    :return: True or False
    '''
    current_c = 0
    for i in range(len(hat_y_gama_sat1)):
        current_c += hat_y_gama_sat1[i]*GAMMA[i][3]
    current_c += GAMMA[new_gamma_idx][3]
    if current_c <=C:
        return True
    else:
        return False

def cal_visit_seq(y, is_opt):
    """
    计算当前gamma下访问m的次序，按从大到小
    :param y_m: size=(1*M)，float值为y_m_gamma中gamma列对应的值
    :param is_opt: 是否使用优化，即优化对m的访问次序，先访问概率大的
    :return: res = [1, 8, 2, ...]先访问y_m[1]再y_m[8]...
    """
    y_temp = np.copy(y)
    if is_opt == True:
        res = np.argsort(-y_temp)
    else:
        res = [i for i in range(len(y))] # 不使用排序
    return res

def recheck_z(y_m_gama, z_m):
    """
    根据y_m_gama重新计算z_m
    :param hat_y_m_gama_sat1:
    :param hat_z_m_sat1:
    :return: None
    """
    for i in range(len(y_m_gama)):
        is_zero = True
        for j in range(len(y_m_gama[i])):
            if y_m_gama[i][j]==1:
                is_zero = False
                break
        if is_zero == True:
            z_m[i] = 0

def alg1sim_sat1(hat_y_m_gama, hat_z_m, objective, y_m_gamma, use_opt):
    hat_y_m_gama_sat1 = np.zeros([len(M), len(GAMMA)])
    hat_z_m_sat1 = np.copy(hat_z_m)
    for i in range(len(GAMMA)):
        tempsum_y_m_gamma = 0
        vis_seq = cal_visit_seq(y_m_gamma.transpose()[i], use_opt)
        # for j in range(len(M)):  # 此时hat_y_m_gama可能一列有多个都为1（多个m）
        for j in vis_seq:  # 此时hat_y_m_gama可能一列有多个都为1（多个m）
            # print('now in ', i,'***',j)
            hat_y_m_gama_sat1[j][i] = hat_y_m_gama[j][i]
            tempsum_y_m_gamma += hat_y_m_gama[j][i]
            if tempsum_y_m_gamma == 1:  # 满足当前sum=1后，后面的都设为0了（即一条流分配给多个NF是大概率事件）
                break
        if tempsum_y_m_gamma == 0:  # 如果遍历m之后y全是0，即一个都没被分配，那么找到一个z_m=1的m，将y^gamma_m=1
            isfind = False
            for j in range(len(M)):
                if hat_z_m_sat1[j] == 1 and try_assign_gamma_with_sat3(hat_y_m_gama_sat1[j], i)==True: # 分配时顺便检查下sat3
                    hat_y_m_gama_sat1[j][i] = 1
                    isfind = True
                    break
            if isfind == False:  # 如果rounding之后NF全没开启（z全是0），随机开启一个
                runwhich = random.randint(0, len(M) - 1)
                hat_z_m_sat1[runwhich] = 1
                hat_y_m_gama_sat1[runwhich][i] = 1

    recheck_z(hat_y_m_gama_sat1, hat_z_m_sat1)

    objective_hat_sat1 = sum(hat_z_m_sat1)

    return hat_y_m_gama_sat1, hat_z_m_sat1, objective_hat_sat1

def alg1sim_sat3(hat_y_m_gama_sat1, hat_z_m_sat1, objective, y_m_gamma, use_opt):
    hat_y_m_gama_sat3 = np.zeros([len(M), len(GAMMA)])
    hat_z_m_sat3 = np.copy(hat_z_m_sat1)
    cur_free_m = [] # 当前没有开启的NF
    for i in range(len(hat_z_m_sat3)):
        if hat_z_m_sat3[i]==0:
            cur_free_m.append([i, C])
    for i in range(len(M)):
        tempsum_y_m_gamma = 0
        vis_seq = cal_visit_seq(y_m_gamma[i], use_opt) # 先把y_m_gama[i]中，值比较大的满足了，小的就扔了
        for j in vis_seq:
            if hat_y_m_gama_sat1[i][j]==1 and hat_y_m_gama_sat1[i][j]*GAMMA[j][3]+tempsum_y_m_gamma <= C: # 如果该行满足，则一切正常
                tempsum_y_m_gamma += hat_y_m_gama_sat1[i][j]*GAMMA[j][3]
                hat_y_m_gama_sat3[i][j]=1
            elif hat_y_m_gama_sat1[i][j]==1:# 否则开启一个新NF
                print('find a disobey in m=',i,' f=',j)
                for idx in range(len(cur_free_m)):
                    # cur_free_m[idx][0] free_id
                    # cur_free_m[idx][1] free_cap
                    # print(cur_free_m[idx][0],' ---',cur_free_m[idx][1])
                    if cur_free_m[idx][1]>=GAMMA[j][3]:
                        cur_free_m[idx][1] -= GAMMA[j][3]
                        hat_z_m_sat3[cur_free_m[idx][0]]=1
                        hat_y_m_gama_sat3[cur_free_m[idx][0]][j]=1
                        break
    recheck_z(hat_y_m_gama_sat3, hat_z_m_sat3)
    objective_hat_sat3 = sum(hat_z_m_sat3)
    return hat_y_m_gama_sat3, hat_z_m_sat3, objective_hat_sat3

def ALG1SIM():
    f.write('M(NFs) : ' + str(M) + '\n')
    f.write('GAMMA(flows) : ' + str(GAMMA) + '\n')
    f.write('T(tenants) : ' + str(T) + '\n')
    f.write('C(capicity) : ' + str(C) + '\n')
    f.write('U(time limit) : ' + str(U) + '\n')
    f.write('u(time cost) : ' + str(u) + '\n')
    f.write('binary(if trans {0,1} to [0,1]) : ' + str(int(binary)) + '\n\n')

    # 原始LP ################################
    beta_n_gamma, y_m_gamma, z_m, objective = alg1sim_ori()
    # beta_n_gamma[i][j]  表示m i gamma j 是否为1，即原始gammaj是否被分配到了mi上，size=len(M)*len(GAMMA)
    # y_m_gamma[i][j]     表示m i gamma j 是否为1，即分配后gammaj是否被分配到了mi上，size=len(M)*len(GAMMA)
    # z_m[i]              表示m i 是否为1，即NF mi是否开启，size=len(M)
    # objective           表示目标函数值
    print('beta_n_gamma\n',beta_n_gamma)
    f.write('beta_n_gamma\n'+str(beta_n_gamma) + '\n')

    print('\nafter using alg1 LP ori\n')
    f.write('\nafter using alg1 LP ori\n')
    print('y_m_gamma\n', y_m_gamma)
    f.write('y_m_gamma\n' + str(y_m_gamma) + '\n')
    print('z_m\n', z_m)
    f.write('z_m\n' + str(z_m) + '\n')
    print('objective: ', objective)
    f.write("objective_hat: " + str(objective) + '\n')


    # rounding(alg1 ESRU) ###################################
    hat_y_m_gama, hat_z_m, objective_hat = alg1sim_round(y_m_gamma, z_m, objective)

    print('\nafter using alg1 rounding\n')
    f.write('\nafter using alg1 rounding\n')
    print('hat_y_m_gama\n', hat_y_m_gama)
    f.write('hat_y_m_gama\n' + str(hat_y_m_gama) + '\n')
    print('hat_z_m\n', hat_z_m)
    f.write('hat_z_m\n' + str(hat_z_m) + '\n')
    print("objective_hat: ", objective_hat)
    f.write("objective_hat: " + str(objective_hat) + '\n')
    if objective_hat == 0:
        print('objective_hat=0, no ratio outputs')
        f.write('objective_hat=0, no ratio outputs\n')
        f.write('===================================\n\n')
    else:
        ratio = objective * 1.0 / objective_hat
        f.write("ratio=" + str(ratio) + '\n')
        f.write('===================================\n\n')
        print("ratio=", ratio)
        print('===================================\n\n')

    # satisfy const 1 ###################################
    # 按m序号从小到大（或y_m_gamma值从大到小）纵向遍历hat_y_m_gama，保证每列只有一个1（每个gamma只对应一个m）
    hat_y_m_gama_sat1, hat_z_m_sat1, objective_hat_sat1 = alg1sim_sat1(hat_y_m_gama, hat_z_m, objective, y_m_gamma, 1)
    print('\nafter satisfy const 1\n')
    f.write('\nafter satisfy const 1\n')
    print('hat_y_m_gama_sat1\n', hat_y_m_gama_sat1)
    f.write('hat_y_m_gama_sat1\n' + str(hat_y_m_gama_sat1) + '\n')
    print('hat_z_m_sat1\n', hat_z_m_sat1)
    f.write('hat_z_m_sat1\n' + str(hat_z_m_sat1) + '\n')
    print("objective_hat_sat1: ", objective_hat_sat1)
    f.write("objective_hat_sat1: " + str(objective_hat_sat1) + '\n')
    if objective_hat_sat1 == 0:
        print('objective_hat_sat1=0, no ratio outputs')
        f.write('objective_hat_sat1=0, no ratio outputs\n')
        f.write('===================================\n\n')
    else:
        ratio_sat1 = objective * 1.0 / objective_hat_sat1
        f.write("ratio_sat1=" + str(ratio_sat1) + '\n')
        f.write('===================================\n\n')
        print("ratio_sat1=", ratio_sat1)
        print('===================================\n\n')

    # satisfy const 3 ###################################
    # 遍历每个m（横向），若超出了C约束，则把最右边（或者y_m_gamma最小的）的gamma踢掉，开一个新的z把踢掉的放在那
    hat_y_m_gama_sat3, hat_z_m_sat3, objective_hat_sat3 = alg1sim_sat3(hat_y_m_gama_sat1, hat_z_m_sat1, objective, y_m_gamma, 1)
    print('\nafter satisfy const 3\n')
    f.write('\nafter satisfy const 3\n')
    print('hat_y_m_gama_sat3\n', hat_y_m_gama_sat3)
    f.write('hat_y_m_gama_sat3\n' + str(hat_y_m_gama_sat3) + '\n')
    print('hat_z_m_sat3\n', hat_z_m_sat3)
    f.write('hat_z_m_sat3\n' + str(hat_z_m_sat3) + '\n')
    print("objective_hat_sat3: ", objective_hat_sat3)
    f.write("objective_hat_sat3: " + str(objective_hat_sat3) + '\n')
    if objective_hat_sat3 == 0:
        print('objective_hat_sat3=0, no ratio outputs')
        f.write('objective_hat_sat3=0, no ratio outputs\n')
        f.write('===================================\n\n')
    else:
        ratio_sat3 = objective * 1.0 / objective_hat_sat3
        f.write("ratio_sat3=" + str(ratio_sat3) + '\n')
        f.write('===================================\n\n')
        print("ratio_sat3=", ratio_sat3)
        print('===================================\n\n')

    return objective,objective_hat,objective_hat_sat1,objective_hat_sat3


if __name__ == '__main__':
    M = []  # NF，如[[0, 0], [1, 0], [2, 0]]，id、服务类别
    C = 1700  # NF服务能力
    GAMMA = []  # 流，如[[0, 0, -1, 70], [1, 0, -1, 80]]，id、从、到、流量
    T = []  # tenant
    U = 100  # 更新时间限制
    u = 5
    binary = False
    CYCLE = 3

    topo_name = ['topo_n500_t300_f5000_1.json','topo_n500_t300_f5000_2.json']
    # topo_name = ['topo_n500_t300_f500_1.json', 'topo_n500_t300_f500_2.json']
    CYCLE_TOPO = len(topo_name)
    res = []

    for topo_id in range(CYCLE_TOPO):
        with open(topo_name[topo_id]) as json_file:
            data = json.load(json_file)
            for i in range(len(data['nf_list'])):
                M.append([data['nf_list'][i]['id'], data['nf_list'][i]['type']])
            for i in range(len(data['flow_list'])):
                GAMMA.append(
                    [data['flow_list'][i]['id'], data['flow_list'][i]['fromwhere'], data['flow_list'][i]['towhere'],
                     data['flow_list'][i]['traffic']])
            for i in range(len(data['tenant_list'])):
                T.append(data['tenant_list'][i]['id'])

        filename = 'ALG1SIM_n' + str(len(M)) + '_t' + str(len(T)) + '_f' + str(len(GAMMA)) + '_' + str(binary)
        f = open('log/' + filename + '.txt', 'w')
        objective = np.zeros(CYCLE)
        objective_hat = np.zeros(CYCLE)
        objective_hat_sat1 = np.zeros(CYCLE)
        objective_hat_sat3 = np.zeros(CYCLE)
        for i in range(CYCLE):
            print('now in cycle: ', i)
            f.write('now in cycle: ' + str(i))
            objective[i], objective_hat[i], objective_hat_sat1[i], objective_hat_sat3[i] = ALG1SIM()
        f.close()

        # temp`````````````
        print(objective, '\n', objective_hat, '\n', objective_hat_sat1, '\n', objective_hat_sat3, '\n')
        res.append(objective)
        res.append(objective_hat)
        res.append(objective_hat_sat1)
        res.append(objective_hat_sat3)
        # `````````````````````````````
    print(res)