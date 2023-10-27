import pulp
import json
import random
import time
import copy

def ALG1SIM(M, GAMMA, T, C, U, u, binary):
    MyLP = pulp.LpProblem("alg1sim", sense=pulp.LpMinimize)

    # 初始化变量
    beta_gamma_n = []
    for gamma in GAMMA:
        for m in M:
            if gamma[2] == m[0]:
                beta_gamma_n.append(1)
            else:
                beta_gamma_n.append(0)

    variables_y_m_gama = []
    if binary:
        for m in M:
            for gamma in GAMMA:
                variables_y_m_gama.append(pulp.LpVariable(
                    'y_m' + str(m[0]) + '_gamma' + str(gamma[0]), lowBound=0, upBound=1, cat=pulp.LpBinary))
    else:
        for m in M:
            for gamma in GAMMA:
                variables_y_m_gama.append(pulp.LpVariable(
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
            tempsum_y_m_gamma += variables_y_m_gama[m[0]*len(GAMMA)+gamma[0]]
        MyLP += tempsum_y_m_gamma == 1

    # 等式2
    for gamma in GAMMA:
        for m in M:
            MyLP += variables_y_m_gama[m[0]*len(GAMMA)+gamma[0]] <= variables_z_m[m[0]]

    # 等式3
    for m in M:
        tempsum_y_m_gamma = 0
        for gamma in GAMMA:
            tempsum_y_m_gamma += variables_y_m_gama[m[0]*len(GAMMA)+gamma[0]] * gamma[3]
        MyLP += tempsum_y_m_gamma <= C

    # 等式4
    tempsum_update_time = 0
    for gamma in GAMMA:
        for m in M:
            tempsum_update_time += beta_gamma_n[gamma[0]*len(M)+m[0]] * (1-variables_y_m_gama[m[0]*len(GAMMA)+gamma[0]]) * u
    MyLP += tempsum_update_time <= U

    MyLP.solve()

    # 结果
    filename = 'ALG1SIM_n' + str(len(M)) + '_t' + str(len(T)) + '_f' + str(len(GAMMA)) + '_' + str(binary)
    f = open('log/' + filename + '.txt', 'w')
    f.write('M(NFs) : ' + str(M) + '\n')
    f.write('GAMMA(flows) : ' + str(GAMMA) + '\n')
    f.write('T(tenants) : ' + str(T) + '\n')
    f.write('C(capicity) : ' + str(C) + '\n')
    f.write('U(time limit) : ' + str(U) + '\n')
    f.write('u(time cost) : ' + str(u) + '\n')
    f.write('binary(if trans {0,1} to [0,1]) : ' + str(int(binary)) + '\n\n')
    f.write('solutions:\n')

    for i, v in enumerate(MyLP.variables()):
        # print(v.name, "=", v.varValue)
        f.write(str(v.name) + "=" + str(v.varValue) + '\n')
    objective = pulp.value(MyLP.objective)
    f.write("objective=" + str(objective)+ '\n')
    f.write('===================================\n\n')
    print("objective=", objective)
    print('===================================\n\n')

    # rounding(alg1 ESRU)
    variables_hat_z_m = []
    variables_hat_y_m_gama = []
    random.seed(time.time())
    if binary == False:
        for m in M:
            if random.random()<variables_z_m[m[0]].value(): # 遍历每个变量z_m，并以概率z_m将其设置为1
                variables_hat_z_m.append(1)
            else:
                variables_hat_z_m.append(0)
        for m in M:
            for gamma in GAMMA:
                # assert m[0]*len(GAMMA)+gamma[0] == the last pos in list
                if variables_hat_z_m[m[0]] == 0:
                    variables_hat_y_m_gama.append(0)
                else:
                    if random.random()<(variables_y_m_gama[m[0] * len(GAMMA) + gamma[0]].value() *1.0 / variables_z_m[m[0]].value()):
                        variables_hat_y_m_gama.append(1)
                    else:
                        variables_hat_y_m_gama.append(0)
        print('\nafter using alg1 rounding\n')
        f.write('\nafter using alg1 rounding\n')
        for m in M:
            for gamma in GAMMA:
                # print('y_m'+str(m[0])+'_gamma'+str(gamma[0])+' = '+str(variables_hat_y_m_gama[m[0]*len(GAMMA)+gamma[0]]))
                f.write('y_m'+str(m[0])+'_gamma'+str(gamma[0])+' = '+str(variables_hat_y_m_gama[m[0]*len(GAMMA)+gamma[0]]) + '\n')
        for m in M:
            # print('z_m'+str(m[0])+' = '+str(variables_hat_z_m[m[0]]))
            f.write('z_m'+str(m[0])+' = '+str(variables_hat_z_m[m[0]])+'\n')
    objective_hat = sum(variables_hat_z_m)
    f.write("objective=" + str(objective_hat) + '\n')
    print("objective: ", objective_hat)
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

    '''
    filename = 'ALG1SIM_n' + str(len(M)) + '_t' + str(len(T)) + '_f' + str(len(GAMMA)) + '_' + str(binary)
    f = open('log/' + filename + '.txt', 'w')
    variables_hat_y_m_gama = [0 for i in range(40)]
    variables_hat_z_m =[0,0,0,0]
    objective = 1.0
    '''
    # satisfy const 1
    variables_hat_y_m_gama_sat1 = [0 for i in range(len(M)*len(GAMMA))]
    variables_hat_z_m_sat1 = copy.deepcopy(variables_hat_z_m)
    for i in range(len(GAMMA)):
        tempsum_y_m_gamma = 0
        for j in range(len(M)):
            variables_hat_y_m_gama_sat1[j*len(GAMMA)+i] = variables_hat_y_m_gama[j*len(GAMMA)+i]
            tempsum_y_m_gamma += variables_hat_y_m_gama[j*len(GAMMA)+i]
            if tempsum_y_m_gamma == 1: # 满足当前sum=1后，后面的都设为0了（即一条流分配给多个NF是大概率事件）
                break
        if tempsum_y_m_gamma == 0: # 如果遍历m之后y全是0，即一个都没被分配，那么找到一个z_m=1的m，将y^gamma_m=1
            isfind = False
            for j in range(len(M)):
                if variables_hat_z_m_sat1[j] == 1:
                    variables_hat_y_m_gama_sat1[j*len(GAMMA)+i] = 1
                    isfind = True
                    break
            if isfind == False: # 如果rounding之后NF全没开启（z全是0），随机开启一个
                runwhich = random.randint(0,len(M)-1)
                variables_hat_z_m_sat1[runwhich] = 1
                variables_hat_y_m_gama_sat1[runwhich * len(GAMMA) + i] = 1

    print('\nafter satisfy const 1\n')
    f.write('\nsatisfy const 1\n')
    for m in M:
        for gamma in GAMMA:
            # print('y_m' + str(m[0]) + '_gamma' + str(gamma[0]) + ' = ' + str(
            #     variables_hat_y_m_gama_sat1[m[0] * len(GAMMA) + gamma[0]]))
            f.write('y_m' + str(m[0]) + '_gamma' + str(gamma[0]) + ' = ' + str(
                variables_hat_y_m_gama_sat1[m[0] * len(GAMMA) + gamma[0]]) + '\n')
    for m in M:
        # print('z_m' + str(m[0]) + ' = ' + str(variables_hat_z_m_sat1[m[0]]))
        f.write('z_m' + str(m[0]) + ' = ' + str(variables_hat_z_m_sat1[m[0]]) + '\n')
    objective_hat_sat1 = sum(variables_hat_z_m_sat1)
    f.write("objective_sat1=" + str(objective_hat_sat1) + '\n')
    print("objective_sat1: ", objective_hat_sat1)
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

    f.close()

if __name__ == '__main__':
    M = [] # NF，如[[0, 0], [1, 0], [2, 0]]，id、服务类别
    C = 170 # NF服务能力
    GAMMA = [] # 流，如[[0, 0, -1, 70], [1, 0, -1, 80]]，id、从、到、流量
    t = [] # tenant
    U = 100 # 更新时间限制
    u = 40
    topo_name = 'topo.json'

    with open(topo_name) as json_file:
        data = json.load(json_file)
        for i in range(len(data['nf_list'])):
            M.append([data['nf_list'][i]['id'],data['nf_list'][i]['type']])
        for i in range(len(data['flow_list'])):
            GAMMA.append([data['flow_list'][i]['id'],data['flow_list'][i]['fromwhere'],data['flow_list'][i]['towhere'],data['flow_list'][i]['traffic']])
        for i in range(len(data['tenant_list'])):
            t.append(data['tenant_list'][i]['id'])

    print('\n\n\n======binary = False(0 to 1)======')
    ALG1SIM(M=M, GAMMA = GAMMA, T=t, C = C, U = U, u = u, binary = False)
    # print('\n\n\n======binary = True(0 or 1 & no rounding)======')
    # ALG1SIM(M=M, GAMMA=GAMMA, T=t, C=C, U=U, u=u, binary=True)
            # NFs, flows, tenants, capacity, timelimt, binary