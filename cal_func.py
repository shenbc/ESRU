import numpy as np

def cal_res(M, C, GAMMA, y_m_gamma, binary):
    """
    计算实验所需结果
    其实不支持binary == False
    :param M:
    :param C:
    :param GAMMA:
    :param y_m_gamma:
    :param binary:
    :return:
    """
    # beta_n_gamma
    beta_n_gamma = []
    for m in M:
        for gamma in GAMMA:
            if gamma[2] == m[0]:
                beta_n_gamma.append(1)
            else:
                beta_n_gamma.append(0)
    temp_beta_n_gamma = np.array(beta_n_gamma)
    beta_n_gamma = np.reshape(temp_beta_n_gamma, (len(M), len(GAMMA)))

    # z_m
    z_m = np.zeros(len(M))
    if binary == True:
        for i in range(len(M)):
            for j in range(len(GAMMA)):
                if y_m_gamma[i][j] == 1:
                    z_m[i] = 1
    else:
        for i in range(len(M)):
            for j in range(len(GAMMA)):
                z_m[i] += y_m_gamma[i][j]

    # objective
    objective = np.sum(z_m)

    # flow_redict
    flow_redict = 0  # 更改映射的流的条数
    for i in range(len(M)):
        for j in range(len(GAMMA)):
            flow_redict += (int)(beta_n_gamma[i][j] * (1 - y_m_gamma[i][j]))

    # m_load
    m_load = np.zeros(len(M))
    for gamma in GAMMA:
        if gamma[2] != -1:
            m_load[gamma[2]] += gamma[3]

    # throughtput
    throughtput = np.sum(m_load)

    # badput
    badput = 0
    for i in range(len(M)):
        if m_load[i]>C:
            badput += (m_load[i]-C)

    # disobayC
    disobayC = 0
    for i in range(len(M)):
        if m_load[i]>C:
            disobayC += 1

    return objective, flow_redict, throughtput, badput, disobayC