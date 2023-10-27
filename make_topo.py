import json
import random
import time


nf_num = 500
tenant_num = 300
flow_num = 100
file_name = 'topo_n' + str(nf_num) + '_t' + str(tenant_num)+ '_f' + str(flow_num) + '_1'
data = {}

# 生成NF
nfs=[]
for i in range(nf_num):
    temp_nf = {
      "name": "m_" + str(i),
      "type": 0,
      "id": i
    }
    nfs.append(temp_nf)


# 生成tenant
tenants = []
for i in range(tenant_num):
    temp_tenant = {
      "name": "t_" + str(i),
      "id": i
    }
    tenants.append(temp_tenant)


# 生成flow
flows = []
random.seed(time.time())
for i in range(flow_num):
    temp_flow = {
      "name": "gamma_" + str(i),
      "fromwhere": random.randint(0,tenant_num),
      "towhere": random.randint(0,nf_num),
      "id": i,
      "traffic": random.randint(0,100) # [0,100) int
    }
    flows.append(temp_flow)


# 载入
data = {
    "nf_list": nfs,
    "tenant_list" : tenants,
    "flow_list" : flows
}

# print(data)
with open('./' + file_name + '.json', 'w') as f:
    json.dump(data, f, indent=2)