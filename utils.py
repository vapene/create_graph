import scipy.stats as st
import networkx as nx

def get_parameters(cluster):
    lambdas = st.uniform.rvs(2, 11, cluster)
    n = st.poisson.rvs(lambdas)
    num_nodes = n.sum()
    mu1_edge, mu2_edge = st.uniform.rvs(0.3, 0.1), st.uniform.rvs(0.1, 0.07)
    mu = st.norm.rvs(loc=10, scale=10, size=4)
    G = nx.random_partition_graph(n, mu1_edge, mu2_edge)
    return lambdas, n, num_nodes, mu1_edge, mu2_edge, mu, G
# # print('lambdas',lambdas,'\n num',n,'\n mu_edge',mu1_edge,mu2_edge, '\n mu_nodes',mu)
# # lambdas [ 7.65337678 10.50481557 11.57470454  2.08851643]
# #  num [ 9 12 18  3]
# #  mu_edge 0.3300091980393784 0.1332434036165328
# #  mu_nodes [-1.72121794 18.36206971 20.092734   22.93099361]
def create_node_features(G,n,mu):
    n1,n2,n3,n4 = [*n]
    mu1, mu2, mu3, mu4= [*mu]
    cnt = 0
    for i in range(n1):
        G.nodes[i]['feature'] = st.norm.rvs(mu1, 3, 128)
        cnt += 1
    for j in range(cnt, cnt + n2):
        G.nodes[j]['feature'] = st.norm.rvs(mu2, 3, 128)
        cnt += 1
    for k in range(cnt, cnt + n3):
        G.nodes[k]['feature'] = st.norm.rvs(mu3, 3, 128)
        cnt += 1
    for l in range(cnt, cnt + n4):
        G.nodes[l]['feature'] = st.norm.rvs(mu4, 3, 128)
        cnt += 1
    return G

def create_label_dict(n):
    n1, n2, n3, n4 = [*n]
    label_dict = {}
    cnt = 0
    for i in range(n1):
        label_dict[i] = 0
        cnt += 1
    for j in range(cnt, cnt + n2):
        label_dict[j] = 1
        cnt += 1
    for k in range(cnt, cnt + n3):
        label_dict[k] = 2
        cnt += 1
    for l in range(cnt, cnt + n4):
        label_dict[l] = 3
        cnt += 1
    return label_dict