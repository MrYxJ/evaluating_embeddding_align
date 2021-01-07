#!/usr/bin/python3
# encoding: utf-8
# Author MrYx
# @Time: 2020/11/24 23:06

import numpy as np
import falconn
import math
import timeit
from utils import *
import os
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import scipy
import time

def generate_candidate_threshold(entity_embedding=None, data_ids="OpenEA", path="", threshold=0.2, output_path=False,
                                 entity_file="ent_embeds,npy", normalize=True, metric="euclidean", lsh_family="hyperplane", number_of_tables=500):

    """
    :param entity_embedding:
    :param data_ids:
    :param path:
    :param threshold:
    :param output_path:
    :param entity_file:
    :param normalize:
    :param metric:  1.inner 向量的内积， 2.euclidean 欧几里的距离(l2 normaliztion 后与cosine distance 成正比)。
    :param lsh_family:
    :return:
    """

    if entity_embedding is None:
        entity_file_path = path + entity_file
        entity_embedding = np.load(entity_file_path)
        print("Load [%s] successfully!" % (entity_file_path))

    if data_ids is "OpenEA":
        ent2id1, id2ent1, max_id = read_ent_id(path + "kg1_ent_ids")
        ent2id2, id2ent2, max_id = read_ent_id(path + "kg2_ent_ids")
        paths = path.split('/')
        test_path = "/".join([paths[1], paths[2], paths[3], "datasets", paths[7], paths[8], paths[9]])
        test_ids = []
        with open('/' + test_path + r"/test_links", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().split("\t")
                id1, id2 = int(ent2id1[items[0]]), int(ent2id2[items[1]])
                # maxx_id = max(maxx_id, id1, id2)
                test_ids.append([id1, id2])
        data_ids = test_ids

    if data_ids is "dbp15k":
        # train_ids = read_ids(path+"sup_ent_ids")
        test_ids = read_ids(path + "ref_ent_ids")  # 只考虑测试集上匹配
        # test_ids.extend(train_ids)
        data_ids = test_ids
    data_ids = np.array(data_ids).astype(int)
    entity_embedding = entity_embedding.astype(np.float32)
    if metric == "euclidean":
        entity_embedding -= np.mean(entity_embedding, axis=0)

    Lvec = np.array([entity_embedding[e] for e in data_ids[:, 0]])
    Rvec = np.array([entity_embedding[e] for e in data_ids[:, 1]])
    if os.path.exists(path + "mapping_mat.npy"):   # OpenEA模型转换后的最终向量
        mapping = np.load(path + "mapping_mat.npy")
        #print("mapping shape:", mapping.shape)
        Lvec = np.matmul(Lvec, mapping)
        #print("load mapping succussuflly!")

    if normalize:
        Lvec = preprocessing.normalize(Lvec, norm="l2", axis=1)
        Rvec = preprocessing.normalize(Rvec, norm="l2", axis=1)

    seed = 119417657
    L_True = data_ids[:, 0].tolist()
    print("shape:", entity_embedding.shape)
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = entity_embedding.shape[1]
    if lsh_family == "crosspolytope":
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    elif lsh_family == "hyperplane":
        params_cp.lsh_family = falconn.LSHFamily.Hyperplane
    if metric == "euclidean":
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    elif metric == "inner":
        params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct

    params_cp.l = number_of_tables
    params_cp.num_rotations = 1
    params_cp.seed = seed
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 2
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(20, params_cp)
    # print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(Lvec)
    t2 = timeit.default_timer()

    print('Construction time: {}'.format(t2 - t1))
    query_object = table.construct_query_object()
    number_of_probes = number_of_tables
    print('Choosing number of probes: ', number_of_probes)
    query_object.set_num_probes(number_of_probes)
    t1 = timeit.default_timer()
    true_cnt = 0
    total = 0
    true_all = data_ids.shape[0]
    node_pairs = []
    print("Metric:", metric, "Threshold:", threshold)
    for ids_index, pair in enumerate(data_ids):
        ans = query_object.find_near_neighbors(Rvec[ids_index], threshold=threshold)
        #print(len(ans))
        for index in range(len(ans)):
            if pair[0] == L_True[ans[index]]:
                true_cnt += 1
                node_pairs.append((pair[0], pair[1], 1))
            else:
                node_pairs.append((L_True[ans[index]], pair[1], 0))
        total += len(ans)
    print('Threshold:[%f] True cnt:[%d] Generate All cnt:[%d] Total:[%d] Recall:[%f] P/E ratio:[%f] Metric:[%s]'
          % (threshold, true_cnt, total, true_all, true_cnt/true_all, total/true_all, metric))

    t2 = timeit.default_timer()
    print('Generate Candidate time: {}'.format(t2 - t1))
    if output_path == True:
        output_path = "/".join(path.split('/')[:-1]) + '/topk_' + str(threshold) + '_name_ngram'
        print('output path:', output_path)
        with open(output_path, 'w', encoding='utf8') as f:
            for pair in node_pairs:
                f.writelines(pair[0] + '\t' + pair[1] + '\t' + str(pair[2]) + '\n')


def generate_candidate_top(entity_embedding=None, data_ids="OpenEA", path="", top_k=5, output_path=False, entity_file="ent_embeds,npy",
                           normalize=True, metric="euclidean", lsh_family="crosspolytope", number_of_tables=500):
    """
    :param entity_embedding:
    :param data_ids:
    :param path:
    :param top_k:
    :param output_path:
    :param entity_file:
    :param normalize:
    :param metric:  inner 向量的内积， euclidean欧几里得距离
    :param lsh_family:
    :return:
    """
    if entity_embedding is None:
        entity_file_path = path + entity_file
        entity_embedding = np.load(entity_file_path)
        print("Load [%s] successfully!" % (entity_file_path))

    if data_ids is "OpenEA":
        ent2id1, id2ent1, max_id = read_ent_id(path + "kg1_ent_ids")
        ent2id2, id2ent2, max_id = read_ent_id(path + "kg2_ent_ids")
        paths = path.split('/')
        test_path = "/".join([paths[1], paths[2], paths[3], "datasets", paths[7], paths[8], paths[9]])
        #print("test_path:", test_path)
        test_ids = []
        with open('/' + test_path + r"/test_links", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().split("\t")
                id1, id2 = int(ent2id1[items[0]]), int(ent2id2[items[1]])
                test_ids.append([id1, id2])

        data_ids = test_ids
    if data_ids is"dbp15k":
        # train_ids = read_ids(path+"sup_ent_ids")
        test_ids = read_ids(path+"ref_ent_ids")   # 只考虑测试集上匹配
        # test_ids.extend(train_ids)
        data_ids = test_ids
    data_ids = np.array(data_ids).astype(int)
    entity_embedding = entity_embedding.astype(np.float32)
    if metric == "euclidean":
        entity_embedding -= np.mean(entity_embedding, axis=0)
    if normalize:
        entity_embedding = preprocessing.normalize(entity_embedding, norm="l2", axis=1)

    kg1_data = np.array([entity_embedding[e] for e in data_ids[:, 0]])
    if os.path.exists(path + "mapping_mat.npy"):  # OpenEA的规定
        mapping = np.load(path + "mapping_mat.npy")
        kg1_data = np.matmul(kg1_data, mapping)
        #print("load mapping  succussuflly!")
    kg1_index = data_ids[:, 0].tolist()
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = entity_embedding.shape[1]

    if lsh_family == "crosspolytope":
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    elif lsh_family == "hyperplane":
        params_cp.lsh_family = falconn.LSHFamily.Hyperplane
    if metric == "euclidean":
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    elif metric == "inner":
        params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
    params_cp.l = number_of_tables
    params_cp.num_rotations = 1
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(18, params_cp)
    #print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(kg1_data)
    t2 = timeit.default_timer()
    # print('Construction time: {}'.format(t2 - t1))
    query_object = table.construct_query_object()
    number_of_probes = number_of_tables
    # print('Choosing number of probes: ', number_of_probes)
    query_object.set_num_probes(number_of_probes)
    t1 = timeit.default_timer()
    true_cnt = 0
    total = 0
    true_all = data_ids.shape[0]
    node_pairs = []
    for pair in data_ids:
        R_entity = entity_embedding[pair[1]]
        ans = query_object.find_k_nearest_neighbors(R_entity, top_k)
        for index in range(len(ans)):
            if pair[0] == kg1_index[ans[index]]:
                true_cnt +=1
                node_pairs.append((pair[0], pair[1], 1))
            else:
                node_pairs.append((kg1_index[ans[index]], pair[1], 0))
        total += len(ans)
    print('Top k:[%d] True cnt:[%d] Generate All cnt:[%d] Total:[%d] Recall:[%f] P/E ratio:[%f] Metric:[%s]' %
          (top_k, true_cnt, total, true_all, true_cnt/true_all, total/true_all, metric))

    t2 = timeit.default_timer()
    print('Generate Candidate time: {}'.format(t2 - t1))
    if output_path == True:
        output_path = "/".join(path.split('/')[:-1]) + '/topk_' + str(top_k) + '_name_ngram'
        print('output path:', output_path)
        with open(output_path, 'w', encoding='utf8') as f:
            for pair in node_pairs:
                f.writelines(pair[0] + '\t' + pair[1] + '\t' + str(pair[2]) + '\n')


def generate_hits(entity_embedding=None, data_ids=None, path="", entity_file="ent_embeds,npy", metric = "cosine", hits=(1,5,10)):
    print("path:", path)
    if entity_embedding is None:
        entity_file_path = path + entity_file
        entity_embedding = np.load(entity_file_path)
    if data_ids is "OpenEA":
        ent2id1, id2ent1, max_id = read_ent_id(path + "kg1_ent_ids")
        ent2id2, id2ent2, max_id = read_ent_id(path + "kg2_ent_ids")
        paths = path.split('/')
        test_path = "/".join([paths[1], paths[2], paths[3], "datasets", paths[7], paths[8], paths[9]])
        #print("test_path:", test_path)
        test_ids = []
        maxx_id = 0
        with open('/' + test_path + r"/test_links", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().split("\t")
                id1, id2 = int(ent2id1[items[0]]), int(ent2id2[items[1]])
                maxx_id = max(maxx_id, id1, id2)
                test_ids.append([id1, id2])
        data_ids = test_ids
    if data_ids is "dbp15k":
        # train_ids = read_ids(path+"sup_ent_ids")
        test_ids = read_ids(path + "ref_ent_ids")  # 只考虑测试集上匹配
        # test_ids.extend(train_ids)
        data_ids = test_ids
    # if normalize:
    #     entity_embedding = preprocessing.normalize(entity_embedding, norm="l2", axis=1)
    test_len = len(data_ids)
    data_ids = np.array(data_ids).astype(int)
    Lvec = np.array([entity_embedding[e] for e in data_ids[:, 0]])
    Rvec = np.array([entity_embedding[e] for e in data_ids[:, 1]])
    if os.path.exists(path + "mapping_mat.npy"):
        mapping = np.load(path + "mapping_mat.npy")
        #print("mapping shape:", mapping.shape)
        Lvec = np.matmul(Lvec, mapping)
        #print("load mapping succussuflly!")
    if metric == "inner":
        sim_mat = np.matmul(Lvec, Rvec.T)
    elif metric == "cosine" or metric == "cityblock":
        sim_mat = 1 - scipy.spatial.distance.cdist(Lvec, Rvec, metric=metric)
    elif metric == "euclidean":
        sim_mat = 1 - euclidean_distances(Lvec, Rvec)
    sim_mat = sim_mat.astype(np.float32)
    top_lr = [0] * len(hits)
    MR_lr = 0
    ans_l = {}
    ans_r = {}
    for i in range(Lvec.shape[0]):
        rank = (-sim_mat[i, :]).argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(hits)):
            if rank_index < hits[j]:
                top_lr[j] += 1

        MR_lr += rank_index + 1
    top_rl = [0] * len(hits)
    MR_rl = 0
    for i in range(Rvec.shape[0]):
        rank = (-sim_mat[:, i]).argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(hits)):
            if rank_index < hits[j]:
                top_rl[j] += 1
        MR_rl += rank_index + 1
    print('For each left:')
    print('Mean Rank:', MR_lr / data_ids.shape[0])
    for i in range(len(top_lr)):
        print('Hits@%d: %.4f%%' % (hits[i], top_lr[i] / test_len * 100))
        ans_l[hits[i]] = top_lr[i] / test_len * 100

    print('For each right:')
    print('Mean Rank:', MR_rl / test_len)
    for i in range(len(top_rl)):
        print('Hits@%d: %.4f%%' % (hits[i], top_rl[i] / test_len * 100))
        ans_r[hits[i]] = top_rl[i] / test_len * 100
    return ans_l, ans_r


def static_openea_mode(path="/home/kgbnu/Code/output/results/", fold="/721_5fold/1/", model="MTransE",
                       threshold="", top_k=0, hits=(), metric="euclidean", csls=0, normalize=True, data_name=None,lsh_family="hyperplane",number_of_tables=500):
    """
    实现lshg
    Block评测：threshold 控制阈值，top_k取排名前k个。
    hits是alignment评测指标
    :param path:
    :param fold:
    :param model:
    :param threshold:
    :param top_k:
    :param hits:
    :param metric:
    :param csls:
    :param normalize:
    :param data_name:
    :return:
    """

    if data_name == None:
        data_name = ["/EN_FR_15K_V1", "/EN_DE_15K_V1", "/D_W_15K_V1", "/D_Y_15K_V1"]

    for data in data_name:
        data_path = path + model + data + fold
        data_path = data_path + os.listdir(data_path)[-1] + '/'
        print("Model:[%s] Data:[%s]" % (model, data_path))

        if threshold != "":
            generate_candidate_threshold(path=data_path, data_ids="OpenEA", entity_file="ent_embeds.npy",
                                         threshold=threshold, metric=metric, normalize=normalize, lsh_family=lsh_family,number_of_tables=number_of_tables)

        if top_k != 0:
            generate_candidate_top(path=data_path, data_ids="OpenEA", entity_file="ent_embeds.npy",
                                   top_k=top_k, metric=metric, normalize=normalize, lsh_family=lsh_family,number_of_tables=number_of_tables)

        if len(hits) !=0:
            generate_hits(path=data_path, data_ids="OpenEA", entity_file="ent_embeds.npy",
                          hits=hits, metric=metric)
        print("\n")

if __name__ == '__main__':
    models = ["MTransE", "AttrE", "MultiKE", "JAPE"]
    thresholds = [0, 0.4, math.sqrt(2), 2, 10]
    top_ks = [1, 5, 10]
    thre = math.sqrt(2)
    metric = "euclidean"  # inner
    hits_metrics = ["cityblock", "inner", "euclidean"]
    data_name = ["/EN_FR_15K_V1"]
    static_openea_mode(model=models[0], hits=(1, 5, 10), metric=metric, data_name=data_name)

    for threshold in thresholds:
        static_openea_mode(model=models[2], threshold=threshold, metric=metric, data_name=data_name, normalize=True, number_of_tables=3000)

    static_openea_mode(model=models[0], top_k=top_ks[0], metric=metric, data_name=data_name)

