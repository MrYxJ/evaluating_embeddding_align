# !/usr/bin/env python
# -*- coding:utf8 -*-
import io
import datetime
import re
from collections import defaultdict
import json
import gluonnlp as nlp
import numpy as np
import tqdm



def read_ent_id(path=""):
    """
    :param path:
    :return:
    """
    id2ent, ent2id, max_id = {}, {}, 0
    with open(path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            if "results" in path:   # OpenEA
                id2ent[items[1]] = items[0]
                ent2id[items[0]] = items[1]
                max_id = max(max_id, int(items[1]))
            else:
                id2ent[items[0]] = items[1]
                ent2id[items[1]] = items[0]
                max_id = max(max_id, int(items[0]))
    return ent2id, id2ent, max_id


def read_ids(path):
    """

    :param path:
    :return:
    """
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            ids.append([items[0], items[1]])
    return ids