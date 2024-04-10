# -*- coding: utf-8 -*-

import os
import sys
import time
from codecs import open
from random import sample
from xml.dom import minidom
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("../..")
from pycorrector.corrector import Corrector
from pycorrector.utils.io_utils import load_json, save_json
from pycorrector.utils.io_utils import load_pkl
from pycorrector.utils.math_utils import find_all_idx


def build_sighan_corpus(data_path, output_path):
    corpus = []
    sighan_data = load_pkl(data_path)
    for error_sentence, error_details in sighan_data:
        ids = []
        error_word = ''
        right_word = ''
        if not error_details:
            continue
        for detail in error_details:
            idx = detail[0]
            error_word = detail[1]
            right_word = detail[2]
            begin_idx = idx - 1
            ids.append(begin_idx)
        correct_sentence = error_sentence.replace(error_word, right_word)
        details = []
        for i in ids:
            details.append([error_sentence[i], correct_sentence[i], i, i + 1])
        line_dict = {"text": error_sentence, "correction": correct_sentence, "errors": details}
        corpus.append(line_dict)
    save_json(corpus, output_path)


def eval_corpus500_by_model(correct_fn, input_eval_path=eval_data_path, verbose=True):
    """
    句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    Args:
        correct_fn:
        input_eval_path:
        output_eval_path:
        verbose:

    Returns:
        Acc, Recall, F1
    """
    corpus = load_json(input_eval_path)
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    start_time = time.time()
    for data_dict in tqdm(corpus):
        src = data_dict.get('text', '')
        tgt = data_dict.get('correction', '')
        errors = data_dict.get('errors', [])

        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        tgt_pred, pred_detail = correct_fn(src)
        if verbose:
            print('/n')
            print('input  :', src)
            print('truth  :', tgt, errors)
            print('predict:', tgt_pred, pred_detail)

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                print('负样本 right')
            # 预测为正
            else:
                FP += 1
                print('负样本 wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                print('正样本 right')
            # 预测为负
            else:
                FN += 1
                print('正样本 wrong')
        total_num += 1
    spend_time = time.time() - start_time
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'cost time:{spend_time:.2f} s, total num: {total_num}')
    return acc, precision, recall, f1


def eval_sighan2015_by_model(correct_fn, sighan_path=sighan_2015_path, verbose=True):
    """
    SIGHAN句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    Args:
        correct_fn:
        input_eval_path:
        output_eval_path:
        verbose:

    Returns:
        Acc, Recall, F1
    """
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    start_time = time.time()
    with open(sighan_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            src = parts[0]
            tgt = parts[1]

            tgt_pred, pred_detail = correct_fn(src)
            if verbose:
                print('/n')
                print('input  :', src)
                print('truth  :', tgt)
                print('predict:', tgt_pred, pred_detail)

            # 负样本
            if src == tgt:
                # 预测也为负
                if tgt == tgt_pred:
                    TN += 1
                    print('right')
                # 预测为正
                else:
                    FP += 1
                    print('wrong')
            # 正样本
            else:
                # 预测也为正
                if tgt == tgt_pred:
                    TP += 1
                    print('right')
                # 预测为负
                else:
                    FN += 1
                    print('wrong')
            total_num += 1
        spend_time = time.time() - start_time
        acc = (TP + TN) / total_num
        precision = TP / (TP + FP) if TP > 0 else 0.0
        recall = TP / (TP + FN) if TP > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        print(
            f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
            f'cost time:{spend_time:.2f} s, total num: {total_num}')
        return acc, precision, recall, f1


def eval_sighan2015_by_model_batch(correct_fn, sighan_path=sighan_2015_path, verbose=True):
    """
    SIGHAN句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    Args:
        correct_fn:
        sighan_path:
        verbose:

    Returns:
        Acc, Recall, F1
    """
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    start_time = time.time()
    srcs = []
    tgts = []
    with open(sighan_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            src = parts[0]
            tgt = parts[1]

            srcs.append(src)
            tgts.append(tgt)

    res = correct_fn(srcs)
    for each_res, src, tgt in zip(res, srcs, tgts):
        if len(each_res) == 2:
            tgt_pred, pred_detail = each_res
        else:
            tgt_pred = each_res
        if verbose:
            print()
            print('input  :', src)
            print('truth  :', tgt)
            print('predict:', each_res)

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                print('right')
            # 预测为正
            else:
                FP += 1
                print('wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                print('right')
            # 预测为负
            else:
                FN += 1
                print('wrong')
        total_num += 1

    spend_time = time.time() - start_time
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'cost time:{spend_time:.2f} s, total num: {total_num}')
    return acc, precision, recall, f1
