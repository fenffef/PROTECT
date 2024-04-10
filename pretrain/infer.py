# -*- coding: utf-8 -*-
"""
@description: 
"""
import os
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import sys
sys.path.append('../..')
import wandb

# wandb.init(project="pycorrector_infer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/media/HD0/checkpoint/t5-robust-20221226/', help='save dir')
    args = parser.parse_args()
    return args


def predict(example_sentences):
    args = parse_args()
    model_dir = args.save_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    results = []
    for s in example_sentences:
        model_inputs = tokenizer(s, max_length=128, truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(**model_inputs, max_length=128)
        r = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(r)
    return results

def load_txt_dataset(text_list, split="train"):
    data = []
    for idx, line in enumerate(text_list):
        # line = line.replace(' ', '')
        linelist = line.strip('').split('\t')
        guid = "%s-%s" % (split, idx)
        tgt_text = linelist[1].lower().strip('\n')
        text_a = linelist[0].lower()
        data.append(InputExample(guid=guid, text_a=text_a, tgt_text=tgt_text))
    return data


if __name__ == '__main__':
    # example_sentences = [
    #     '货来的倒shi蛮快的只shi真de好小ya而且 味道也不好闻性价比不高',
    #     '有一个小问ti抱枕有一点高了趴着不是特别舒服价ge有点gui',
    #     '比想x中小太d。', 
    #     '我是买去送数学l师的',
    #     '小de可爱小得可lian',
    #     '还有在疫情期间不作为的狗zhengfu包括该死的daxue！！！',
    #     'opps这是可以说的吗 对政策zhengfu zhengzhi的不满不会被鲨的吧！',
    #     '拿dao货都jue得比预liao的小太多',
    #     '因某原因shi去送礼物de机hui可自己拿着也没法用',
    #     '在学xiao午睡的话应该派得上yong场因为它比较小方bian携带.我是买去送老师的',
    #     'wu论从书籍的内rong还是纸张zhi量都非常的差劲',
    #     'guan键是拿到的图shu封面与网上公布的封面大xiang径庭于是可确定suo购书是dao版的',
    #     '新lao版本相差太远本书一共才72页且寄来的书版本与描shu的极wei不fu。',
    #     '一般de书都是女装几乎没有男的bu适合pu通人',
    #     '图pian太老土le',
    #     '老是较书。',
    #     "我跟我朋唷打算去法国玩儿。",
    #     "少先队员因该为老人让坐。",
    #     "我们是新时代的接斑人",
    #     "我咪路，你能给我指路吗？",
    #     "他带了黑色的包，也带了照像机",
    #     '因为爸爸在看录音机，所以我没得看',
    #     '不过在许多传统国家，女人向未得到平等',
    # ]
    # print("开始预测...")
    # r = predict(example_sentences)
    # for i, o in zip(example_sentences, r):
    #     print(i, ' -> ', o)

    print("开始评估t5模型的纠错准召率...")
    
    from t5_corrector import T5Corrector
    from utils.eval import eval_robust
    from datasets import load_dataset

    # dataset = {}
    # dataset = load_dataset('fenffef/RobustT5_abb', use_auth_token=True)
    # train_data = dataset['train']['text']
    # validation_data = dataset['test']['text']
    # dataset['validation'] = load_txt_dataset(validation_data, split="validation")
    
    # 从本地文件加载数据集
    def load_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
            # Split each line by tab character (\t)
            data = [line.strip().split('\t') for line in lines]
            
            return data


    # 设置本地文件路径
    # train_file_path = 'path_to_train_file.txt'
    validation_file_path = '/media/HD0/T5-Corrector/src/finetune/RobustCSC/pinyin/test_pinyin.txt'

    # 加载训练和验证数据
    # train_data = load_txt_dataset(train_file_path, split='train')
    validation_data = load_txt_dataset(validation_file_path, split='validation')
    sources = load_txt(validation_file_path)

    model = T5Corrector()
    print("开始评估t5模型的句级纠错准召率...")
    result = model._predict(validation_data)
    eval_robust(result, validation_data)