# -*- coding: utf-8 -*-
"""
@author:FengXuan(fenffef@qq.com)
@description: PROTECT: Parameter-Efficient Tuning for Few-Shot Robust Chinese Text Correction
"""

import argparse
import torch
from tqdm import tqdm
import wandb
import os
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BertTokenizer, T5Tokenizer, MT5ForConditionalGeneration

# 超参数定义
parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='/media/HD0/checkpoint/t5-robust-20221226/')
# parser.add_argument("--model_name_or_path", default='/media/HD0/checkpoint/bart-mix-20221221/')
parser.add_argument('--train_path', type=str, default='/media/HD0/T5-Corrector/src/finetune/RobustCSC/csc/14train.txt',
                        help='train dataset')
parser.add_argument('--test_path', type=str, default='/media/HD0/T5-Corrector/src/finetune/RobustCSC/csc/14test.txt',
                        help='test dataset')
parser.add_argument("--template", default='prefix') #manual soft mix prefix
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--eval_steps', type=int, default=200, help='eval steps num')
parser.add_argument("--epoch", default=30)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--k_shot", type=int, default=512)
parser.add_argument("--zero_shot", type=bool, default=False)
parser.add_argument("--delta_type", type=str, default="protect", help="lora, bitfit")
parser.add_argument("--task", type=str, default='chaizi')
parser.add_argument("--output_dir", type=str, default='')
parser.add_argument("--prefix_length", type=int, default=5)
parser.add_argument("--seq_length", type=int, default=128)
parser.add_argument("--seed", type=int, default=12)
#todo 添加随机种子，全参数微调，t5-correct训练！
args = parser.parse_args()

# 设置随机种子
seed = args.seed
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.wandb:
    wandb.init(project="T5-Robust-few-shot")

print(args)


# 数据集加载
import datasets
from datasets import load_dataset
from openprompt.data_utils.utils import InputExample

print("加载数据集...")

class CscDataset(object):
    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def load(self):
        data_list = []
        for item in self.data:
            data_list.append(item['original_text'] + '\t' + item['correct_text'])
        return {'text': data_list}


dataset = {}

if args.train_path.endswith('.tsv'):
    dataset = load_dataset('text', data_files={'train': [args.train_path], 'test': args.test_path})
    train_dataset = dataset['train']
    valid_dataset = dataset['test']
if args.train_path.endswith('.txt'):
    dataset = load_dataset('text', data_files={'train': [args.train_path], 'test': args.test_path})
    train_dataset = dataset['train']
    valid_dataset = dataset['test']
elif args.train_path.endswith('.json'):
    d = CscDataset(args.train_path)
    data_dict = d.load()
    train_dataset = Dataset.from_dict(data_dict, split='train')

    d = CscDataset(args.test_path)
    data_dict = d.load()
    valid_dataset = Dataset.from_dict(data_dict, split='test')
else:
    raise ValueError('train_path must be tsv or json')


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


if args.k_shot==0:
    train_data = dataset['train']['text']
else:
    train_data = dataset['train']['text'][:args.k_shot]
validation_data = dataset['test']['text']
dataset['train'] = load_txt_dataset(train_data, split="train")
dataset['validation'] = load_txt_dataset(validation_data, split="validation")
print(dataset['validation'][0])
print("数据集加载完成...")

# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
# 加载模型
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
# _, _, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
# plm = AutoModel.from_pretrained(args.model_name_or_path)
# tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, from_slow = True)


# Instantiating the PrefixTuning Template !
# 初始化 PrefixTuning Template
if args.template == 'manual':
    from openprompt.prompts import ManualTemplate
    template_text = '{"placeholder":"text_a"} 纠正句子中的形似、音似、字符拆分、拼音和拼音缩写错误 {"mask"}.'
    mytemplate = ManualTemplate(model=plm, tokenizer=tokenizer, text=template_text)
elif args.template == 'soft':
    from openprompt.prompts import MixedTemplate
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
             text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"mask"}.')
elif args.template == 'mix':
    from openprompt.prompts import MixedTemplate
    mytemplate1 = MixedTemplate(model=plm, tokenizer=tokenizer, 
                        text='{"placeholder":"text_a"} {"soft": "纠错:"} {"mask"}.')
elif args.template == 'prefix':
    from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
    mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False, num_token=args.prefix_length)

# To better understand how does the template wrap the example, we visualize one instance.
# 可视化一个实例
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)


# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
    batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
    batch_size=32,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
#     batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
#     truncate_method="head")

# load the pipeline model PromptForGeneration.
if args.delta_type and not args.zero_shot:
    from opendelta import LoraModel, BitFitModel, BitFitModel, AdapterModel, CompacterModel, LowRankAdapterModel, ParallelAdapterModel

    if args.delta_type == "lora":
        delta_model = LoraModel(backbone_model=plm, modified_modules=["SelfAttention.q", "SelfAttention.v"])
    elif args.delta_type == "bitfit":
        delta_model = BitFitModel(backbone_model=plm, modified_modules=['SelfAttention'])
    elif args.delta_type == "adapter":
        delta_model = AdapterModel(backbone_model=plm, modified_modules=['SelfAttention'])
    elif args.delta_type == "compacter":
        delta_model = CompacterModel(backbone_model=plm)
    elif args.delta_type == "low_rank_adapter":
        delta_model = LowRankAdapterModel(backbone_model=plm)
    elif args.delta_type == "parallel_adapter":
        delta_model = ParallelAdapterModel(backbone_model=plm)

    delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    delta_model.log()
else:
    # Handle case where args.delta_type is empty
    pass  # Do nothing, simply skip the setting


from openprompt import PromptForGeneration
use_cuda = True
# prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=False, tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, tokenizer=tokenizer)
if use_cuda:
    prompt_model=  prompt_model.cuda()
else:
 
    from openprompt import PromptForGeneration
    use_cuda = True
    prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True, tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
    if use_cuda:
        prompt_model=  prompt_model.cuda()

generation_arguments = {
    "max_length": args.seq_length,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": True,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 10
}

def eval_by_model_batch(predict, dataset, verbose=False):
    import unicodedata
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    pos_num = 0
    neg_num = 0
    total_num = 0
    import time
    start_time = time.time()
    srcs = []
    tgts = []
    for line in dataset:
        line = line.strip()
        # 转换中文符号
        line = unicodedata.normalize('NFKC', line)
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        src = parts[0].lower()
        tgt = parts[1].lower()

        # 移除“的”，“得”，“地”影响
        src = src.replace("得", "的").replace("地", "的")
        tgt = tgt.replace("得", "的").replace("地", "的")
        srcs.append(src)
        tgts.append(tgt)

    res = predict
    for each_res, src, tgt in zip(res, srcs, tgts):
        if len(each_res) == 2:
            tgt_pred, pred_detail = each_res.replace("得", "的").replace("地", "的")
        else:
            tgt_pred = each_res.replace("得", "的").replace("地", "的").replace("～", "~")
        if verbose:
            print()
            print('input  :', src)
            print('truth  :', tgt)
            print('predict:', tgt_pred)

        # 负样本
        if src == tgt:
            neg_num += 1
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                if verbose:
                    print('neg right')
            # 预测为正
            else:
                FP += 1
                if verbose:
                    print('neg wrong')
        # 正样本
        else:
            pos_num += 1
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                if verbose:
                    print('pos right')
            # 预测为负
            else:
                FN += 1
                if verbose:
                    print('pos wrong')
        total_num += 1

    spend_time = time.time() - start_time
    print(total_num)
    print(TP)
    print(TN)
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    if args.wandb:
        wandb.log({'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1})
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'cost time:{spend_time:.2f} s, total num: {total_num}, pos num: {pos_num}, neg num: {neg_num}')
    return acc, precision, recall, f1

def evaluate(prompt_model, dataloader, verbose=False):
        generated_sentence = []
        groundtruth_sentence = []
        prompt_model.eval()

        for step, inputs in tqdm(enumerate(dataloader)):
            if use_cuda:
                inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
            
            generated_sentence.extend(output_sentence)
            # groundtruth_sentence.extend(inputs['tgt_text'])
            # print(output_sentence[0])
            # print(inputs['tgt_text'][0])
        
        acc, precision, recall, f1 =  eval_by_model_batch(generated_sentence, dataset=validation_data, verbose=verbose)
        if args.wandb:
            wandb.log({"acc": acc})
        print("dev_acc {}, dev_precision {} dev_recall: {} dev_f1: {}".format(acc, precision, recall, f1), flush=True)
        return generated_sentence, acc, precision, recall, f1

# if os.path.exists('/media/HD0/T5-Corrector/scripts/result/{}.txt'.format(args.task)):
#     pass
# else:
#     os.mknod('/media/HD0/T5-Corrector/scripts/result/{}.txt'.format(args.task))

# zero-shot test
if args.zero_shot:
    generated_sentence, acc, precision, recall, f1 = evaluate(prompt_model, validation_dataloader, verbose=False)
    print(args)
    print(("zero-shot_acc {}, zero-shot_precision {} zero-shot_recall: {} zero-shot_f1: {}".format(acc, precision, recall, f1)))
    with open('../scripts/output/{}.txt'.format(args.task), 'w', encoding='utf-8') as f:
        f.write(str(args) + '\n')
        f.write("k_shot = {}, task = {}, prefix_length = {}".format(args.k_shot, args.task, args.prefix_length) + '\n')
        f.write(f'zero-shot_acc_precision_recall_f1: {acc:.4f} {precision:.4f} {recall:.4f} {f1:.4f}')
        f.write('\n')
        f.write('\n')

# few_shot finetune
else:
    from transformers import AdamW
    # Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
    # only include the template's parameters in training.

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    },
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    from transformers.optimization import get_linear_schedule_with_warmup

    tot_step  = len(train_dataloader)*args.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

    # training and generation.
    global_step = 0
    tot_loss = 0
    log_loss = 0
    best_val_acc = 0
    best_val_precision= 0
    best_val_recall = 0
    best_val_f1 = 0
    acc_traces = []
    for epoch in tqdm(range(args.epoch)):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            global_step +=1
            if use_cuda:
                inputs = inputs.cuda()
            loss = prompt_model(inputs)
            if args.wandb:
                wandb.log({"loss": loss})
            loss.backward()
            tot_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if global_step % args.eval_steps ==0:
                print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/args.eval_steps, scheduler.get_last_lr()[0]), flush=True)
                log_loss = tot_loss
                generated_sentence, acc, precision, recall, f1 = evaluate(prompt_model, validation_dataloader, verbose=False
)
                if acc >= best_val_acc:
                        # torch.save(prompt_model.state_dict(),f"{args.project_root}/ckpts/{this_run_unicode}.ckpt")
                        best_val_acc = acc
                        best_val_precision = precision
                        best_val_recall = recall
                        best_val_f1 = f1
                        acc_traces.append(acc)
    
    generated_sentence, acc, precision, recall, f1 = evaluate(prompt_model, validation_dataloader, verbose=False)
    print(args)
    print("best_acc {}, best_precision {} best_recall: {} best_f1: {}".format(best_val_acc, best_val_precision, best_val_recall, best_val_f1), flush=True)
    
    with open('/media/HD0/T5-Corrector/scripts/result/{}.txt'.format(args.task), 'a', encoding='utf-8') as f:
        f.write("args.k_shot {}, args.task {}".format(args.k_shot, args.task) + '\n')
        if best_val_acc < acc:
            f.write(f'{args.k_shot}-shot_acc_precision_recall_f1: {acc:.4f} {precision:.4f} {recall:.4f} {f1:.4f}')
        else:
            f.write(f'{args.k_shot}-shot_acc_precision_recall_f1: {best_val_acc:.4f} {best_val_precision:.4f} {best_val_recall:.4f} {best_val_f1:.4f}')
        f.write('\n')
        f.write('\n')
