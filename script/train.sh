#!/usr/bin/env bash

echo "Start few-shot test..."

bash train_abb.sh

bash train_chaizi.sh

bash train_csc.sh

bash train_mix.sh

bash train_pinyin.sh

# bash train_xingsi.sh

# bash train_yinsi.sh
