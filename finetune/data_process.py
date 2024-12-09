data_path = '/media/HD0/Datasets/data/raw_data/RobustT5_data/MCTC_yinsi/yinsi.txt'
processed_data = '/media/HD0/Datasets/data/raw_data/RobustT5_data/MCTC_yinsi/test_yinsi.txt'

with open(data_path, 'r') as f:
    data = f.readlines()
    for idx, line in enumerate(data):
        # line = line.replace(' ', '')
        linelist = line.strip('').split('\t')
        if idx % 2 == 0:
            tgt_text = linelist[1].lower().strip('\n')
            text_a = linelist[0].lower()
        else:
            tgt_text = linelist[1].lower().strip('\n')
            text_a = linelist[1].lower().strip('\n')
        with open(processed_data, 'a') as f1:
            f1.write(text_a + '\t' + tgt_text + '\n')



