import os

data_dir = r'D:\Data\med_ner_re'
ner_dir = os.path.join(data_dir, 'ner')

ner_train_path = os.path.join(ner_dir, 'new_train.json')
ner_test_path = os.path.join(ner_dir, 'new_val.json')
