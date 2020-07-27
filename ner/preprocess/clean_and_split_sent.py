import re
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from nltk import word_tokenize, sent_tokenize
import json
from ner.config import path
from lib import utils
from nltk import sent_tokenize

__reg_split_sub_sent = re.compile(r'[,]')
max_words = 65
skip_sent_count = 0


def check_entities(_val):
    text = str(_val['text'])
    entities = _val['entities']
    len_entities = len(entities)

    # traverse all the entities and check if the position is correct or not
    for _i, entity_val in enumerate(entities[::-1]):
        _entity = entity_val['entity']
        in_text = text[entity_val['start']: entity_val['end']]

        is_same = in_text == _entity

        # if not correct
        if not is_same:
            # try to fix the error
            if _entity in text:
                entity_val['start'] = text.index(_entity)
                entity_val['end'] = entity_val['start'] + len(_entity)

            # remove the noise
            else:
                del entities[len_entities - 1 - _i]
    return _val


def split_sentence(_val):
    ret = []
    global skip_sent_count, max_words

    text = str(_val['text'])
    entities = list(_val['entities'])

    texts = sent_tokenize(text)
    # texts = list(map(lambda x: x.replace('`', "'").replace("''", '"'), texts))

    new_texts = []
    for tmp_text in texts:
        if len(word_tokenize(tmp_text)) <= max_words:
            new_texts.append(tmp_text)
        else:
            tmp_texts = __reg_split_sub_sent.split(tmp_text)
            tmp_texts = list(map(lambda x: x.strip(), tmp_texts))
            tmp_texts = list(filter(lambda x: len(word_tokenize(x)) <= max_words, tmp_texts))
            new_texts += tmp_texts

    cur = 0
    for tmp_text in new_texts:
        index_start = text.index(tmp_text, cur)
        index_end = index_start + len(tmp_text)

        if not entities or index_start < 0:
            break

        if len(word_tokenize(tmp_text)) > max_words:
            skip_sent_count += 1
            continue

        tmp_ret = {'text': tmp_text, 'entities': []}

        entity_val = entities.pop(0)
        while index_start <= entity_val['start'] and entity_val['end'] <= index_end:
            entity_val['start'] -= index_start
            entity_val['end'] -= index_start
            tmp_ret['entities'].append(entity_val)

            if not entities:
                entity_val = {'start': -1}
                break
            entity_val = entities.pop(0)

        if tmp_ret['entities']:
            tmp_ret = check_entities(tmp_ret)
            ret.append(tmp_ret)

        if index_start <= entity_val['start'] and entity_val['end'] <= index_end:
            pass
        else:
            entities.insert(0, entity_val)

    return ret


# with open(path.ner_test_path, 'rb') as f:
with open(path.ner_train_path, 'rb') as f:
    data = f.readlines()
data = list(map(lambda x: json.loads(x), data))

new_data = []

for i, val in enumerate(data):
    clean_val = check_entities(val)
    split_val = split_sentence(clean_val)

    new_data.append(split_val)

print('saving data ... ')
utils.write_json(utils.get_relative_file_path('ner', 'data', 'preprocessed', 'train_clean_split.json'), new_data)

data = reduce(lambda a, b: a + b, new_data)

entity_list = list(map(lambda x: x['entities'], data))

type_list = list(map(lambda x: list(set(list(map(lambda a: a['type'], x)))), entity_list))
type_list = list(set(reduce(lambda a, b: a + b, type_list)))

type_list.sort()
print('\n---------------------------------------------------------------')
print(f'entity types: {type_list}')
print(f'len entity types: {len(type_list)}')

begin_prefix = 'B-'
next_prefix = 'I-'
single_prefix = 'O-'
not_entity = 'N'

entity_labels = list(map(lambda x: begin_prefix + x, type_list)) + \
                list(map(lambda x: next_prefix + x, type_list)) + \
                list(map(lambda x: single_prefix + x, type_list)) + [not_entity]

print('\n---------------------------------------------------------------')
print(f'all labels: {entity_labels}')
print(f'number of labels: {len(entity_labels)}')

entity_len_list = list(map(lambda x: list(map(lambda a: len(word_tokenize(a['entity'])), x)), entity_list))
entity_len_list = list(reduce(lambda a, b: a + b, entity_len_list))
print(f'\n------------------------------------------')
print(f'Entities length statistics:')
print(f'mean entity word length: {np.mean(entity_len_list)}')
print(f'std entity word length: {np.std(entity_len_list)}')
print(f'max entity word length: {np.max(entity_len_list)}')
print(f'min entity word length: {np.min(entity_len_list)}')

plt.figure(figsize=(20., 20 * 4.8 / 10.4))
bins = list(range(0, 10, 1))
plt.hist(entity_len_list, bins=bins, edgecolor='#a0f0a0')
plt.title(f'histogram of entity word length', fontsize=30)
plt.xlabel('entity word length', fontsize=30)
plt.ylabel('numbers', fontsize=30)
plt.xticks(bins, fontsize=10)
plt.savefig(utils.get_relative_file_path('ner', 'analysis', 'plots', 'after_process', 'hist_of_entity_word_length.png'), dpi=400)
plt.show()
plt.close()

print(f'\n---------------------------------------------------------------')
print(f'\ntotal data len: {len(data)}')
print(f'skip sent count: {skip_sent_count}')

length_list = list(map(lambda x: len(x['text']), data))
print(f'\n------------------------------------------')
print(f'character level length statistics:')
print(f'mean text length: {np.mean(length_list)}')
print(f'std text length: {np.std(length_list)}')
print(f'max text length: {np.max(length_list)}')
print(f'min text length: {np.min(length_list)}')

length_list = list(map(lambda x: len(word_tokenize(x['text'])), data))
print(f'\n------------------------------------------')
print(f'word level length statistics:')
print(f'mean text length: {np.mean(length_list)}')
print(f'std text length: {np.std(length_list)}')
print(f'max text length: {np.max(length_list)}')
print(f'min text length: {np.min(length_list)}')

plt.figure(figsize=(20., 20 * 4.8 / 10.4))
plt.hist(length_list, bins=50, edgecolor='#a0f0a0')
plt.title(f'histogram of count of words per example', fontsize=30)
plt.xlabel('count of words', fontsize=30)
plt.ylabel('numbers', fontsize=30)
plt.xticks(list(range(0, 120, 5)), fontsize=10)
plt.savefig(utils.get_relative_file_path('ner', 'analysis', 'plots', 'after_process', 'hist_of_count_words.png'), dpi=400)
plt.show()
plt.close()

length_list = list(map(lambda x: len(sent_tokenize(x['text'])), data))
print(f'\n------------------------------------------')
print(f'sentence level length statistics:')
print(f'mean text length: {np.mean(length_list)}')
print(f'std text length: {np.std(length_list)}')
print(f'max text length: {np.max(length_list)}')
print(f'min text length: {np.min(length_list)}')

plt.figure(figsize=(20., 20 * 4.8 / 10.4))
bins = list(range(0, 5, 1))
plt.hist(length_list, bins=bins, edgecolor='#a0f0a0')
plt.title(f'histogram of count of sentences per example', fontsize=30)
plt.xlabel('count of sentences', fontsize=30)
plt.ylabel('numbers', fontsize=30)
plt.xticks(bins, fontsize=18)
plt.savefig(utils.get_relative_file_path('ner', 'analysis', 'plots', 'after_process', 'hist_of_count_sentences.png'),
            dpi=400)
plt.show()
plt.close()


# ---------------------------------------------------------------
# entity types: ['Chemical', 'ChemicalCompound', 'Disease', 'Drug', 'Gene', 'Organization', 'Phenotype', 'Virus']
# len entity types: 8
#
# ---------------------------------------------------------------
# all labels: ['B-Chemical', 'B-ChemicalCompound', 'B-Disease', 'B-Drug', 'B-Gene', 'B-Organization', 'B-Phenotype', 'B-Virus', 'I-Chemical', 'I-ChemicalCompound', 'I-Disease', 'I-Drug', 'I-Gene', 'I-Organization', 'I-Phenotype', 'I-Virus', 'O-Chemical', 'O-ChemicalCompound', 'O-Disease', 'O-Drug', 'O-Gene', 'O-Organization', 'O-Phenotype', 'O-Virus', 'N']
# number of labels: 25
#
# ---------------------------------------------------------------
# total data len: 20960
# skip sent count: 0
#
# ------------------------------------------
# character level length statistics:
# mean text length: 205.4510973282443
# std text length: 86.28903140370602
# max text length: 523
# min text length: 4
#
# ------------------------------------------
# word level length statistics:
# mean text length: 32.07767175572519
# std text length: 12.982610121683736
# max text length: 65
# min text length: 1
#
# ------------------------------------------
# sentence level length statistics:
# mean text length: 1.0
# std text length: 0.0
# max text length: 1
# min text length: 1
