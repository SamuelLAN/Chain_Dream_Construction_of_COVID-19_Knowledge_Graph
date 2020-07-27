import json
import numpy as np
from ner.config import path
from lib import utils
from nltk.tokenize import word_tokenize, sent_tokenize
from matplotlib import pyplot as plt


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


# with open(path.ner_test_path, 'rb') as f:
with open(path.ner_train_path, 'rb') as f:
    data = f.readlines()
data = list(map(lambda x: json.loads(x), data))

for i, v in enumerate(data):
    data[i] = check_entities(v)

print(f'total data len: {len(data)}')

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
plt.xticks(list(range(0, 1000, 20)), fontsize=10)
plt.savefig(utils.get_relative_file_path('ner', 'analysis', 'plots', 'before_process', 'hist_of_count_words.png'), dpi=400)
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
bins = list(range(0, 30, 1))
plt.hist(length_list, bins=bins, edgecolor='#a0f0a0')
plt.title(f'histogram of count of sentences per example', fontsize=30)
plt.xlabel('count of sentences', fontsize=30)
plt.ylabel('numbers', fontsize=30)
plt.xticks(bins, fontsize=18)
plt.savefig(utils.get_relative_file_path('ner', 'analysis', 'plots', 'before_process', 'hist_of_count_sentences.png'), dpi=400)
plt.show()
plt.close()

# total data len: 9679
#
# ------------------------------------------
# character level length statistics:
# mean text length: 692.3632606674244
# std text length: 512.2443961570557
# max text length: 5224
# min text length: 29
#
# ------------------------------------------
# word level length statistics:
# mean text length: 110.57020353342287
# std text length: 88.78624684345115
# max text length: 909
# min text length: 6
#
# ------------------------------------------
# sentence level length statistics:
# mean text length: 3.706064676102903
# std text length: 3.3446630235558756
# max text length: 29
# min text length: 1
