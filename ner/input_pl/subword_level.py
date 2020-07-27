from lib.preprocess import utils

pl = [
    {
        'name': 'train_tokenizer',
        'func': utils.train_subword_tokenizer_by_tfds,
        'input_keys': ['input', 'vocab_size'],
        'output_keys': 'tokenizer',
        'show_dict': {'tokenizer_size': 'input_1'},
    },
    {
        'name': 'get vocab_size',
        'func': lambda x: x.vocab_size,
        'input_keys': ['tokenizer'],
        'output_keys': 'vocab_size',
        'show_dict': {'vocab_size': 'vocab_size'},
    },
    {
        'name': 'encode sentence to list of idx',
        'func': utils.encoder_string_2_subword_idx_by_tfds,
        'input_keys': ['tokenizer', 'input'],
        'output_keys': 'X',
        'show_dict': {'X': 'X'},
    },
    {
        'name': 'decode list of idx to list of tokens',
        'func': utils.decode_subword_idx_2_tokens_by_tfds,
        'input_keys': ['tokenizer', 'X'],
        'output_keys': 'input',
        'show_dict': {'input': 'input'},
    },
    {
        'name': 'calculate position for list of tokens',
        'func': utils.calculate_pos_for_list_of_list_tokens,
        'input_keys': ['input'],
        'output_keys': 'input',
        'show_dict': {'input': 'input'},
    }
]
