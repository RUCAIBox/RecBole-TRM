import os
from tqdm import tqdm
import random
import logging


def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)


def prepare_training_behavior_data(config):
    random.seed(config['seed'])
    behaviors = []

    behavior_file_path = config['behavior_train_path']
    with open(behavior_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            iid, uid, time, history, imp = line.strip().split('\t')
            impressions = [x.split('-') for x in imp.split(' ')]
            pos, neg = [], []
            for news_ID, label in impressions:
                if label == '0':
                    neg.append(news_ID)
                elif label == '1':
                    pos.append(news_ID)
            if len(pos) == 0 or len(neg) == 0:
                continue
            for pos_id in pos:
                neg_candidate = get_sample(neg, config['npratio'])
                neg_str = ' '.join(neg_candidate)
                new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                behaviors.append(new_line)

    random.shuffle(behaviors)

    behaviors_file = []
    for i, line in enumerate(behaviors):
        behaviors_file.append(line)

    logging.info('Writing train files.')
    processed_file_path = config['processed_npbehavior_train_path']
    with open(processed_file_path, 'w') as f:
        f.writelines(behaviors_file)

    return len(behaviors)
