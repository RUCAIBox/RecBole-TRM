from collections import Counter
from tqdm import tqdm
import torch
import numpy as np
import os
import logging
import importlib
from nltk.tokenize import word_tokenize
from recbole_trm.metrics import roc_auc_score, ndcg_score, mrr_score, get_mean, get_sum, print_metrics
from enum import Enum


class ModelType(Enum):
    SEQ=0
    NEWS=8


def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value


def read_news(config):
    news_path=config['news_path']
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}
    word_cnt = Counter()

    logging.info('Reading news.')
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, url, title_ent, abstract_ent = splited
            update_dict(news_index, doc_id)

            title = title.lower()
            title = word_tokenize(title)
            update_dict(news, doc_id, [title, category, subcategory])
            word_list = title
            if config['use_category']:
                update_dict(category_dict, category)
                word_list.append(category)
            if config['use_subcategory']:
                update_dict(subcategory_dict, subcategory)
                word_list.append(subcategory)
            word_cnt.update(word_list)

    word = [k for k, v in word_cnt.items() if v > config['filter_num']]
    word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
    return news, news_index, category_dict, subcategory_dict, word_dict


def get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, config):
    news_num = len(news) + 1
    news_title = np.zeros((news_num, config['num_words_title']), dtype='int32')
    news_category = np.zeros((news_num, 1), dtype='int32') if config['use_category'] else None
    news_subcategory = np.zeros((news_num, 1), dtype='int32') if config['use_subcategory'] else None

    for key in tqdm(news):
        title, category, subcategory = news[key]
        doc_index = news_index[key]

        for word_id in range(min(config['num_words_title'], len(title))):
            if title[word_id] in word_dict:
                news_title[doc_index, word_id] = word_dict[title[word_id]]

        if config['use_category']:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if config['use_subcategory']:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

    return news_title, news_category, news_subcategory


def load_matrix(embedding_file_path, word_dict, word_embedding_dim):
    embedding_matrix = np.zeros(shape=(len(word_dict) + 1, word_embedding_dim))
    have_word = []
    logging.info('Loading glove.')
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                word = line[0].decode()
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_word.append(word)
    return embedding_matrix, have_word


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def get_ckpt_path(config):
    return os.path.join(config['saved_path'], f'epoch-{config["epochs"]}.pt')


def get_module(model):
    return getattr(importlib.import_module(f'recbole_trm.model.{model.lower()}'), model)


def train(config, dataloader, model, optimizer, category_dict, subcategory_dict, word_dict):
    logging.info('Training.')
    for ep in range(config['epochs']):
        loss = 0.0
        accuary = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
            if config['enable_gpu']:
                log_ids = log_ids.cuda(config['n_gpu'], non_blocking=True)
                log_mask = log_mask.cuda(config['n_gpu'], non_blocking=True)
                input_ids = input_ids.cuda(config['n_gpu'], non_blocking=True)
                targets = targets.cuda(config['n_gpu'], non_blocking=True)

            bz_loss, y_hat = model(log_ids, log_mask, input_ids, targets)
            loss += bz_loss.data.float()
            accuary += acc(targets, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if cnt != 0 and cnt % config['log_steps'] == 0:
                logging.info(
                    '[{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        config['n_gpu'], cnt * config['batch_size'], loss.data / cnt, accuary / cnt)
                )

        ckpt_path = os.path.join(config['saved_path'], f'epoch-{ep+1}.pt')
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'category_dict': category_dict,
                'subcategory_dict': subcategory_dict,
                'word_dict': word_dict,
            }, ckpt_path)
        logging.info(f"Model saved to {ckpt_path}.")


def collate_fn(tuple_list):
    log_vecs = torch.FloatTensor([x[0] for x in tuple_list])
    log_mask = torch.FloatTensor([x[1] for x in tuple_list])
    news_vecs = [x[2] for x in tuple_list]
    labels = [x[3] for x in tuple_list]
    return (log_vecs, log_mask, news_vecs, labels)


def test(config, dataloader, model):
    model.eval()
    torch.set_grad_enabled(False)
    
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    
    local_sample_num = 0
    for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):
        local_sample_num += log_vecs.shape[0]

        if config['enable_gpu']:
            log_vecs = log_vecs.cuda(config['n_gpu'], non_blocking=True)
            log_mask = log_mask.cuda(config['n_gpu'], non_blocking=True)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device('cpu')).detach().numpy()

        for user_vec, news_vec, label in zip(user_vecs, news_vecs, labels):
            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(news_vec, user_vec)

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        if cnt != 0 and cnt % config['log_steps'] == 0:
            print_metrics(config['n_gpu'], local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))

    logging.info('[{}] local_sample_num: {}'.format(config['n_gpu'], local_sample_num))
    print_metrics('*', local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))
