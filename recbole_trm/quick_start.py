import logging
from logging import getLogger
import torch
import torch.optim as optim

import numpy as np
import pickle
import yaml
import importlib

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
from recbole_trm.utils import update_dict, read_news, get_doc_input, load_matrix, get_module, train, test, collate_fn
from recbole_trm.data import DatasetTrain, DatasetTest, NewsDataset
from recbole_trm.data import prepare_training_behavior_data
from torch.utils.data import DataLoader


def run_recbole_trm(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    if model in ['NRMS', 'NAML', 'NPA']:
        properties = open(r'./recbole_trm/properties/'+model+'.yaml')
        config = yaml.load(properties, Loader=yaml.FullLoader)
        init_seed(config['seed'], config['reproducibility'])
        # preprocess news data
        news, news_index, category_dict, subcategory_dict, word_dict = read_news(config)
        news_title, news_category, news_subcategory = get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, config)
        news_combined = np.concatenate([x for x in [news_title, news_category, news_subcategory] if x is not None], axis=-1)
        embedding_matrix, have_word = load_matrix(config['glove_embedding_path'], word_dict, config['word_embedding_dim'])
        # preprocess training behavior data
        total_sample_num = prepare_training_behavior_data(config)
        dataset = DatasetTrain(config, news_index, news_combined)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'])
        # train and valid
        module = get_module(model)
        model = module(config, embedding_matrix, num_category=len(category_dict), num_subcategory=len(subcategory_dict))
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        if config['enable_gpu']:
            model = model.cuda(config['n_gpu'])
        train(config, dataloader, model, optimizer)
        # test
        checkpoint = torch.load(config['ckpt_path'], map_location='cpu')
        subcategory_dict = checkpoint['subcategory_dict']
        category_dict = checkpoint['category_dict']
        word_dict = checkpoint['word_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {ckpt_path}")
        news_dataset = NewsDataset(news_combined)
        news_dataloader = DataLoader(news_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
        news_scoring = []
        with torch.no_grad():
            for input_ids in tqdm(news_dataloader):
                input_ids = input_ids.cuda(config['n_gpu'])
                news_vec = model.news_encoder(input_ids)
                news_vec = news_vec.to(torch.device('cpu')).detach().numpy()
                news_scoring.extend(news_vec)
        news_scoring = np.array(news_scoring)
        logging.info("news scoring num: {}".format(news_scoring.shape[0]))
        dataset = DatasetTest(config, news_index, news_scoring)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
        if config['enable_gpu']:
            model = model.cuda(config['n_gpu'])
        test(config, dataloader, model)
    else:
        # configurations initialization
        config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
        init_seed(config['seed'], config['reproducibility'])
        # logger initialization
        init_logger(config)
        logger = getLogger()

        logger.info(config)

        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        init_seed(config['seed'], config['reproducibility'])
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config['show_progress']
        )

        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
