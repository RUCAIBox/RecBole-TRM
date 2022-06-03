"""
SSE-PT
################################################

Reference:
    Liwei Wu et al. "SSE-PT: Sequential Recommendation Via Personalized Transformer." in RecSys 2020.

Reference:
    https://github.com/wuliwei9278/SSE-PT

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SSEPT(SequentialRecommender):
    r"""
    SSEPT is implemented based on SASRec.
    """

    def __init__(self, config, dataset):
        super(SSEPT, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        # self.hidden_size = config['hidden_size']  # same as embedding_size
        self.user_hidden_size = config['user_hidden_size']
        self.item_hidden_size = config['item_hidden_size']
        self.hidden_size = self.user_hidden_size + self.item_hidden_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.n_users = dataset.num(self.USER_ID)

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.user_hidden_size, padding_idx=0) # concat with item_seq
        self.user_test_embedding = nn.Embedding(self.n_users, self.user_hidden_size, padding_idx=0) # concat with test_items
        self.item_embedding = nn.Embedding(self.n_items, self.item_hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, user_id):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq) # [B, L, D]
        user_emb = self.user_embedding(user_id) # [B, D]
        user_emb = user_emb.unsqueeze(1).expand_as(item_emb) # [B, L, D]
        input_emb = torch.cat([item_emb, user_emb], dim=-1) + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, item_seq_len, user_id)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            user_emb = self.user_embedding(user_id) # [B, D]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)

            pos_items_emb = torch.cat([pos_items_emb, user_emb], dim=-1)
            neg_items_emb = torch.cat([neg_items_emb, user_emb], dim=-1)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)

            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            test_item_emb = test_item_emb.unsqueeze(0).expand(seq_output.shape[0], self.n_items, -1) # [B, n_items, D]
            user_test_emb = self.user_embedding(user_id) # [B, D]
            user_test_emb = user_test_emb.unsqueeze(1).expand_as(test_item_emb) # [B, n_items, D]
            test_item_emb = torch.cat([test_item_emb, user_test_emb], dim=-1) # [B, n_items, D]
            seq_output = seq_output.unsqueeze(1).expand_as(test_item_emb) # [B, n_items, D]
            logits = torch.mul(seq_output, test_item_emb).sum(dim=-1)  # [B, n_items]


            # logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        user_id = interaction[self.USER_ID]
        user_test_emb = self.user_test_embedding(user_id)

        seq_output = self.forward(item_seq, item_seq_len, user_id)
        test_item_emb = self.item_embedding(test_item)
        user_test_emb = user_test_emb.expand_as(test_item_emb)
        test_item_emb = torch.cat([test_item_emb, user_test_emb], dim=-1)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        user_test_emb = self.user_test_embedding(user_id)

        seq_output = self.forward(item_seq, item_seq_len, user_id) # [B, D]
        test_items_emb = self.item_embedding.weight # [n_items, D]
        test_items_emb = test_items_emb.unsqueeze(0).expand(seq_output.shape[0], self.n_items, -1) # [B, n_items, D]
        user_test_emb = user_test_emb.unsqueeze(1).expand_as(test_items_emb) # [B, n_items, D]
        test_items_emb = torch.cat([test_items_emb, user_test_emb], dim=-1) # [B, n_items, D]
        seq_output = seq_output.unsqueeze(1).expand_as(test_items_emb) # [B, n_items, D]
        scores = torch.mul(seq_output, test_items_emb).sum(dim=-1)  # [B, n_items]
        # scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        return scores
