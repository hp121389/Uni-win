import torch
import dataclasses
from typing import Optional
from torch import nn

from transformers import (PreTrainedModel, BertPreTrainedModel, BertConfig,
                          BertTokenizerFast)
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertOnlyNSPHead, BertForMaskedLM, BertLMHeadModel
from .modify_bert import BertModel

@dataclasses.dataclass
class UniRelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    head_preds: Optional[torch.FloatTensor] = None
    tail_preds: Optional[torch.FloatTensor] = None
    span_preds: Optional[torch.FloatTensor] = None

class UniRelModel(BertPreTrainedModel):
    """
    Model for learning Interaction Map
    """
    def __init__(self, config, model_dir=None):
        super(UniRelModel, self).__init__(config=config)#调用父类的初始化方法
        self.config = config
        if model_dir is not None: #如果目录存在，则加载预训练的BERT模型；否则，使用传入的配置创建一个新的BERT模型
            self.bert = BertModel.from_pretrained(model_dir, config=config)
        else:
            self.bert = BertModel(config)

        # Easy debug
        self.tokenizer = BertTokenizerFast.from_pretrained(  # 初始化了一个BERT分词器
            "bert-base-cased", do_basic_tokenize=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Dropout层，用于正则化和防止过拟合
        
        # Abaltion experiment 检查两个配置标志，确定是否进行某种消融实验
        if config.is_additional_att or config.is_separate_ablation:
            self.key_linear = nn.Linear(768, 64)
            self.value_linear = nn.Linear(768, 64)

    def forward(
        self,#输入参数
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        token_len_batch=None,
        labels=None,
        head_label=None,
        tail_label=None,
        span_label=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        tail_logits = None
        # For span extraction
        head_logits= None
        span_logits = None
        # 初始化变量，用于后续计算损失
        if not self.config.is_separate_ablation:
            '''
            如果不进行分离的消融实验:
                使用BERT模型进行前向传播。
                使用注意力分数计算head_logits, tail_logits和span_logits。
            否则：
                分别对文本和预测进行编码。
                使用额外的注意力层计算tail_logits。
            '''
            # Encoding the sentence and relations simultaneously, and using the inside Attention score
            # 同时对句子和关系进行编码，并使用内部注意力得分
            outputs = self.bert(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=False,
                            output_attentions_scores=True,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
            attentions_scores = outputs.attentions_scores[-1]
            BATCH_SIZE, ATT_HEADS, ATT_LEN, _ = attentions_scores.size()
            ATT_LAYERS = len(attentions_scores)
            if self.config.test_data_type == "unirel_span":
                head_logits = nn.Sigmoid()(
                        attentions_scores[:, :4, :, :].mean(1)
                    )
                tail_logits = nn.Sigmoid()(
                        attentions_scores[:, 4:8, :, :].mean(1)
                    )
                span_logits = nn.Sigmoid()(
                        attentions_scores[:, 8:, :, :].mean(1)
                    )
            else:
                tail_logits = nn.Sigmoid()(
                        attentions_scores[:, :, :, :].mean(1)
                    )
        else:
            # Encoding the sentence and relations in a separate manner, and add another attention layer
            TOKEN_LEN = token_len_batch[0]
            text_outputs = self.bert(
                            input_ids=input_ids[:, :TOKEN_LEN],
                            attention_mask=attention_mask[:, :TOKEN_LEN],
                            token_type_ids=token_type_ids[:, :TOKEN_LEN],
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=None,
                            output_attentions=False,
                            output_attentions_scores=False,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
            pred_outputs = self.bert(
                            input_ids=input_ids[:, TOKEN_LEN:],
                            attention_mask=attention_mask[:, TOKEN_LEN:],
                            token_type_ids=token_type_ids[:, TOKEN_LEN:],
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=None,
                            output_attentions=False,
                            output_attentions_scores=False,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

            last_hidden_state = torch.cat((text_outputs.last_hidden_state, pred_outputs.last_hidden_state), -2)
            key_layer = self.key_linear(last_hidden_state)
            value_layer = self.value_linear(last_hidden_state)
            tail_logits = nn.Sigmoid()(torch.matmul(key_layer, value_layer.permute(0, 2,1)))

        loss = None

        if tail_label is not None:
            tail_loss = nn.BCELoss()(tail_logits.float().reshape(-1),
                                    tail_label.reshape(-1).float())
            if loss is None:
                loss = tail_loss
            else:
                loss += tail_loss
        if head_label is not None:
            head_loss = nn.BCELoss()(head_logits.float().reshape(-1),
                                    head_label.reshape(-1).float())
            if loss is None:
                loss = head_loss
            else:
                loss += head_loss
        if span_label is not None:
            span_loss = nn.BCELoss()(span_logits.float().reshape(-1),
                                    span_label.reshape(-1).float())
            if loss is None:
                loss = span_loss
            else:
                loss += span_loss
        if tail_logits is not None:
            tail_predictions = tail_logits > self.config.threshold
        else:
            tail_predictions = None
        if head_logits is not None:
            head_predictions = head_logits > self.config.threshold
        else:
            head_predictions = None
        if span_logits is not None:
            span_predictions = span_logits > self.config.threshold
        else:
            span_predictions = None

        return UniRelOutput(
            loss=loss,
            head_preds=head_predictions,
            tail_preds=tail_predictions,
            span_preds=span_predictions,
        )
