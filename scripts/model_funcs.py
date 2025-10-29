import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


class Detector(nn.Module):
    def __init__(self, encoder, n_labels=2, lr=1e-5, is_roberta=False,
                 add_token_type_embeddings=False,
                 accumulation_step=1,
                 device="cuda", use_batch_norm=False, dropout=0.0,
                 add_class_weights=False, aggregation_mode="first"):
        super(Detector, self).__init__()
        self.encoder = encoder
        self.n_labels = n_labels
        self.is_roberta = is_roberta
        self.hidden_size = self.encoder.config.hidden_size
        ## добавлять ли эмбеддинги типа токенов
        self.add_token_type_embeddings = add_token_type_embeddings
        if self.add_token_type_embeddings:
            self.type_embeddings = nn.Embedding(3, self.hidden_size, padding_idx=2)
        self.accumulation_step = accumulation_step
        ## линейный слой для классификации
        self.proj_layer = nn.Linear(self.hidden_size, self.n_labels)
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.norm_layer = nn.BatchNorm1d(self.n_labels)
        self.activation = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self = self.to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        if add_class_weights:
            self.criterion = nn.NLLLoss(weight=torch.Tensor([1, 4]).to(self.device), reduction="mean", ignore_index=-100)
        else:
            self.criterion = nn.NLLLoss(reduction="mean", ignore_index=-100)
        self.aggregation_mode = aggregation_mode
        self.batch_idx = 0

    def train_on_batch(self, input_ids, labels, labels_mask, left_mask, right_mask, token_type_ids, **kwargs):
        """
        обучение на одном батче
        """
        self.train()
        if self.batch_idx % self.accumulation_step == 0:
            self.optimizer.zero_grad()
        batch_output = self._validate(input_ids, labels, labels_mask, left_mask, right_mask, token_type_ids, **kwargs)
        batch_output["loss"].backward()
        if self.batch_idx % self.accumulation_step == self.accumulation_step-1:
            self.optimizer.step()
        self.batch_idx += 1
        return batch_output

    def validate_on_batch(self, input_ids, labels, labels_mask, left_mask, right_mask, token_type_ids, 
                          **kwargs):
        """
        валидация на одном батче
        """
        self.eval()
        with torch.no_grad():
            return self._validate(input_ids, labels, labels_mask, left_mask, right_mask, token_type_ids, 
                                  **kwargs)

    def _validate(self, input_ids, labels, labels_mask, left_mask, right_mask, token_type_ids, 
                  **kwargs):
        """
        Применение модели к батчу, извлечение истинных меток и предсказанных log_probs
        Вычисление и запись значения функции потерь
        """
        input_ids, labels, left_mask, right_mask, token_type_ids, labels_mask =\
         input_ids.to(self.device), labels.to(self.device), left_mask.to(self.device), right_mask.to(self.device), token_type_ids.to(self.device),\
         labels_mask.to(self.device)
        batch_output = self(input_ids, labels_mask, left_mask, right_mask, token_type_ids, 
                            **kwargs)
        # labels = torch.where(labels_mask, labels, -100)
        preds = batch_output["log_probs"].permute(0, 2, 1)
        loss = self.criterion(preds, labels)
        batch_output["loss"] = loss
        return batch_output


    def forward(self, input_ids, labels_mask, left_mask, right_mask, token_type_ids, 
                **kwargs):
        """
        Пока работаем только с аггрегацией по первому сабтокену, 
        т.е. каждый пробел представляется усредненным значением между эмбеддингом левого соседа и эмб-гом правого соседа
        для каждого же реального слова левый и правый сосед -- эмбеддинг его первого сабтокена
        это отражено в left_mask и right mask, поэтому для вычисления векторных представлений токенов и пробелов в предложении
        - извлекаются эмбеддинги всех левых соседей
        - извлекаются эмбеддинги всех правых соседей
        - они усредняются
        - дополняются нулями до одинаковой длины
        далее к ним могут прибавляться эмбеддинги типа токенов,
        вычисляются логиты после применения линейного слоя,
        log_probs -- после функции активации
        применяется dropout
        возвращаются log_probs и метка класса с наибольшей вероятностью
        """
        if self.add_token_type_embeddings:
            token_type_embeddings = self.type_embeddings(token_type_ids)

        self.encoder = self.encoder.to(self.device)
        bert_output = self.encoder(input_ids=input_ids)["last_hidden_state"].to(self.device)
        assert bert_output.shape[-1] == self.hidden_size
        bert_output = self.dropout(bert_output)
        max_len = len(labels_mask[0])
        embeddings = []
        for i, sentence in enumerate(bert_output):
            if self.aggregation_mode == "first" or self.aggregation_mode == "last":
                pos_mask = labels_mask[i]
                left_indices = left_mask[i][pos_mask][None] # [0,1,1]->[[0,1,1]]
                right_indices = right_mask[i][pos_mask][None]
                left_values = sentence[left_indices]
                right_values = sentence[right_indices]
                sent_embedding = (left_values+right_values)/2 # W * 768, W - нужные слова
                assert sent_embedding.shape[1] == left_indices.shape[1]
            sent_embedding = torch.cat((sent_embedding[0],
                                        torch.zeros(size=(max_len-sent_embedding[0].shape[0], self.hidden_size)).to(self.device)))
            embeddings.append(sent_embedding)
        embeddings = torch.stack(embeddings)
        if self.add_token_type_embeddings:
            assert embeddings.shape == token_type_embeddings.shape
            embeddings = embeddings + token_type_embeddings
        logits = self.proj_layer(embeddings)
        if self.use_batch_norm:
            logits = self.norm_layer(logits)
        log_probs = self.activation(logits)
        _, answer = torch.max(log_probs, dim=-1)
        answer = torch.where(labels_mask, answer, -100)
        answer_dict = {"log_probs": log_probs, "labels": answer}
        return answer_dict
    
def update_metrics(metrics, batch_output, batch_labels, mask=None, to_print=False, copy_index=None, ignore_index=-100):
    """
    обновление метрик при обучении и валидации
    """
    n_batches = metrics["n_batches"]
    metrics["loss"] = (metrics["loss"] * n_batches + batch_output["loss"].item()) / (n_batches + 1)
    metrics["n_batches"] += 1
    assert mask is not None
    mask = mask.cpu().numpy().astype("int")
    batch_labels = batch_labels.cpu().numpy()
    are_equal = np.array(batch_output["labels"].cpu().numpy() == batch_labels).astype(int)
    curr_correct = (are_equal * mask).sum() # количество совпавших позиций
    curr_total = mask.sum() # количество совпавших реальных позиций
    metrics["correct"] += (are_equal * mask).sum()
    metrics["total"] += mask.sum()
    are_seq_correct = np.min(np.maximum(are_equal, 1-mask), axis=1)
    metrics["sent_correct"] += are_seq_correct.sum()
    metrics["sent_total"] += mask.shape[0]
    metrics["accuracy"] = metrics["correct"] / max(metrics["total"], 1)
    metrics["sent_accuracy"] = metrics["sent_correct"] / max(metrics["sent_total"], 1)
    if copy_index is not None:
        nonkeep_mask = (batch_labels != copy_index) * (batch_labels != ignore_index)
        metrics["nonkeep_correct"] += (are_equal * nonkeep_mask).sum()
        metrics["nonkeep_total"] += nonkeep_mask.sum()
        metrics["nonkeep_accuracy"] = metrics["nonkeep_correct"] / max(metrics["nonkeep_total"], 1)

def do_epoch(model, dataloader, mode="validate", epoch=1, copy_index=None, ignore_index=-100):
    """
    осуществление одной эпохи обучения или валидации, возвращение метрик
    """
    metrics = {"correct": 0, "total": 0, "sent_correct": 0, "sent_total": 0, "loss": 0.0, "n_batches": 0}
    if copy_index is not None:
        metrics.update({"nonkeep_correct": 0, "nonkeep_total": 0})
    func = model.train_on_batch if mode == "train" else model.validate_on_batch
    progress_bar = tqdm(dataloader, leave=True)
    progress_bar.set_description(f"{mode}, epoch={epoch}")
    for i, batch in enumerate(progress_bar):
        batch_output = func(**batch)
        update_metrics(metrics, batch_output, batch["labels"], mask=batch["labels_mask"], copy_index=copy_index, ignore_index=ignore_index)
        postfix = {"loss": round(metrics["loss"], 4), "acc": round(100 * metrics["accuracy"], 2),
                   "sent_acc": round(100 * metrics["sent_accuracy"], 2)}
        if copy_index is not None:
            postfix["nonkeep_acc"] = round(100 * metrics["nonkeep_accuracy"], 2)
        progress_bar.set_postfix(postfix)
    return metrics