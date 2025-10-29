import torch

def collate_fn(sample, bert_tokenizer, dtype=torch.int64, device="cuda"):
    """
    Дополняет input_ids, mask паддингом до макс. длины input_ids
    Дополняет маски индексов, меток, типы токенов, метки до макс. длины меток
    """
    labels_lengths, tokens_lengths = [], []
    for elem in sample:
        for key in ["words", "labels", "labels_mask"]:
            if key in elem:
                labels_lengths.append(len(elem[key])) # elem["labels"] может не быть
                break
        tokens_lengths.append(len(elem["input_ids"]))
    answer = dict()
    answer["input_ids"] = torch.stack([
            torch.cat([
                elem["input_ids"].to(device),
                torch.ones(size=(max(tokens_lengths)-len(elem["input_ids"]),), dtype=dtype, device=device)*(bert_tokenizer.pad_token_id)
            ]) for elem in sample
        ]).to(device)
    for key in ["labels", "left_mask", "right_mask"]:
        if key in elem:
            answer[key] = torch.stack([
                torch.cat([
                    elem[key].to(device),
                    torch.ones(size=(max(labels_lengths)-len(elem[key]),), dtype=dtype, device=device)*(-100)
                ]) for elem in sample
            ]).to(device)
    if "token_type_ids" in sample[0]:
        answer["token_type_ids"] = torch.stack([
            torch.cat([
                elem["token_type_ids"].to(device),
                torch.ones(size=(max(labels_lengths)-len(elem["token_type_ids"]),), dtype=dtype, device=device)*(2)
            ]) for elem in sample
        ]).to(device)
    if "labels_mask" in elem:
        answer["labels_mask"] = torch.stack([
                    torch.cat([
                        elem["labels_mask"].to(device),
                        torch.zeros(size=(max(labels_lengths)-len(elem["labels_mask"]),), dtype=torch.bool, device=device)
                    ]) for elem in sample
                ]).to(device)
    return answer