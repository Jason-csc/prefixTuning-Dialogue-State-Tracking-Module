from openprompt.data_utils import InputExample
import json

def process_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    output_items = []
    gid = 0
    for _, d in data.items():
        logs = d["log"]
        dialogue = []
        for i in range(0,len(logs),2):
            log1 = logs[i]
            log2 = logs[i+1]
            dialogue.append(log1["processed_text"])
            dialogue.append(log2["processed_text"])
            assert log2["processed_metadata"] != {}
            metadata = log2["processed_metadata"]
            target_text = ""
            # for slot, value in metadata["restaurant"]["semi"].items():
            #     if len(value) > 0:
            #         target_text += f"{slot} {value[0]}, "
            for l1, layer1 in metadata.items():
                for l2, layer2 in layer1.items():
                    for slot, value in layer2.items():
                        if len(value) > 0:
                            if isinstance(value[0], dict):
                                for vv in value:
                                    for s,v in vv.items():
                                        if isinstance(v, str):
                                            target_text += f"{s} {v}, "
                                        elif isinstance(v, list):
                                            if len(v) == 0:
                                                print(value)
                                            else:
                                                target_text += f"{s} {v[0]}, "
                                        else:
                                            assert False, print(value)
                            elif isinstance(value[0], str):
                                target_text += f"{slot} {value[0]}, "
                            else:
                                assert False, print(value)
            target_text = "<pad> "+target_text[0:-2]
            input_example = InputExample(
                guid = gid,
                tgt_text = target_text,
                meta = {
                    "sentence": " ".join(dialogue)
                }
            )
            gid += 1
            output_items.append(input_example)
    return output_items






def evaluation(pred, label):
    em = 0
    print(pred)
    print([l[6:] for l in label])
    for p,l in zip(pred, label):
        if p == l[6:].lower():
            em += 1
    return em/len(pred)



def to_device(data, device):
    for key in ["input_ids", "attention_mask", "decoder_input_ids", "loss_ids"]:
        if key in data:
            data[key] = data[key].to(device)
    return data
            
            

