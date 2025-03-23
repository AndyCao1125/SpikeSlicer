import ltr.admin.settings as ws_settings
from ltr.dataset.fe108 import FE108

if __name__ == '__main__':
    groupnum = 5
    settings = ws_settings.Settings()
    root = settings.env.fe108_dir
    subset_list = ["train", "test"]
    txt_suffix_list = ["_all.txt", ".txt"]
    for subset, txt_suffix in zip(subset_list, txt_suffix_list):
        dataset = FE108(root, subset=subset, groupnum=groupnum, txt_suffix=txt_suffix)
        for idx in range(len(dataset.sequence_list)):
            print("current sequence: ", dataset.sequence_list[idx], "percent: ", idx / len(dataset.sequence_list))
            dataset.raw_event_saver(idx,groupnum)
            dataset.image_saver(idx, groupnum)