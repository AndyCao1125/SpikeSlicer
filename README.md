<br />
<p align="center">
  <h1 align="center">Spiking Neural Network as Adaptive Event Stream Slicer
(NeurIPS'24)</h1>
  </p>
  <p align="center">
    <a href='https://arxiv.org/pdf/2410.02249'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://proceedings.neurips.cc/paper_files/paper/2024/file/893a5db6100028ec814cfd99fe92c31b-Paper-Conference.pdf' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Proceeding-HTML-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Proceeding Supp'>
    </a>
  </p>
  <p align="center">
    <img src="images/spikeslicer.gif" alt="Logo" width="80%">
  </p>
</p>


## Installation
The environment is largely adopted from [pytracking](https://github.com/visionml/pytracking), if you have any issues, please carefully follow their official guidelines.
#### Install dependencies
* Create and activate a conda environment 
    ```bash
    conda create -n spikeslicer python=3.9 -y
    conda activate spikeslicer
    ```  
* Install packages
    ```bash
    pip install -r requirements.txt
    pip install tb-nightly
    conda install cpython  
    ```  

Note: if error occurs when installing tb-nightly, please use ```pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple```  after other packages are installed.



## Dataset
Please prepare [FE108](https://zhangjiqing.com/dataset/) on your device, and enter its path into `fe108_dir` of `ltr/admin/local.py` and `pytracking/evaluation/local.py`.

Replace the `txt` files in the dataset directory with our `txt` files splited by train_all (ANN pretraining), train (ANN training), val (SNN training), and test (Evaluation) sets:
```
cp dataset_txts/* your_fe108_directory/
```

Then please convert the raw event data of `aedat4` format into `npy` format that is convenient to load:
```
python event2frame.py
```
Note that this file has a default setting with group number of 5, which divides the event stream into 5 frames with same time intervals in voxel grid event representation.

## Train
Enter your directory of code to replace "SpikeSlicer" in `workspace_dir` of `ltr/admin/local.py`.

To pretrain a model, use command like this:
```
cd ltr
python run_training.py --train_module transt --train_name transt
```


To train an SNN along with ANN, first enter the path of the checkpoint saved in the pretraining stage to the corresponding file within `ltr/train_settings`. Then run the spikeslicer version, for example:
```
cd ltr
python run_training.py --train_module transt --train_name spikeslicer_transt
```
## Test

### Step 1: Generate the tracking results.
Activate the conda environment and run the script pytracking/run_tracker_fe108.py to run test code with different SNN types.
```bash
cd pytracking
python run_tracker_fe108.py --tracker_name transt --tracker_param transt50 --trained_model MODEL/WEIGHT/PATH --eval_save_dir SAVE/BBOX/PATH --event_representation frame --snn_type small
```  

### Step 2: Calculate the quantitative results.
Based on the tracking files from step 1, we can now compute the quantitative results, e.g., RSR, OP and RPR.
```bash
cd notebooks
python eval_results.py --tracker_name transt --tracker_param transt50 --eval_save_dir SAVE/BBOX/PATH --mode "all"
```  
We also provide an example experimental results reported in the article (test logs and weights of TransT can be found in [link](https://drive.google.com/drive/folders/1hfYP_f0rNFZdNJbYISL8B1XkP08WtuaM?usp=sharing)), and evaluate through:
```bash
cd notebooks
python eval_results.py --tracker_name transt --tracker_param transt50 --eval_save_dir "TransT_SpikeSlicer_Small_test_log.zip" --mode "all"
```  


Example Test Logs and Weights:
* Logs: [TransT_baseline_test_log.zip](https://drive.google.com/drive/folders/1hfYP_f0rNFZdNJbYISL8B1XkP08WtuaM?usp=sharing);  [TransT_SpikeSlicer_Small_test_log.zip](https://drive.google.com/drive/folders/1hfYP_f0rNFZdNJbYISL8B1XkP08WtuaM?usp=sharing);  [TransT_SpikeSlicer_Base_test_log.zip](https://drive.google.com/drive/folders/1hfYP_f0rNFZdNJbYISL8B1XkP08WtuaM?usp=sharing); 
* Weights: [TransT_baseline_pth](https://drive.google.com/drive/folders/1hfYP_f0rNFZdNJbYISL8B1XkP08WtuaM?usp=sharing);  [TransT_SpikeSlicer_Small_pth](https://drive.google.com/drive/folders/1hfYP_f0rNFZdNJbYISL8B1XkP08WtuaM?usp=sharing);  [TransT_SpikeSlicer_Base_pth](https://drive.google.com/drive/folders/1hfYP_f0rNFZdNJbYISL8B1XkP08WtuaM?usp=sharing) 


## Acknowledgement
Our code is generally built upon: [pytracking](https://github.com/visionml/pytracking), [Transformer Tracking](https://github.com/chenxin-dlut/TransT), and [spikingjelly](https://github.com/fangwei123456/spikingjelly). We thank all these authors for their nicely open sourced code and their great contributions to the community.

For any help or issues of this project, please contact [Jiahang Cao](jcao248@connect.hkust-gz.edu.cn) or [Mingyuan Sun](mingyuansun20@gmail.com).

## Citation

If you find our work useful, please consider citing:
```
@inproceedings{cao2024spiking,
  title={Spiking Neural Network as Adaptive Event Stream Slicer},
  author={Cao, Jiahang and Sun, Mingyuan and Wang, Ziqing and Cheng, Hao and Zhang, Qiang and Xu, Renjing and others},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
