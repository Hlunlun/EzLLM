# Conditional generation of detergent-compatible enzyme

This project runs from July 1, 2024, to August 30, 2024, in collaboration with the [Lab of Systems Biology and Network Biology](https://github.com/lsbnb), Institute of Information Science, Academia Sinica, aiming to generate more heat-resistant detergent-compatible enzymes by fine-tuning the EvoDiff model and analyzing the generated sequences from both structural and sequence perspectives.

We collected sequences associated with detergent-compatible enzymes by identifying relevant EC numbers and microbial species from the literature. These sequences were retrieved from UniProt, aligned using MSA, and split into training and test sets. The training set is used as input for EvoDiff, where we incorporate a disulfide bond reward mechanism during fine-tuning to calculate training error. The test set is used to generate sequences, which are then analyzed using EpHod and TemStaPro to predict optimal pH and temperature ranges, identifying high-temperature microbes to optimize the fine-tuning process for generating heat-resistant enzymes.



<br>

## Table of contents
- [Reference](#reference)
- [Installation](#installation)
- [Dataset](#dataset)
- [Fine-tuning](#fine-tuning)
- [Fine-tuned models](#fine-tuned-models)
- [Generation](#generation)
- [Analysis](#analysis)
- [Application](#application)

## Reference
| Model        | Details                                                                                                                                       | Reference                                                                                                                                                                    |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [**Clustal Omega**](https://www.ebi.ac.uk/jdispatcher/msa/clustalo) | Tool for multiple sequence alignment.| Sievers, Fabian and Desmond G. Higgins. “The Clustal Omega Multiple Alignment Package.” Methods in molecular biology 2331 (2021): 3-16.                                       |
| [**CD-Hit**](https://sites.google.com/view/cd-hit) | Tool for clustering and comparing protein/nucleotide sequences. | Li, Weizhong and Adam Godzik. “Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences.” Bioinformatics 22 13 (2006): 1658-9. Fu, Limin, Beifang Niu, Zhengwei Zhu, Sitao Wu and Weizhong Li. “CD-HIT: accelerated for clustering the next-generation sequencing data.” Bioinformatics 28 (2012): 3150-3152. |
| [**EvoDiff**](https://github.com/microsoft/evodiff)       | Generates diverse protein sequences and predicts their structures using OmegaFold. | Alamdari, Sarah, Nitya Thakkar, Rianne van den Berg, Alex X. Lu, Nicolo Fusi, Ava P. Amini and Kevin Kaichuang Yang. “Protein generation with evolutionary diffusion: sequence is all you need.” bioRxiv (2023): n. pag. | 
| [**OmegaFold**](https://github.com/HeliXonProtein/OmegaFold)     | Predicts protein structure from sequence.| Wu, Rui Min, Fan Ding, Rui Wang, Rui Shen, Xiwen Zhang, Shitong Luo, Chenpeng Su, Zuofan Wu, Qi Xie, Bonnie Berger, Jianzhu Ma and Jian Peng. “High-resolution de novo structure prediction from primary sequence.” bioRxiv (2022): n. pag. |
| [**InterProScan**](https://interproscan-docs.readthedocs.io/en/latest/)  | Scan motif for sequence. Released on 25 July 2024: InterProScan 5.69-101.| Jones, Philip, David Binns, Hsin-Yu Chang, Matthew Fraser, Weizhong Li, Craig McAnulla, Hamish McWilliam, John Maslen, Alex L. Mitchell, Gift Nuka, Sebastien Pesseat, Antony F. Quinn, Amaia Sangrador-Vegas, Maxim Scheremetjew, Siew-Yit Yong, Rodrigo Lopez and Sarah Hunter. “InterProScan 5: genome-scale protein function classification.” Bioinformatics 30 (2014): 1236-1240. |
| [**TemStaPro**](https://github.com/ievapudz/TemStaPro)|Predicts protein thermostability using embeddings generated by protein language models (pLMs).| Pudžiuvelytė, Ieva, Kliment Olechnovič, Egle Godliauskaite, Kristupas Sermokas, Tomas Urbaitis, Giedrius Gasiunas and Darius Kazlauskas. “TemStaPro: protein thermostability prediction using sequence representations from protein language models.” Bioinformatics 40 (2024): n. pag. |
| [**EpHod**](https://github.com/beckham-lab/EpHod) |Predicts enzyme optimum pH from sequence data. | Gado, Japheth E., Matthew Knotts, Ada Y. Shaw, Debora S. Marks, Nicholas Paul Gauthier, Christensen Carsten Sander and Gregg T. Beckham. “Deep learning prediction of enzyme optimum pH.” bioRxiv (2023): n. pag. |


## Installation
#### 1. Create projects folder
```
mkdir projects
cd projects
```

#### 2. Download Requirements
```
git clone https://github.com/Hlunlun/EzLLM.git
cd EzLLM
./setup.sh
```

#### 3. Download Required model
```
cd projects
# download TemStaPro
git clone https://github.com/ievapudz/TemStaPro.git
cd TemStaPro
make all
# download EpHod
git clone https://github.com/jafetgado/EpHod.git
```

#### 4. Install InterProScan
Refer to this [link](https://interproscan-docs.readthedocs.io/en/dev/UserDocs.html#obtaining-a-copy-of-interproscan) to install the latest version of InterProScan, or simply run the script below to install it:
```
./install_installproscan.sh
```

<br>

## Dataset
#### How It Was Compiled
1. Collection from Literature:\
Detergent-compatible enzymes were gathered from various scientific papers and compiled into a CSV file. This dataset categorizes enzyme data by types commonly found in detergents, including amylase, protease, mannanase, lipase, cellulase, and others. The dataset is available on Hugging Face: [lun610200/detergent-papers](https://huggingface.co/datasets/lun610200/detergent-papers). Note that not all papers specify the EC number of the enzymes. For instance, while protease is frequently mentioned across papers for its detergent functions, detailed EC numbers are often omitted.

2. Sequence Collection from [UniProt](https://www.uniprot.org/help/api_queries):\
To study the evolutionary and functional relationships of detergent enzymes, sequence data was collected from UniProt using EC numbers and corresponding organisms or species mentioned in the literature. Since enzymes with the same EC number catalyze similar reactions, this approach facilitates the analysis of evolutionary relationships and functional similarities within specific organisms. Sequence data in FASTA format was obtained from UniProt based on the EC numbers and the organisms identified in the papers. 

3. Data Collection Process:
    - By EC Number and Specific Organism:\
        Sequence data was gathered using the EC number and the specific organism.
    - By EC Number and Species:\
        The search was extended from specific organisms to species to increase the dataset size for conditional generation input.
4. Clustering to Identify Representative Sequences:\
    Clustering was performed using [CD-Hit](https://sites.google.com/view/cd-hit) with 90% sequence similarity to identify representative sequences. This simplifies sequence analysis and increases the likelihood of generating representative detergent-compatible sequences during conditional generation, as the representative sequences are positioned first in the MSA. Below is a comparison of the number of representative sequences versus the total number of sequences before and after clustering, using different datasets:
    - By EC Number and Specific Organism
        <div style="text-align: center;">
            <img src="img\ec_organism_cd.png" alt="EC + Organism before and after clustering" width="400" />
        </div>
    - By EC Number and Species
        <div style="text-align: center;">
            <img src="img\ec_species_cd.png" alt="EC + Species before and after clustering" width="400" />
        </div>

5. To perform Multiple Sequence Alignments (MSA)
    - Install Clustal Omega by [`pip`](https://pypi.org/project/clustalo/0.1.1/) or utilize the online version available at [EMBL-EBI](https://www.ebi.ac.uk/jdispatcher/msa/clustalo)
    - Perform MSA for:
        - Representative sequences with the same EC number.
        - Representative sequences with the same microorganism.

    These MSAs will be used as input for [`conditional_generation_msa.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/conditional_generation_msa.py) and [`train-msa.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/training/train-msa.py).

6. Motif Position Scanning:\
Motif positions were identified using [InterProScan](https://interproscan-docs.readthedocs.io/en/latest/index.html) with the Protein family (Pfam) database, as it contains the most extensive entry collection. Motif positions are conserved and functionally significant, and by fixing these positions during sequence generation, we aim to produce sequences that retain their original detergent functionality and are more evolutionarily aligned. 

7. Reference Values for pH and Temperature\
To find the optimal pH and temperature values for enzyme activity, use the EC number and microorganism as keys on [Brenda](https://www.brenda-enzymes.org/all_enzymes.php). These values can serve as reference points for pH and temperature ranges when generating sequences.


#### Organized Detergent-Compatible Enzyme Data:
|Data name|Detail|
|---|---|
|[lun610200/detergent-papers](https://huggingface.co/datasets/lun610200/detergent-papers)|Papers categorized by amylase, cellulase, lipase, mannanase, protease, and others. These papers include records of EC numbers, organisms, optimal pH, and optimal temperature for all detergent-compatible enzymes. These records were then used to collect sequence data from UniProt.|
|[lun610200/detergent-motif](https://huggingface.co/datasets/lun610200/detergent-motif)|Sequence data categorized by different EC numbers, recording motif positions, pH optimum, and temperature optimum.|
|[lun610200/detergent-enzyme](https://huggingface.co/datasets/lun610200/detergent-enzyme)|This dataset is used for fine-tuning EvoDiff. It is split into training and test datasets, with a total of 644 representative detergent-compatible sequences.|

#### Code
Collate the customized dataset using code in [`src/data_preprocess/`](https://github.com/Hlunlun/EzLLM/tree/master/src/data_preprocess):
1. [`collate_data.ipynb`](https://github.com/Hlunlun/EzLLM/blob/master/src/data_preprocess/collate_data.ipynb)\
This notebook handles the entire data processing workflow. It includes collecting and downloading sequence data, scanning motifs, clustering sequences, and organizing representative sequence data. The notebook contains comprehensive markdown annotations explaining each step of the process.
2. [`collect_data_webcrawl.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/data_preprocess/collect_data_webcrawl.py)\
This script defines functions for web scraping data from Brenda and UniProt. It allows for querying by EC number, organism, and species to search for reviewed or unreviewed sequence data from these sources. It also supports downloading FASTA files.
3. [`create_datset.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/data_preprocess/create_dataset.py)\
This script creates datasets from papers, motifs, and sequences for uploading to Hugging Face. You need to define a `secrets.ini` file to store your personal Hugging Face API token.


#### How to Use
1. Accessing the Dataset:\
The sequence data for the detergent-compatible enzyme dataset is available on the Hugging Face Hub: [lun610200/detergent-enzyme](https://huggingface.co/datasets/lun610200/detergent-enzyme)
2. Loading the Dataset:
    ```python
    from datasets import load_dataset
    ds = load_dataset("lun610200/detergent-enzyme")
    ```



## Fine-tuning
### Fine-tuning EvoDiff with sequence data in format `.fasta`
1. Run the Training Script
    ```
    cd src/training/
    python train.py --gpus 0 --random-seed 42 --lr 1e-4 --epochs 60 --train-batch-size 4 --warmup-steps 10 --save-steps 10 --reweight True
    ```
2. Prameter
    - `gpus`: Index of the GPU to use for training.
    - `random-seed`: Seed for random number generators to ensure reproducibility.
    - `lr`: Learning rate for the optimizer.
    - `train-batch-size`, `test-batch-size`, `validation-batch-size`: Batch sizes for loading data during training, evaluation, and testing.
    - `save-stpes`: The interval of steps after which the model is saved. If not specified, the program will save only the best model during training.
    - `reweight`: By default, this is set to True, meaning the loss will be reweighted using the Optimal Automatic Differentiation Method (OADM). If set to False, the model will use standard cross-entropy loss for weight updates.




### Fine-tuning EvoDiff with MSA data in format `.a3m`
1. Run the Training Script
    ```
    cd src/training/
    python train-msa.py --gpus 0 --random-seed 42 --lr 1e-4 --epochs 60 --train-batch-size 4 --warmup-steps 10 --save-steps 10 --reweight True
    ```
2. Parameter





<br>

## Fine-tuned models






## Conditional Generation
Refer to [evodiff](https://github.com/microsoft/evodiff) by Microsoft, which is a framework designed for evolutionary protein sequence generation and analysis. We can utilize this framework to generate detergent-compatible sequences by running the [`conditional_generation.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/conditional_generation.py) and [`conditional_generation_msa.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/conditional_generation_msa.py) scripts. 


#### Generate from dataset
run 
```
python src/main.py
```


#### `conditional_generation.py`

```
python conditional_generation_msa.py --cond-task scaffold --pdb A0A0A0PHP9 --num-seqs 1 --start-idx 0 --end-idx 10
```


#### `conditional_generation_msa.py`

```
python conditional_generation.py --cond-task scffold --pdb A0A0A0PHP9 --num-seqs 1 --start-idx 0 --end-idx 10 
```


#### Example
run [`example.ipynb`](https://github.com/Hlunlun/EzLLM/tree/master/example) to generate sequence by using finetuned model and [dataset](https://huggingface.co/datasets/lun610200/detergent-enzyme) already prepared to generate detergent-compatible sequence.
 
<br>


## Analysis
#### Evaluation
1. Structure Prediction and Evaluation:

    - Objective: \
        Determine the structural reliability of generated sequences using OmegaFold.
    - Metrics: \
        RMSD and pLDDT are used to evaluate the accuracy and confidence of the predicted structures.
    - Criteria for Success: \
        Sequences meeting the criteria of **RMSD < 1** and **pLDDT > 70** are considered for further analysis, ensuring that only high-quality structures are used.

2. Activity Prediction and Grouping:

    - Objective:\
        Assess the functional properties of the successful sequences.
    - Method: \
        Utilize **EpHod** for predicting optimal pH and **TemStaPro** for assessing thermal stability.
    - Analysis: \
        Categorize predictions by species to identify patterns and variations in pH and temperature preferences across different species.

3. Visualization and Interpretation:

    - Objective: \
        To visually identify which species' sequences exhibit higher heat tolerance.
    - Visualization Tools: \
        Use plots such as violin, histograms, box plots, or scatter plots to illustrate the ranges and distributions.
    - Expectation: \
        The goal is to achieve a higher temperature range, which would indicate a greater tolerance to heat, aligning with our project objectives for generating heat-resistant enzymes.

#### Structure Prediction and Evaluation
1. Execute [`src/main.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/main.py). This script contains a subprocess that runs [`src/rmsd_analysis.py`](https://github.com/Hlunlun/EzLLM/blob/master/src/rmsd_analysis.py) to calculate RMSD and pLDDT values for the generated sequences.
2. For more detailed information about the process and parameters used, please refer to [evodiff](https://github.com/microsoft/evodiff) 

#### Plot
1. Execute the script to plot pH and temperature distribution
    ```
    python src/pH_temp_analysis/pH_temp_analysis.py
    ```
2. The generated data will be collected, grouped by species, and plots will be saved to the default path `plot/`.

#### Comparison

||pH value|Temperature Range|
|---|---|---|
|Pretrained EvoDiff|<img src="img\pH_violin_species.jpg" alt="pH_violin_species.jpg" width="400" />|<img src="img\temp_violin_species.jpg" alt="temp_violin_species.jpg" width=800 />|
|Fine-tuned EvoDiff|<img src="img\pH_violin_species_ft.png" alt="pH_violin_species_ft.png" width="400" />|<img src="img\temp_violin_species_ft.png" alt="temp_violin_species_ft.png" width=800 />|



<br>

## Application
#### 1. Running the Application as a Web Service

This will initiate the web server and make the application accessible via your web browser.
```
python app/main.py
```
Expected Output
After executing the command, you should see output indicating that the Flask server is running, typically on http://127.0.0.1:5000/. You can access the application by entering this URL in your web browser.
<div style="text-align: center;">
    <img src="img\app.png" alt="alt text" width="400" />
</div>

#### 2. Front-end Development
The front-end of this application is built using JavaScript, which is responsible for controlling the user interface and handling user interactions.

Technologies Used:
- HTML/CSS for structure and styling
- JavaScript for dynamic functionality and UI control

File description
- `static/`: Contains css/, img/, and js/ related to front-end assets.
- `templates/`: Contains .html files corresponding to different topics such as home, results, and about.

#### 3. Back-end Development
The back-end of the application is built using Flask, a lightweight web framework for Python. This component is responsible for handling file uploads and generating sequences based on the uploaded files.

Technologies Used:
- Flask for creating the web server and managing requests
- Python for back-end logic and processing

File description
- `uploads/`: Files uploaded through the drop zone will be stored here.
- `main.py`: Flask application to process uploaded files.

<br>
By following the instructions above, you can set up and run the application locally. Feel free to explore and modify the code to suit your needs! Feel free to modify any sections or add additional details specific to your application!


## Future work