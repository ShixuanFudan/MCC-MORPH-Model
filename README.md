              
                                   MORPH-Model Framework
                  
                        ███╗   ███╗ ██████╗ ██████╗ ██████╗ ██╗  ██╗
                        ████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ██║
                        ██╔████╔██║██║   ██║██████╔╝██████╔╝███████║
                        ██║╚██╔╝██║██║   ██║██╔══██╗██╔═══╝ ██╔══██║
                        ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║
                        ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
                  
                             [AI] --> ( proteome reversal ) --> [ Score ]
                       
                        MORPH: AI-Driven Molecular Morphology & DPI Predictor
                        [■] TASK: Drug-Protein Interaction [■] VER: 1.1.0
                        [■] AUTH: Shixuan.Z & ZhenQiu.L    [■] SYS: Mac M or Win 10

    
# 🧬 MORPH: An AI-Driven Framework for Proteome Reversal Score (PPMS) & Drug-Protein Interaction Prediction

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![R Version](https://img.shields.io/badge/R-4.3%2B-green.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-darkgray.svg)](LICENSE)

**MORPH (formerly)** is an advanced, multi-target drug discovery (MTDD) framework designed to decipher complex metabolic comorbidity clusters (MCCs) and prioritize synergistic drug combinations via deep learning and network pharmacology. 

## 🛠️ Environmental Prerequisites

MORPH requires a hybrid Python and R execution environment. Ensure both runtimes meet the following specifications:

## Python Environment (v3.8+)

Install the required scientific and deep learning dependencies via `pip`:
```bash
pip install torch torch_geometric pandas numpy scikit-learn networkx tqdm matplotlib seaborn lightgbm joblib rdkit
```
##⚠️💡 Prepare documents

Please download the Rawdata-Model/ folder: https://zenodo.org/uploads/21309729 

## 📂 Repository & Data Architecture

```text
.
├── MORPH.py                  # DPI execution engine (Deep Learning & Gradient Boosting inference)
├── PPMS.py                   # Network propagation & proteomic synergy scoring framework
├── Rawdata/                  # training files
│   └── drug_fp_features_final_new/              # molecular fingerprint
│   ├── protein_esm2_features_final_new/         # Protein ESM2 fingerprint
│   └── PPI.v12.20260122.csv/ # PPI network diagram
├── enhanced_features/        # GNN fusion features
│   └── enhanced_feature_extractor_model.pth/        
│   ├── enhanced_features.pkl/       
│   └── graph_data.pkl/       
```

## 💻 Model training
### Step 1: GNN feature fusion

```bash
python 1.GNNmulti.py 
```

###Step 2: Classifier

Propagate localized perturbation vectors across the target disease interactome spectrum to yield unified synergy scores:

```bash
python 2.Classifier.py
```

## Acknowledgments

We extend our deepest gratitude to the clinical pharmacists and collaborators who made the stringent double-blind validation possible:

Shixuan.Z  (Algorithm Implementation)
ZhenQiu.L (Algorithm Implementation)
Jingru.G (Clinical Pharmacist, Independent Blinded Rater)
Shun.S (Clinical Pharmacist, Independent Blinded Rater)

## ✉️ Contact & Citatio
Author: Shixuan Zhang
Institution: Fudan University, Shanghai, China
Email: sxzhang21@m.fudan.edu.cn



