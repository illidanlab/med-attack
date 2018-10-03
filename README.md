# Identify Susceptible Locations in Medical Records via Adversarial Attacks on Deep Predictive Models

## Abstract
The surging availability of electronic medical records (EHR) leads to increased research interests in medical predictive modeling. Recently many deep learning based predicted models are also developed for EHR data and demonstrated impressive performance. However, a series of recent studies showed that these deep models are not safe: they suffer from certain vulnerabilities. In short, a well-trained deep network can be extremely sensitive to inputs with negligible changes. These inputs are referred to as adversarial examples. In the context of medical informatics, such attacks could alter the result of a high performance deep predictive model by slightly perturbing a patient’s medical records. Such instability not only reflects the weakness of deep architectures, more importantly, it offers a guide on detecting susceptible parts on the inputs. In this paper, we propose an efficient and effective framework that learns a time-preferential minimum attack targeting the LSTM model with EHR inputs, and we leverage this attack strategy to screen medical records of patients and identify susceptible events and measurements. The efficient screening procedure can assist decision makers to pay extra attentions to the locations that can cause severe consequence if not measured correctly. We conduct extensive empirical studies on a real-world urgent care cohort and demonstrate the effectiveness of the proposed screening.

The current version of the draft is available [here](https://arxiv.org/abs/1802.04822). 

## Compatibility
The code is compatible with python 2.7 and 3.5, and tensorflow version >1.4.

## Code Usage
- cw.py Algorithm for Generating Adversarial Examples
- cw_main.py Main functions
- Implementation was interagted from [EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples](https://github.com/ysharma1126/EAD_Attack) and [Adversarial Algorithms in TensorFlow](https://zenodo.org/record/1154272#.W7TYc5NKhTY). 
