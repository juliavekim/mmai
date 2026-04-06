# MMAI 2026 - Julia Kim 

Welcome to my site for Modeling Multimodal AI 2026! 💜 
This repo contains my homework assignments and random thoughts throughout the class.

## Bio
<img src="./imgs/profile.png" style="width:200px;">
Hi, I'm Julia. I'm a first-year PhD student at MIT's Operations Research Centre, working at the intersection of machine learning, interpretable AI, and educational technology. Prior to MIT, I read physics and mathematics at the University of Toronto and spent a year reading graduate mathematics at Cambridge. I'm interested in building systems that are both theoretically grounded and genuinely useful in practice.

## Final Project
For my final project, I built [**EgoBlind-RA**](https://github.com/juliavekim/EgoBlind-RA/tree/main) with Xander Backus, a multimodal AI assistant designed to support visually impaired users by answering natural language queries about their physical environment using egocentric video. The system fine-tunes Kimi-VL-A3B-Instruct on first-person footage, enabling responses to real-time queries such as identifying objects, reading text, or describing spatial context. Training was conducted using LLaMA-Factory on an A100 GPU, with checkpoints saved to Google Drive for persistence across sessions. The project involved resolving a critical data pipeline issue in which visual frames were excluded from SFT training, and retraining the model with proper multimodal inputs to achieve meaningful performance gains.

## Homework
- [Homework 1 - Datasets and Preprocessing Data](./homework/homework-1/)
- [Homework 2 - Einsum + Tensors, Multimodal fusion, Contrastive Learning](./homework/homework-2/)
- [Homework 3 - Multimodal LLMsk](./homework/homework-3/)

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
