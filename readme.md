# This is the code for the Paper "Gumbelsoft:Diversified Language Model Watermarking via the GumbelMax-trick"
see the link: https://arxiv.org/abs/2402.12948
## The Watermark Generator
The watermark generator is implemented in a hierarchical manner. 
see wm/generator.py
## The Watermark Detector
The watermark detector is also implemented in a hierarchical manner. 
see wm/detector.py

# To run the experiment:
1. conda create --name wm --file environment.yml
2. conda activate wm
3. bash run.sh