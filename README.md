<a href="https://open-app.runwayml.com/?model=sree_harsha/PPLM" target="_blank"><img src="https://open-app.runwayml.com/gh-badge.svg" /></a>

# runway-transformers-PPLM : Runway port of Plug and Play Language Models, a Simple Approach to Controlled Text Generation

## Generation Script
Code was adapted from the excellent [run_generation](https://github.com/huggingface/transformers/blob/master/examples/run_generation.py) script and weights provided by  [huggingface/transformers](https://github.com/huggingface/transformers).


Authors: [Sumanth Dathathri](https://dathath.github.io/), [Andrea Madotto](https://andreamad8.github.io/), Janice Lan, Jane Hung, Eric Frank, [Piero Molino](https://w4nderlu.st/), [Jason Yosinski](http://yosinski.com/), and [Rosanne Liu](http://www.rosanneliu.com/)

Paper link: https://arxiv.org/abs/1912.02164

Blog link: https://eng.uber.com/pplm

Please check out the repo under uber-research for more information: https://github.com/uber-research/PPLM


### Tuning hyperparameters for bag-of-words control

1. Increase `--stepsize` to intensify topic control, and decrease its value to soften the control. `--stepsize 0` recovers the original uncontrolled GPT-2 model. 

2. If the language being generated is repetitive (For e.g. "science science experiment experiment"), there are several options to consider: </br>
	a) Reduce the `--stepsize` </br>
	b) Increase `--kl_scale` (the KL-loss coefficient) or decrease `--gm_scale` (the gm-scaling term) </br>
	c) Add `--grad-length xx` where xx is an (integer <= length, e.g. `--grad-length 30`).</br>

### Tuning hyperparameters for discriminator control

1. Increase `--stepsize` to intensify topic control, and decrease its value to soften the control. `--stepsize 0` recovers the original uncontrolled GPT-2 model. 

2. Use `--class_label 3` for negative, and `--class_label 2` for positive


