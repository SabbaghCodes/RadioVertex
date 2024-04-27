<h1 class="code-line" data-line-start=0 data-line-end=1 ><a id="RadioVertex_0"></a>RadioVertex</h1>
<h2 class="code-line" data-line-start=1 data-line-end=2 ><a id="_VQA_System_for_Neuroradiology_Brain_Tumors_and_Lung_Cancer_Screening_Using_MRI_and_CT_Scans__1"></a><em>VQA System for Neuroradiology, Brain Tumors, and Lung Cancer Screening Using MRI and CT Scans.</em></h2>
<p class="has-line-data" data-line-start="2" data-line-end="3">RadioVertex is designed to assist radiologists by using AI to interpret MRI and CT scans, specifically for brain tumors and lung cancer. This AI-driven chatbot aims to provide real-time answers to clinical questions, improve diagnostic accuracy, and enhance the efficiency of medical imaging workflows. By integrating such technology, the goal is to reduce the heavy workload on radiologists, expedite the diagnostic process, and ultimately improve patient outcomes through earlier and more precise detection of critical health issues.</p>
<h2 class="code-line" data-line-start=3 data-line-end=4 ><a id="Demo_3"></a>Demo</h2>
<p class="has-line-data" data-line-start="4" data-line-end="5">Check out the examples <a href="https://github.com/RosolSharairh/RadioVertex/tree/main/examples">here</a>.</p>
<h2 class="code-line" data-line-start=5 data-line-end=6 ><a id="Installation_5"></a>Installation</h2>
<p class="has-line-data" data-line-start="6" data-line-end="8"><strong>1- Prepare the code and the environment</strong><br>
Git clone our repository, creating a python environment and activate it via the following command.</p>
<pre><code class="has-line-data" data-line-start="9" data-line-end="14" class="language-sh">git <span class="hljs-built_in">clone</span> 
<span class="hljs-built_in">cd</span> RadioVertex
conda env create <span class="hljs-operator">-f</span> environment.yml
conda activate radiovertex
</code></pre>
<p class="has-line-data" data-line-start="14" data-line-end="23"><strong>2- Prepare the pretrained LLM weights</strong><br>
RadioV etex is a finetuned model of MiniGPT-v2, which is based on Llama2 Chat 7B. Download the LLM weights from the following huggingface space via clone the repository using git-lfs. <a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main">Download</a><br>
Then, set the variable llama_model in the model config file to the LLM weight path.<br>
Set the LLM path [here](minigpt_v2.yaml file path) at Line 14. ##$**<br>
<strong>3- Prepare the pretrained model checkpoints</strong><br>
Download the pretrained model checkpoints.<br>
<a href="https://drive.google.com/file/d/1-uwRLa3xrD2h15UbdS8-gznx8UDH2zNY/view?usp=sharing">Download</a><br>
Set the path to the pretrained checkpoint in the evaluation config file <a href="https://github.com/RosolSharairh/RadioVertex/blob/main/eval_configs/minigptv2_eval.yaml">eval_configs/minigptv2_eval.yaml</a> at Line 8.<br>
<a href="https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view">Download</a> our pretrained model.</p>
<h2 class="code-line" data-line-start=24 data-line-end=25 ><a id="Launching_Demo_Locally_24"></a>Launching Demo Locally</h2>
<p class="has-line-data" data-line-start="25" data-line-end="26">Run:</p>
<pre><code class="has-line-data" data-line-start="27" data-line-end="29" class="language-sh">python demo_v2.py --cfg-path <span class="hljs-built_in">eval</span>_configs/minigptv2_eval.yaml  --gpu-id <span class="hljs-number">0</span>
</code></pre>
<p class="has-line-data" data-line-start="29" data-line-end="31">To save GPU memory, LLMs loads as 8 bit by default, with a beam search width of 1. This configuration requires about 23G GPU memory for 13B LLM and 11.5G GPU memory for 7B LLM. For more powerful GPUs, you can run the model in 16 bit by setting <code>low_resource</code> to <code>False</code> in the relevant config file:<br>
<a href="https://github.com/RosolSharairh/RadioVertex/blob/main/eval_configs/minigptv2_eval.yaml">minigptv2_eval.yaml</a></p>
<h2 class="code-line" data-line-start=32 data-line-end=33 ><a id="Training_32"></a>Training</h2>
<p class="has-line-data" data-line-start="33" data-line-end="35">You can download the data we used <a href="https://www.med-vqa.com/slake/">here</a>.<br>
In the train_configs/minigptv2_finetune.yaml, you need to set up the following paths:</p>
<p class="has-line-data" data-line-start="36" data-line-end="39">llama_model checkpoint path: “/path/to/llama_checkpoint”<br>
ckpt: “/path/to/pretrained_checkpoint”<br>
ckpt save path: “/path/to/save_checkpoint”</p>
<p class="has-line-data" data-line-start="40" data-line-end="42">For ckpt, you may load from our pretrained model checkpoints:<br>
<a href="https://drive.google.com/file/d/1-uwRLa3xrD2h15UbdS8-gznx8UDH2zNY/view?usp=sharing">Download</a></p>
<pre><code class="has-line-data" data-line-start="43" data-line-end="45" class="language-sh">torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigptv2_finetune.yaml
</code></pre>
<h2 class="code-line" data-line-start=45 data-line-end=46 ><a id="Evaluation_45"></a>Evaluation</h2>
<p class="has-line-data" data-line-start="46" data-line-end="47"><strong>Evaluation dataset structure</strong></p>
<pre><code class="has-line-data" data-line-start="48" data-line-end="54" class="language-sh">├── Slake
│   └── imgs
│   ├── train.json
│   ├── validate.json
│   ├── test.json
</code></pre>
<h2 class="code-line" data-line-start=54 data-line-end=55 ><a id="Requirements_54"></a>Requirements</h2>
<ul>
<li class="has-line-data" data-line-start="55" data-line-end="56">python 2.7</li>
<li class="has-line-data" data-line-start="56" data-line-end="57">scikit-image (visit <a href="https://scikit-image.org/docs/stable/user_guide/install.html">this page</a> for installation)</li>
<li class="has-line-data" data-line-start="57" data-line-end="59">matplotlib (visit <a href="https://matplotlib.org/stable/users/installing/index.html">this page</a> for installation)</li>
</ul>
<h2 class="code-line" data-line-start=59 data-line-end=60 ><a id="Acknowledgement_59"></a>Acknowledgement</h2>
<p class="has-line-data" data-line-start="60" data-line-end="61">Thanks to the creators of minigpt-v2 on their amazing model, check out their website <a href="https://minigpt-4.github.io/">here</a>.</p>
