<div align=center>
<img src="https://github.com/PKU-YuanGroup/MagicTime/blob/main/__assets__/magictime_logo.png?raw=true" width="150px">
</div>
<h2 align="center"> <a href="https://arxiv.org/abs/2404.05014">MagicTime: Time-lapse Video Generation Models 

<a href="https://arxiv.org/abs/2404.05014">as Metamorphic Simulators</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">


[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/BestWishYsh/MagicTime?logs=build)
[![Replicate demo and cloud API](https://replicate.com/camenduru/magictime/badge)](https://replicate.com/camenduru/magictime)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/MagicTime-jupyter/blob/main/MagicTime_jupyter.ipynb)
[![hf_space](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2404.05014)
[![arXiv](https://img.shields.io/badge/Arxiv-2404.05014-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2404.05014) 
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://pku-yuangroup.github.io/MagicTime/) 
[![Dataset](https://img.shields.io/badge/Dataset-<HuggingFace>-green)](https://huggingface.co/datasets/BestWishYsh/ChronoMagic)
[![zhihu](https://img.shields.io/badge/-Twitter@AK%20-black?logo=twitter&logoColor=1D9BF0)](https://twitter.com/_akhaliq/status/1777538468043792473)
[![zhihu](https://img.shields.io/badge/-Twitter@Jinfa%20Huang%20-black?logo=twitter&logoColor=1D9BF0)](https://twitter.com/vhjf36495872/status/1777525817087553827?s=61&t=r2HzCsU2AnJKbR8yKSprKw)
[![DOI](https://zenodo.org/badge/783303222.svg)](https://zenodo.org/doi/10.5281/zenodo.10960665)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/MagicTime/blob/main/LICENSE) 
[![github](https://img.shields.io/github/stars/PKU-YuanGroup/MagicTime.svg?style=social)](https://github.com/PKU-YuanGroup/MagicTime)

</h5>

<div align="center">
This repository is the official implementation of MagicTime, a metamorphic video generation pipeline based on the given prompts. The main idea is to enhance the capacity of video generation models to accurately depict the real world through our proposed methods and dataset.
</div>


<br>
<details open><summary>💡 We also have other video generation projects that may interest you ✨. </summary><p>
<!--  may -->


> [**Open-Sora Plan: Open-Source Large Video Generation Model**](https://arxiv.org/abs/2412.00131) <br>
> Bin Lin, Yunyang Ge and Xinhua Cheng etc. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social)](https://github.com/PKU-YuanGroup/Open-Sora-Plan) [![arXiv](https://img.shields.io/badge/Arxiv-2412.00131-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.00131) <br>
>
> [**ConsisID: Identity-Preserving Text-to-Video Generation by Frequency Decomposition**](https://arxiv.org/abs/2411.17440) <br>
> Shenghai Yuan, Jinfa Huang and Xianyi He etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/ConsisID/)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social)](https://github.com/PKU-YuanGroup/ConsisID/) [![arXiv](https://img.shields.io/badge/Arxiv-2411.17440-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.17440) <br>
>
> [**ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation**](https://arxiv.org/abs/2406.18522) <br>
> Shenghai Yuan, Jinfa Huang and Yongqi Xu etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/ChronoMagic-Bench/)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/ChronoMagic-Bench.svg?style=social)](https://github.com/PKU-YuanGroup/ChronoMagic-Bench/) [![arXiv](https://img.shields.io/badge/Arxiv-2406.18522-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.18522) <br>
> </p></details>

## 📣 News
* ⏳⏳⏳ Training a stronger model with the support of [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).
* ⏳⏳⏳ Release the training code of MagicTime.
* `[2025.04.08]`  🔥 We have updated our technical report. Please click [here](https://arxiv.org/abs/2404.05014) to view it.
* `[2025.03.28]`  🔥 MagicTime has been accepted by **TPAMI**, and we will update arXiv with more details soon, keep tuned!
* `[2024.07.29]`  We add *batch inference* to [inference_magictime.py](https://github.com/PKU-YuanGroup/MagicTime/blob/main/inference_magictime.py) for easier usage.
* `[2024.06.27]`  Excited to share our latest [ChronoMagic-Bench](https://github.com/PKU-YuanGroup/ChronoMagic-Bench), a benchmark for metamorphic evaluation of text-to-time-lapse video generation, and is fully open source! Please check out the [paper](https://arxiv.org/abs/2406.18522).
* `[2024.05.27]`  Excited to share our latest Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out the [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.1.0.md).
* `[2024.04.14]`  Thanks [@camenduru](https://twitter.com/camenduru) and [@ModelsLab](https://modelslab.com/) for providing [Jupyter Notebook](https://github.com/camenduru/MagicTime-jupyter) and [Replicate Demo](https://replicate.com/camenduru/magictime).
* `[2024.04.13]`  🔥 We have compressed the size of repo with less than 1.0 MB, so that everyone can clone easier and faster. You can click [here](https://github.com/PKU-YuanGroup/MagicTime/archive/refs/heads/main.zip) to download, or use `git clone --depth=1` command to obtain this repo.
* `[2024.04.12]`  Thanks [@Kijai](https://github.com/kijai) and [@Baobao Wang](https://www.bilibili.com/video/BV1wx421U7Gn/?spm_id_from=333.1007.top_right_bar_window_history.content.click) for providing ComfyUI Extension [ComfyUI-MagicTimeWrapper](https://github.com/kijai/ComfyUI-MagicTimeWrapper). If you find related work, please let us know. 
* `[2024.04.11]`  🔥 We release the Hugging Face Space of MagicTime, you can click [here](https://huggingface.co/spaces/BestWishYsh/MagicTime?logs=build) to have a try.
* `[2024.04.10]`  🔥 We release the inference code and model weight of MagicTime.
* `[2024.04.09]`  🔥 We release the arXiv paper for MagicTime, and you can click [here](https://arxiv.org/abs/2404.05014) to see more details.
* `[2024.04.08]`  🔥 We release the subset of ChronoMagic dataset used to train MagicTime. The dataset includes 2,265 metamorphic video-text pairs and can be downloaded at [HuggingFace Dataset](https://huggingface.co/datasets/BestWishYsh/ChronoMagic) or [Google Drive](https://drive.google.com/drive/folders/1WsomdkmSp3ql3ImcNsmzFuSQ9Qukuyr8?usp=sharing).
* `[2024.04.08]`  🔥 **All codes & datasets** are coming soon! Stay tuned 👀!

## 😮 Highlights

MagicTime shows excellent performance in **metamorphic video generation**.

### Related Resources
* [ChronoMagic](https://huggingface.co/datasets/BestWishYsh/ChronoMagic): including 2265 time-lapse video-text pairs. (captioned by GPT-4V)
* [ChronoMagic-Bench](https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench/tree/main): including 1649 time-lapse video-text pairs. (captioned by GPT-4o)
* [ChronoMagic-Bench-150](https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench/tree/main): including 150 time-lapse video-text pairs. (captioned by GPT-4o)
* [ChronoMagic-Pro](https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Pro): including 460K time-lapse video-text pairs. (captioned by ShareGPT4Video)
* [ChronoMagic-ProH](https://huggingface.co/datasets/BestWishYsh/ChronoMagic-ProH): including 150K time-lapse video-text pairs. (captioned by ShareGPT4Video)

### Metamorphic Videos vs. General Videos 

Compared to general videos, metamorphic videos contain physical knowledge, long persistence, and strong variation, making them difficult to generate. We show compressed .gif on github, which loses some quality. The general videos are generated by the [Animatediff](https://github.com/guoyww/AnimateDiff) and **MagicTime**.

<table>
  <tr>
    <td colspan="1"><center>Type</center></td>  
    <td colspan="1"><center>"Bean sprouts grow and mature from seeds"</center></td>
    <td colspan="1"><center>"[...] construction in a Minecraft virtual environment"</center></td>
    <td colspan="1"><center>"Cupcakes baking in an oven [...]"</center></td>
    <td colspan="1"><center>"[...] transitioning from a tightly closed bud to a fully bloomed state [...]"</center></td>
  </tr>
  <tr>
    <td>General Videos</td>  
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_0_0.gif?raw=true" alt="MakeLongVideo"></td>
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_0_1.gif?raw=true" alt="MakeLongVideo"></td>
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_0_2.gif?raw=true" alt="MakeLongVideo"></td>
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_0_3.gif?raw=true" alt="MakeLongVideo"></td>
  </tr>
  <tr>
    <td>Metamorphic Videos</td>  
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_1_0.gif?raw=true" alt="ModelScopeT2V"></td>
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_1_1.gif?raw=true" alt="ModelScopeT2V"></td>
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_1_2.gif?raw=true" alt="ModelScopeT2V"></td>
    <td><img src="https://github.com/PKU-YuanGroup/MagicTime/blob/ProjectPage/static/videos/C_1_3.gif?raw=true" alt="ModelScopeT2V"></td>
  </tr>
</table>

### Gallery

We showcase some metamorphic videos generated by **MagicTime**, [MakeLongVideo](https://github.com/xuduo35/MakeLongVideo), [ModelScopeT2V](https://github.com/modelscope), [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter?tab=readme-ov-file), [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w), [LaVie](https://github.com/Vchitect/LaVie), [T2V-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero), [Latte](https://github.com/Vchitect/Latte) and [Animatediff](https://github.com/guoyww/AnimateDiff) below.

<table>
  <tr>
    <td colspan="1"><center>Method</center></td>  
    <td colspan="1"><center>"cherry blossoms transitioning [...]"</center></td>
    <td colspan="1"><center>"dough balls baking process [...]"</center></td>
    <td colspan="1"><center>"an ice cube is melting [...]"</center></td>
    <td colspan="1"><center>"a simple modern house's construction [...]"</center></td>
  </tr>
  <tr>
    <td>MakeLongVideo</td>  
    <td><img src="https://github.com/user-attachments/assets/32ebcab3-8103-47e0-91cf-03325099088b" alt="MakeLongVideo"></td>
    <td><img src="https://github.com/user-attachments/assets/1b154e96-402a-4833-b4f4-c05b42fb07f4" alt="MakeLongVideo"></td>
    <td><img src="https://github.com/user-attachments/assets/cfa1468d-2703-431a-9733-49b260f00bf9" alt="MakeLongVideo"></td>
    <td><img src="https://github.com/user-attachments/assets/87f8c37c-3073-4501-ae8c-7dc0c36d519e" alt="MakeLongVideo"></td>
  </tr>
  <tr>
    <td>ModelScopeT2V</td>  
    <td><img src="https://github.com/user-attachments/assets/fadde609-2416-4e3a-8616-1a7b8b67df16" alt="ModelScopeT2V"></td>
    <td><img src="https://github.com/user-attachments/assets/e62259b1-aa16-4031-9898-abe00023dfe8" alt="ModelScopeT2V"></td>
    <td><img src="https://github.com/user-attachments/assets/f35cf24f-1c05-46ad-bae2-c6118ed721b8" alt="ModelScopeT2V"></td>
    <td><img src="https://github.com/user-attachments/assets/972584d5-1e33-41e5-85a8-a51dfb6f215b" alt="ModelScopeT2V"></td>
  </tr>
  <tr>
    <td>VideoCrafter</td>  
    <td><img src="https://github.com/user-attachments/assets/fdb200da-98c5-4698-b7b0-b856962f9fc6" alt="VideoCrafter"></td>
    <td><img src="https://github.com/user-attachments/assets/37885eb8-3f52-49fb-bfba-b62fe5350775" alt="VideoCrafter"></td>
    <td><img src="https://github.com/user-attachments/assets/a24f79df-d204-4477-b519-bad53a755b23" alt="VideoCrafter"></td>
    <td><img src="https://github.com/user-attachments/assets/5651d952-dd17-4293-8cb5-6e652053f759" alt="VideoCrafter"></td>
  </tr>
  <tr>
    <td>ZeroScope</td>  
    <td><img src="https://github.com/user-attachments/assets/b009efb6-7e45-4345-a29e-12c77c0f1df2" alt="ZeroScope"></td>
    <td><img src="https://github.com/user-attachments/assets/4afbd1d5-66ec-4f3d-b455-8068c51faff1" alt="ZeroScope"></td>
    <td><img src="https://github.com/user-attachments/assets/8df1044b-80ba-4ae6-89da-7a8b8018b89d" alt="ZeroScope"></td>
    <td><img src="https://github.com/user-attachments/assets/e024a24e-0d5c-4986-b400-3933fcdc9534" alt="ZeroScope"></td>
  </tr>
  <tr>
    <td>LaVie</td>  
    <td><img src="https://github.com/user-attachments/assets/12b9144a-1ce9-472d-98d0-86c3e1a03972" alt="LaVie"></td>
    <td><img src="https://github.com/user-attachments/assets/0fee9321-2278-45f8-a6eb-0ed3f4e11d15" alt="LaVie"></td>
    <td><img src="https://github.com/user-attachments/assets/1f4ee694-39f0-4190-8b03-d68260c1ee7e" alt="LaVie"></td>
    <td><img src="https://github.com/user-attachments/assets/3bbb43cc-5e9a-4ee1-bf51-72ee0fa67b12" alt="LaVie"></td>
  </tr>
  <tr>
    <td>T2V-Zero</td> 
    <td><img src="https://github.com/user-attachments/assets/d4e618f3-7a2f-4f17-b12d-d390013ea39c" alt="T2V-Zero"></td>
    <td><img src="https://github.com/user-attachments/assets/a02e13c3-3cef-4086-b5e9-f06218de2ef4" alt="T2V-Zero"></td>
    <td><img src="https://github.com/user-attachments/assets/55d81342-b51a-4137-a27a-878fc4d4d441" alt="T2V-Zero"></td>
    <td><img src="https://github.com/user-attachments/assets/9b6b50ba-a300-4eb6-8ce8-6d3c102acd2f" alt="T2V-Zero"></td>
  </tr>
  <tr>
    <td>Latte</td>
    <td><img src="https://github.com/user-attachments/assets/77d39bba-0486-4294-bc56-e6e3f9c1ddca" alt="Latte"></td>
    <td><img src="https://github.com/user-attachments/assets/694c6a23-af51-43c1-81eb-341964d9aa26" alt="Latte"></td>
    <td><img src="https://github.com/user-attachments/assets/dcddf819-7bd3-400e-901f-b5e4c79489a0" alt="Latte"></td>
    <td><img src="https://github.com/user-attachments/assets/069ed0a7-28db-4a34-8033-f38f3479f18c" alt="Latte"></td>
  </tr>
  <tr>
    <td>Animatediff</td>
    <td><img src="https://github.com/user-attachments/assets/13193a84-f4a4-4ba5-b49e-177c32fc410e" alt="Animatediff"></td>
    <td><img src="https://github.com/user-attachments/assets/4c961e3e-3556-497e-8695-47e1a02c0f29" alt="Animatediff"></td>
    <td><img src="https://github.com/user-attachments/assets/0e5cbd30-4e05-466e-aee7-58244ea244db" alt="Animatediff"></td>
    <td><img src="https://github.com/user-attachments/assets/2c2afd44-63e4-4ef9-9550-da89a1ebf236" alt="Animatediff"></td>
  </tr>
  <tr>
    <td>Ours</td>  
    <td><img src="https://github.com/user-attachments/assets/f6c5594a-5b49-4175-b4cf-60faa5f20db1" alt="Ours"></td>
    <td><img src="https://github.com/user-attachments/assets/4ad90c93-ba1e-4d75-8b09-294e5c291fe9" alt="Ours"></td>
    <td><img src="https://github.com/user-attachments/assets/bc6941a5-6b84-4a3b-9248-238cfade5dd4" alt="Ours"></td>
    <td><img src="https://github.com/user-attachments/assets/e7776d31-1412-40cd-bcd5-f3094310bbd8" alt="Ours"></td>
  </tr>
</table>


We show more metamorphic videos generated by **MagicTime** with the help of [Realistic](https://civitai.com/models/4201/realistic-vision-v20), [ToonYou](https://civitai.com/models/30240/toonyou) and [RcnzCartoon](https://civitai.com/models/66347/rcnz-cartoon-3d).

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8124cc26-0f27-40f3-89b2-f493400adc41" alt="Realistic"></td>
    <td><img src="https://github.com/user-attachments/assets/6ee87c95-a0cc-4948-af3a-fb70689f731a" alt="Realistic"></td>
    <td><img src="https://github.com/user-attachments/assets/1a89d55e-8a5e-45ed-a4a6-766cb16d8334" alt="Realistic"></td>
  </tr>
  <tr>
    <td colspan="1"><center>"[...] bean sprouts grow and mature from seeds"</center></td>
    <td colspan="1"><center>"dough [...] swells and browns in the oven [...]"</center></td>
    <td colspan="1"><center>"the construction [...] in Minecraft [...]"</center></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9523bc8a-b41a-477d-ba84-b7423d7fcdd8" alt="RcnzCartoon"></td>
    <td><img src="https://github.com/user-attachments/assets/68651214-3f94-4f9d-b20f-1575da7552f9" alt="RcnzCartoon"></td>
    <td><img src="https://github.com/user-attachments/assets/6555d8d5-dd03-4bf9-b4b1-113cc52d95b5" alt="RcnzCartoon"></td>
  </tr>
  <tr>
    <td colspan="1"><center>"a bud transforms into a yellow flower"</center></td>
    <td colspan="1"><center>"time-lapse of a plant germinating [...]"</center></td>
    <td colspan="1"><center>"[...] a modern house being constructed in Minecraft [...]"</center></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/62168dc6-e3db-4c2e-833f-e878d7e55227" alt="ToonYou"></td>
    <td><img src="https://github.com/user-attachments/assets/f41153f3-f9fd-4836-9115-2b6a6caa741f" alt="ToonYou"></td>
    <td><img src="https://github.com/user-attachments/assets/09b74420-e4d2-48f8-81ba-ecc0ef02af57" alt="ToonYou"></td>
  </tr>
  <tr>
    <td colspan="1"><center>"an ice cube is melting"</center></td>
    <td colspan="1"><center>"bean plant sprouts grow and mature from the soil"</center></td>
    <td colspan="1"><center>"time-lapse of delicate pink plum blossoms [...]"</center></td>
  </tr>
</table>

Prompts are trimmed for display, see [here](https://github.com/PKU-YuanGroup/MagicTime/blob/main/__assets__/promtp_unet.txt) for full prompts.
### Integrate into DiT-based Architecture

The mission of this project is to help reproduce Sora and provide high-quality video-text data and data annotation pipelines, to support [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) or other DiT-based T2V models. To this end, we take an initial step to integrate our MagicTime scheme into the DiT-based Framework. Specifically, our method supports the Open-Sora-Plan v1.0.0 for fine-tuning. We first scale up with additional metamorphic landscape time-lapse videos in the same annotation framework to get the ChronoMagic-Landscape dataset. Then, we fine-tune the Open-Sora-Plan v1.0.0 with the ChronoMagic-Landscape dataset to get the MagicTime-DiT model. The results are as follows (**257×512×512 (10s)**):

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/ed66b4c7-4352-456a-bee4-267090857e3e" autoplay loop muted></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/f2498e6b-4556-4909-b582-954f36a71281" autoplay loop muted></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/17f8d675-6315-4c48-9933-8b3a7f4e72d2" autoplay loop muted></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/ec29677d-2ed7-411b-8a41-e744614916bf" autoplay loop muted></video>
    </td>
  </tr>
  <tr>
    <td><center>"Time-lapse of a coastal landscape [...]"</center></td>
    <td><center>"Display the serene beauty of twilight [...]"</center></td>
    <td><center>"Sunrise Splendor: Capture the breathtaking moment [...]"</center></td>
    <td><center>"Nightfall Elegance: Embrace the tranquil beauty [...]"</center></td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/bc961c34-7a3d-416e-be3b-80e09a5145bc" autoplay loop muted></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/4feba55e-3259-4a1c-821b-51eb905e289b" autoplay loop muted></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/7a69f551-ac22-4d69-a5dd-e03819f295bb" autoplay loop muted></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/b92ae91a-5f7b-4f1b-9a36-82b95478c388" autoplay loop muted></video>
    </td>
  </tr>
  <tr>
    <td colspan="1"><center>"The sun descending below the horizon [...]"</center></td>
    <td colspan="1"><center>"[...] daylight fades into the embrace of the night [...]"</center></td>
    <td colspan="1"><center>"Time-lapse of the dynamic formations of clouds [...]"</center></td>
    <td colspan="1"><center>"Capture the dynamic formations of clouds [...]"</center></td>
  </tr>
</table>

Prompts are trimmed for display, see [here](https://github.com/PKU-YuanGroup/MagicTime/blob/main/__assets__/promtp_opensora.txt) for full prompts.

## 🤗 Demo

### Gradio Web UI

Highly recommend trying out our web demo by the following command, which incorporates all features currently supported by MagicTime. We also provide [online demo](https://huggingface.co/spaces/BestWishYsh/MagicTime?logs=build) in Hugging Face Spaces.

```bash
python app.py
```

### CLI Inference

```bash
# For Realistic
python inference_magictime.py --config sample_configs/RealisticVision.yaml --human

# or you can directly run the .sh
sh inference_cli.sh
```

warning: It is worth noting that even if we use the same seed and prompt but we change a machine, the results will be different.

## ⚙️ Requirements and Installation

We recommend the requirements as follows.

### Environment

```bash
git clone --depth=1 https://github.com/PKU-YuanGroup/MagicTime.git
cd MagicTime
conda create -n magictime python=3.10.13
conda activate magictime
pip install -r requirements.txt
```

### Download MagicTime

The weights are available at [🤗HuggingFace](https://huggingface.co/BestWishYsh/MagicTime/tree/main) and [🟣WiseModel](https://wisemodel.cn/models/SHYuanBest/MagicTime/file), or you can download it with the following commands.

```bash
# way 1
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model \
BestWishYsh/MagicTime \
--local-dir ckpts

# way 2
git lfs install
git clone https://www.wisemodel.cn/SHYuanBest/MagicTime.git
```

Once ready, the weights will be organized in this format:

```
📦 ckpts/
├── 📂 Base_Model/
│   ├── 📂 motion_module/
│   ├── 📂 stable-diffusion-v1-5/
├── 📂 DreamBooth/
├── 📂 Magic_Weights/
│   ├── 📂 magic_adapter_s/
│   ├── 📂 magic_adapter_t/
│   ├── 📂 magic_text_encoder/
```

## 🗝️ Training & Inference

The training code is coming soon! 

For inference, some examples are shown below:

```bash
# For Realistic
python inference_magictime.py --config sample_configs/RealisticVision.yaml
# For ToonYou
python inference_magictime.py --config sample_configs/ToonYou.yaml
# For RcnzCartoon
python inference_magictime.py --config sample_configs/RcnzCartoon.yaml
# or you can directly run the .sh
sh inference.sh
```

You can also put all your *custom prompts* in a <u>.txt</u> file and run:

```bash
# For Realistic
python inference_magictime.py --config sample_configs/RealisticVision.yaml --run-txt XXX.txt --batch-size 2
# For ToonYou
python inference_magictime.py --config sample_configs/ToonYou.yaml --run-txt XXX.txt --batch-size 2
# For RcnzCartoon
python inference_magictime.py --config sample_configs/RcnzCartoon.yaml --run-txt XXX.txt --batch-size 2
```

## Community Contributions

We found some plugins created by community developers. Thanks for their efforts: 

  - ComfyUI Extension. [ComfyUI-MagicTimeWrapper](https://github.com/kijai/ComfyUI-MagicTimeWrapper) (by [@Kijai](https://github.com/kijai)). And you can click [here](https://www.bilibili.com/video/BV1wx421U7Gn/?spm_id_from=333.1007.top_right_bar_window_history.content.click) to view the installation tutorial.
  - Replicate Demo & Cloud API. [Replicate-MagicTime](https://replicate.com/camenduru/magictime) (by [@camenduru](https://twitter.com/camenduru)).
  - Jupyter Notebook. [Jupyter-MagicTime](https://github.com/camenduru/MagicTime-jupyter) (by [@ModelsLab](https://modelslab.com/)).

If you find related work, please let us know. 

## 🐳 ChronoMagic Dataset
ChronoMagic with 2265 metamorphic time-lapse videos, each accompanied by a detailed caption. We released the subset of ChronoMagic used to train MagicTime. The dataset can be downloaded at [HuggingFace Dataset](https://huggingface.co/datasets/BestWishYsh/ChronoMagic), or you can download it with the following command. Some samples can be found on our [Project Page](https://pku-yuangroup.github.io/MagicTime/).
```bash
huggingface-cli download --repo-type dataset \
--resume-download BestWishYsh/ChronoMagic \
--local-dir BestWishYsh/ChronoMagic \
--local-dir-use-symlinks False
```

## 👍 Acknowledgement
* [Animatediff](https://github.com/guoyww/AnimateDiff/tree/main) The codebase we built upon and it is a strong U-Net-based text-to-video generation model.

* [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) The codebase we built upon and it is a simple and scalable DiT-based text-to-video generation repo, to reproduce [Sora](https://openai.com/sora).

## 🔒 License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/MagicTime/blob/main/LICENSE) file.
* The service is a research preview. Please contact us if you find any potential violations.

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{yuan2025magictime,
  title={Magictime: Time-lapse video generation models as metamorphic simulators},
  author={Yuan, Shenghai and Huang, Jinfa and Shi, Yujun and Xu, Yongqi and Zhu, Ruijie and Lin, Bin and Cheng, Xinhua and Yuan, Li and Luo, Jiebo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

## 🤝 Contributors
<a href="https://github.com/PKU-YuanGroup/MagicTime/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/MagicTime&anon=true" />

</a>

