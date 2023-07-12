# LENeRF_public (CVPR'23)

project page: [Link](https://lenerf.github.io/)
arxiv: [Link](https://arxiv.org/abs/2306.12570)




Our codebase is borrowed from [EG3D](https://github.com/NVlabs/eg3d).

## Environment Install
Install conda environment usig `lenerf_env.yml`.

## Model inference
Configure `config_list` in `inferece.py`, and run 

```
bash inference.sh
```

## LENeRF training
Make a custom config file, and set `--config-path` in the `clipedit.sh` file, and run 
```
bash clipedit.sh
```

## BibTeX
```
@inproceedings{hyung2023local,
  title     =   {Local 3D Editing via 3D Distillation of CLIP Knowledge},
  author    =   {Hyung, Junha and Hwang, Sungwon and Kim, Daejin and Lee, Hyunji and Choo, Jaegul},
  booktitle =   {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     =   {12674--12684},
  year      =   {2023}
}
```
