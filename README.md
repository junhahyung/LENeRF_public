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

