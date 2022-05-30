CUDA_VISIBLE_DEVICES=4 python main.py --train config/webqsp_graftnet.yml

CUDA_VISIBLE_DEVICES=4 python main.py --train config/webqsp_kvmem.yml

CUDA_VISIBLE_DEVICES=4 python main.py --train config/webqsp_graphmem.yml

CUDA_VISIBLE_DEVICES=3 python main.py --test config/webqsp_graphmem.yml

CUDA_VISIBLE_DEVICES=6 python main.py --train config/metaqa_1hop_graphmem.yml

CUDA_VISIBLE_DEVICES=6 python main.py --train config/metaqa_1hop_graphmem.yml


CUDA_VISIBLE_DEVICES=1 python main.py --train config/metaqa_3hop_graphmem.yml

CUDA_VISIBLE_DEVICES=1 python main.py --train config/metaqa_2hop_graphmem.yml


CUDA_VISIBLE_DEVICES=5 python main.py --test config/metaqa_3hop_graphmem.yml



CUDA_VISIBLE_DEVICES=0 python main.py --test config_/metaqa_3hop_graphmem_v20.0.yml
