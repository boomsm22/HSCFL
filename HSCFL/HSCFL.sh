#SCCT: --lora_ranks:{[128,32], [64, 16], [32,8]} --gate_dims:{[16,8,8,16], [8,4,4,8], [4,2,2,4], [2,1,1,2]} --lr:{1e-2, 5e-3, 1e-3}
#SMSF: --listwise_K:{8,12} --listwise_weight:{0,0.001,0.01,0.1,0.5,1.0,1.5,2.0}
#gowalla
python pretrain.py --dataset gowalla --anneal_cap 1.0 --ratio 50
python hybrid_finetuning.py --dataset gowalla --lr 1e-2 --anneal_cap 1.0 --lora_ranks 128 32 --gate_dims 2 1 1 2 --ratio 50
python listwise_relational_distillation.py --dataset gowalla --ratio 50 --recdim 64 --layer 3 --lora_ranks 128 32 --gate_dims 2 1 1 2 --listwise_K 8 --distill_temperature 4.0 --listwise_weight 0.01
python multi-source_fusion.py --dataset gowalla --ratio 50
#yelp2018
python pretrain.py --dataset yelp2018 --anneal_cap 1.0 --ratio 50
python hybrid_finetuning.py --dataset yelp2018 --lr 5e-3 --anneal_cap 1.0 --lora_ranks 128 32 --gate_dims 2 1 1 2 --ratio 50
python listwise_relational_distillation.py --dataset yelp2018 --ratio 50 --recdim 64 --layer 4 --lora_ranks 128 32 --gate_dims 2 1 1 2 --listwise_K 8 --distill_temperature 4.0 --listwise_weight 0
python multi-source_fusion.py --dataset yelp2018 --ratio 50
#amazon
python pretrain.py --dataset amazon --anneal_cap 0.6 --ratio 50
python hybrid_finetuning.py --dataset amazon --lr 1e-2 --anneal_cap 0.6 --lora_ranks 32 8 --gate_dims 2 1 1 2 --ratio 50
python listwise_relational_distillation.py --dataset amazon --ratio 50 --recdim 64 --layer 3 --lora_ranks 32 8 --gate_dims 2 1 1 2 --listwise_K 8 --distill_temperature 4.0 --listwise_weight 0.01
python multi-source_fusion.py --dataset amazon --ratio 50
