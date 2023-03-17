import torch

tp = torch.load("/home/logs/tp/iter_0000001/mp_rank_00/model_optim_rng.pt", map_location="cpu")
ep = torch.load("/home/logs/ep/iter_0000001/mp_rank_00/model_optim_rng.pt", map_location="cpu")
print(tp.keys())
# print(tp['model']['language_model']['embedding']['word_embeddings']['weight'].size())
# print(tp['model']['language_model']['embedding']['word_embeddings']['weight'])
# print(ep['model']['language_model']['embedding']['word_embeddings']['weight'])
# print((tp['model']['language_model']['embedding']['word_embeddings']['weight'][:12672//2] - ep['model']['language_model']['embedding']['word_embeddings']['weight']).abs().max())
# print(tp[0])
print(tp['optimizer']['state'].keys())
print(tp['optimizer']['state'][0].keys())
print(tp['optimizer']['state'][0]['exp_avg'])
# print(tp['optimizer']['param_groups'].keys())
print(ep['optimizer']['state'].keys())
# print(ep['optimizer']['param_groups'].keys())
# print((tp['optimizer']['language_model']['embedding']['word_embeddings']['weight'][:12672//2] - ep['optimizer']['language_model']['embedding']['word_embeddings']['weight']).abs().max())
# print((tp['optimizer']['state'][0]['exp_avg'][:12672//2] - ep['optimizer']['state'][0]['exp_avg']).abs().max())
# for i in range(1, 75):
#     print(i, (tp['optimizer']['state'][i]['exp_avg'] - ep['optimizer']['state'][i]['exp_avg']).abs().max())
#     print(i, (tp['optimizer']['state'][i]['exp_avg_sq'] - ep['optimizer']['state'][i]['exp_avg_sq']).abs().max())

# print("=="*20)
print("difference in embedding params:")
print((tp['model']['language_model']['embedding']['word_embeddings']['weight'][:12672//2] - ep['model']['language_model']['embedding']['word_embeddings']['weight']).abs().max())

print("difference in encoder params:")
for key in ep['model']['language_model']['encoder'].keys():
    print(key, (tp['model']['language_model']['encoder'][key] - ep['model']['language_model']['encoder'][key]).abs().max())

print("difference in optimizer params:")
print((tp['optimizer']['state'][0]['exp_avg'][:12672//2] - ep['optimizer']['state'][0]['exp_avg']).abs().max())
print((tp['optimizer']['state'][0]['exp_avg_sq'][:12672//2] - ep['optimizer']['state'][0]['exp_avg_sq']).abs().max())
for i in range(1, 75):
    print(i, 'exp_avg', (tp['optimizer']['state'][i]['exp_avg'] - ep['optimizer']['state'][i]['exp_avg']).abs().max())
    print(i, 'exp_avg_sq', (tp['optimizer']['state'][i]['exp_avg_sq'] - ep['optimizer']['state'][i]['exp_avg_sq']).abs().max())
# print(ep['model']['language_model']['encoder']['layers.0.self_attention.query_key_value.weight'])
# print(tp['model']['language_model']['encoder']['layers.0.self_attention.query_key_value.weight'])

# print("difference in optims exp_avg")
