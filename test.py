import torch

# Example tensors
tensor0 = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]])
negative_mask = torch.tensor([2, 4, 5, 6, 7])
positive_mask = torch.tensor([0, 1, 3, 8, 9])
chosen_mask = torch.tensor([0, 1, 4])

# Use tensorB as indices to index positive_mask
result_mask = positive_mask[chosen_mask]
result_ftr_mask = positive_mask[~torch.isin(positive_mask, result_mask)]
new_mask = torch.cat((result_mask, negative_mask))
#sort the new mask
new_mask = new_mask.sort()[0]

print(tensor0[new_mask])
print(result_ftr_mask)

