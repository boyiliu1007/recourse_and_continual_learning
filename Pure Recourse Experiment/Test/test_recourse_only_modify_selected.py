import torch as pt

def test_only_selected_data_modified(model, original, modified, selected_indices):
    # Check shapes match
    assert original.x.shape == modified.x.shape, "Data shape changed unexpectedly!"
    assert original.y.shape == modified.y.shape, "Data shape changed unexpectedly!"

    # Loop over all data points
    for idx in range(len(original.x)):
        if idx in selected_indices:
            # Selected indices should be modified (allow for floating point diff)
            if pt.allclose(original.x[idx], modified.x[idx]) and pt.allclose(original.y[idx], modified.y[idx]):
                # print("original.x[idx]", original.x[idx])
                # print("modified.x[idx]", modified.x[idx])
                # print("original.y[idx]", original.y[idx])
                # print("modified.y[idx]", modified.y[idx])
                # print("orginal score", model(original.x[idx].unsqueeze(0)))
                # print("modified score", model(modified.x[idx].unsqueeze(0)))
                # print("selected_indices", selected_indices)
                print(f"index {idx} choose not to recourse due to extreme cost (it is too far from the decision boundary)")
                
        else:
            if not (pt.equal(original.x[idx], modified.x[idx]) and pt.equal(original.y[idx], modified.y[idx])):
                print("original.x[idx]", original.x[idx])
                print("modified.x[idx]", modified.x[idx])
                print("original.y[idx]", original.y[idx])
                print("modified.y[idx]", modified.y[idx])
                print("orginal score", model(original.x[idx].unsqueeze(0)))
                print("modified score", model(modified.x[idx].unsqueeze(0)))
                print("selected_indices", selected_indices)
                raise AssertionError(f"Selected index {idx} was NOT modified!")
                
        
    return True