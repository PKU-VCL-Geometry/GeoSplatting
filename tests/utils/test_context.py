import torch

from rfstudio.utils.context import create_random_seed_context

if __name__ == '__main__':
    torch.manual_seed(1)
    gt_results1 = torch.tensor([0.7576315999031067, 0.2793108820915222, 0.40306925773620605])
    gt_results2 = torch.tensor([0.7346844673156738, 0.029281556606292725, 0.7998586297035217])
    assert torch.rand(3).allclose(gt_results1)
    assert torch.rand(3).allclose(gt_results2)
    torch.manual_seed(1)
    assert torch.rand(3).cuda().allclose(gt_results1.cuda())
    assert torch.rand(3).cuda().allclose(gt_results2.cuda())

    torch.manual_seed(1)
    assert torch.rand(3).allclose(gt_results1)
    with create_random_seed_context(seed=1):
        assert torch.rand(3).allclose(gt_results1)
        assert torch.rand(3).cuda().allclose(gt_results2.cuda())
    assert torch.rand(3).cuda().allclose(gt_results2.cuda())
