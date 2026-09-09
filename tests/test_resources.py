from image_curator.resources import ResourceProfile, choose_resource_plan


def test_large_free_vram_chooses_bounded_parallel_plan():
    plan = choose_resource_plan(ResourceProfile(12, 32.0, 10.0, "gpu", 16000, 9000, 7000, 20))

    assert plan.workers == 2
    assert plan.batch_size == 4
    assert plan.decode_threads == 4
    assert plan.pause_below_free_vram_mib == 2048


def test_missing_gpu_falls_back_to_single_small_batch():
    plan = choose_resource_plan(ResourceProfile(2, 4.0, 1.0))

    assert plan.workers == 1
    assert plan.batch_size == 1
    assert plan.decode_threads == 1
