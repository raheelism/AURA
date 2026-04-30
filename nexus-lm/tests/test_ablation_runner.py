from evaluation.ablation_runner import build_model, plan_ablation_runs


def test_plan_ablation_runs_filters_unknown_names():
    planned = plan_ablation_runs(['baseline', 'does_not_exist', 'aurora_full'])
    names = [name for name, _ in planned]
    assert names == ['baseline', 'aurora_full']


def test_build_model_smoke_small_configs():
    base_config = {
        'model': {
            'n_blocks': 2,
            'd_surface': 32,
            'n_heads_surface': 2,
            'n_kv_heads_surface': 1,
            'vocab_size': 64,
            'max_seq_len': 16,
            'd_reasoning': 16,
            'd_verify': 8,
            'n_reasoning_slots': 4,
            'n_heads_reasoning': 2,
            'd_ffn_pattern': 32,
            'd_ffn_semantic': 64,
            'd_ffn_reasoning': 32,
            'local_window': 4,
            'bridge_top_k': 2,
            'cope_positions': 4,
            'surprise_loss_weight': 0.1,
            'difficulty_entropy_weight': 0.01,
        },
        'training': {
            'batch_size': 2,
            'gradient_accumulation': 1,
            'max_lr': 1e-3,
            'min_lr': 1e-4,
            'grad_clip': 1.0,
            'weight_decay': 0.1,
            'dtype': 'float32',
        },
    }
    llama = build_model(base_config, {'model_type': 'llama'})
    aurora = build_model(base_config, {'model_type': 'aurora'})
    assert llama is not None
    assert aurora is not None
