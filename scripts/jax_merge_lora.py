#!/usr/bin/env python3
"""
merge_lora.py - 合并 OpenPI LoRA 微调权重到基础模型

用法:
    python merge_lora.py \
        --lora_checkpoint checkpoints/my_lora_finetune/ \
        --output checkpoints/my_merged/ \
        --verify
"""

# 必须在导入 JAX 之前设置
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Set
import logging
from copy import deepcopy

import jax
import jax.numpy as jnp
from flax import traverse_util
import numpy as np

from openpi.models.model import restore_params
import orbax.checkpoint as ocp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    加载 OpenPI checkpoint (支持 OCDBT 和传统格式)
    """
    checkpoint_path = Path(checkpoint_path)

    ocdbt_dirs = []
    if (checkpoint_path / "manifest.ocdbt").exists():
        ocdbt_dirs.append(checkpoint_path)
    if (checkpoint_path / "params" / "manifest.ocdbt").exists():
        ocdbt_dirs.append(checkpoint_path / "params")

    for params_dir in ocdbt_dirs:
        try:
            logger.info(f"Loading OCDBT checkpoint from {params_dir}")
            params = restore_params(
                params_dir,
                restore_type=np.ndarray,
                dtype=None
            )
            checkpoint = {"params": params}
            logger.info("✓ Loaded using OCDBT/Orbax format")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load as OCDBT from {params_dir}: {e}")

    possible_files = [
        checkpoint_path / 'checkpoint',
        checkpoint_path / 'checkpoint.msgpack',
        checkpoint_path,
    ]

    for ckpt_file in possible_files:
        if ckpt_file.exists() and ckpt_file.is_file():
            try:
                logger.info(f"Loading checkpoint from {ckpt_file}")
                try:
                    with open(ckpt_file, 'rb') as f:
                        checkpoint = pickle.load(f)
                    logger.info("✓ Loaded using pickle format")
                    return checkpoint
                except:
                    pass
                try:
                    import msgpack
                    with open(ckpt_file, 'rb') as f:
                        checkpoint = msgpack.unpack(f)
                    logger.info("✓ Loaded using msgpack format")
                    return checkpoint
                except:
                    pass
            except Exception as e:
                logger.warning(f"Failed to load {ckpt_file}: {e}")
                continue

    raise FileNotFoundError(
        f"Could not find valid checkpoint in {checkpoint_path}"
    )


def detect_lora_params(flat_params: Dict[str, Any]) -> Tuple[Set[str], Dict[str, int]]:
    """检测 LoRA 参数并统计信息"""
    lora_keys = set()
    lora_stats = {}

    for key in flat_params.keys():
        if 'lora_a' in key or 'lora_A' in key:
            base_key = None
            if '/lora_a' in key:
                base_key_w = key.replace('/lora_a', '/w')
                base_key_kernel = key.replace('/lora_a', '/kernel')
                if base_key_w in flat_params:
                    base_key = base_key_w
                elif base_key_kernel in flat_params:
                    base_key = base_key_kernel
            elif '/lora_A' in key:
                base_key_kernel = key.replace('/lora_A', '/kernel')
                base_key_w = key.replace('/lora_A', '/w')
                if base_key_kernel in flat_params:
                    base_key = base_key_kernel
                elif base_key_w in flat_params:
                    base_key = base_key_w
            elif '_lora_a' in key:
                base_key = key.replace('_lora_a', '')
                if base_key not in flat_params:
                    base_key = None

            if base_key:
                lora_keys.add(base_key)
                rank = flat_params[key].shape[-1]
                lora_stats[base_key] = rank

    return lora_keys, lora_stats


def merge_lora_weights(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """合并 LoRA 权重到基础模型"""
    logger.info("Flattening parameter tree...")
    flat_params = traverse_util.flatten_dict(params, sep='/')
    
    logger.info("Detecting LoRA parameters...")
    lora_keys, lora_stats = detect_lora_params(flat_params)
    
    if not lora_keys:
        logger.warning("⚠️  No LoRA parameters found in checkpoint!")
        return params, {'num_merged': 0, 'details': {}}
    
    logger.info(f"Found {len(lora_keys)} layers with LoRA weights")
    
    merged_flat = {}
    skip_keys = set()
    merge_info = {'num_merged': 0, 'details': {}}
    
    for key, value in flat_params.items():
        if key in skip_keys:
            continue
        
        if key in lora_keys:
            lora_a_key = None
            lora_b_key = None
            scale_key = None

            if '/w' in key:
                lora_a_key = key.replace('/w', '/lora_a')
                lora_b_key = key.replace('/w', '/lora_b')
                scale_key = key.replace('/w', '/lora_scale')
            elif '/kernel' in key:
                lora_a_key = key.replace('/kernel', '/lora_A')
                if lora_a_key not in flat_params:
                    lora_a_key = key.replace('/kernel', '/lora_a')
                lora_b_key = key.replace('/kernel', '/lora_B')
                if lora_b_key not in flat_params:
                    lora_b_key = key.replace('/kernel', '/lora_b')
                scale_key = key.replace('/kernel', '/lora_scale')
            else:
                lora_a_key = key + '_lora_a'
                lora_b_key = key + '_lora_b'
                scale_key = key + '_lora_scale'

            if lora_a_key not in flat_params or lora_b_key not in flat_params:
                merged_flat[key] = value
                continue
            
            base_weight = value
            lora_a = flat_params[lora_a_key]
            lora_b = flat_params[lora_b_key]

            # 简化的维度检查
            if len(base_weight.shape) < 2:
                raise ValueError(f"Insufficient dimensions for {key}")

            lora_delta = jnp.matmul(lora_a, lora_b)

            scale = 1.0
            if scale_key in flat_params:
                scale = float(flat_params[scale_key])
                lora_delta = lora_delta * scale
                skip_keys.add(scale_key)

            merged_weight = base_weight + lora_delta

            delta_norm = float(jnp.linalg.norm(lora_delta))
            base_norm = float(jnp.linalg.norm(base_weight))
            relative_change = delta_norm / (base_norm + 1e-8)

            rank = lora_a.shape[1]
            merge_info['details'][key] = {
                'rank': rank,
                'scale': scale,
                'delta_norm': delta_norm,
                'base_norm': base_norm,
                'relative_change': relative_change
            }

            merged_flat[key] = merged_weight
            merge_info['num_merged'] += 1

            skip_keys.add(lora_a_key)
            skip_keys.add(lora_b_key)
            
        elif 'lora_' not in key:
            merged_flat[key] = value
    
    logger.info("Rebuilding parameter tree...")
    merged_params = traverse_util.unflatten_dict(merged_flat, sep='/')
    return merged_params, merge_info


def save_checkpoint(checkpoint: Dict[str, Any], output_path: Path):
    """以标准 Orbax/OCDBT 训练检查点目录保存合并后的 checkpoint。"""
    output_path = Path(output_path)
    pure_params = checkpoint["params"]
    params_tree = {"params": pure_params}

    if output_path.name.isdigit():
        step = int(output_path.name)
        parent = output_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving merged checkpoint to {parent / str(step)} (OCDBT)")
        mngr = ocp.CheckpointManager(
            parent,
            item_handlers={"params": ocp.PyTreeCheckpointHandler()},
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1, create=False, async_options=ocp.AsyncOptions(timeout_secs=7200)
            ),
        )
        mngr.save(step, {"params": params_tree})
        mngr.wait_until_finished()
        logger.info(f"✓ Saved successfully")
    else:
        params_dir = output_path / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving merged params to {params_dir}")
        
        parent = params_dir.parent
        mngr = ocp.CheckpointManager(
            parent,
            item_handlers={"params": ocp.PyTreeCheckpointHandler()},
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1, create=True, async_options=ocp.AsyncOptions(timeout_secs=7200)
            ),
        )
        tmp_step = 0
        mngr.save(tmp_step, {"params": params_tree})
        mngr.wait_until_finished()
        
        tmp_params = parent / str(tmp_step) / "params"
        import shutil
        if params_dir.exists():
            shutil.rmtree(params_dir)
        shutil.move(str(tmp_params), str(params_dir))
        try:
            shutil.rmtree(parent / str(tmp_step))
        except:
            pass
        logger.info(f"✓ Saved successfully")


def verify_checkpoint(original_path: Path, merged_path: Path):
    """验证合并后的 checkpoint"""
    logger.info("Verifying merged checkpoint...")
    original = load_checkpoint(original_path)
    merged = load_checkpoint(merged_path)
    
    orig_params = traverse_util.flatten_dict(original['params'], sep='/')
    merged_params = traverse_util.flatten_dict(merged['params'], sep='/')
    
    orig_keys = set(k for k in orig_params.keys() if 'lora_' not in k)
    merged_keys = set(merged_params.keys())
    
    # 检查所有非 LoRA 参数是否都存在
    missing = orig_keys - merged_keys
    if missing:
        logger.warning(f"⚠️  Missing {len(missing)} parameters")

    # 检查 LoRA 参数是否已被移除
    remaining_lora = [k for k in merged_params.keys() if 'lora_' in k]
    if remaining_lora:
        logger.warning(f"⚠️  Found {len(remaining_lora)} remaining LoRA parameters")
    else:
        logger.info("✓ No LoRA parameters remaining (clean merge)")


def main():
    parser = argparse.ArgumentParser(description='Merge LoRA weights into base model')
    parser.add_argument('--lora_checkpoint', type=str, required=True, help='Path to LoRA checkpoint')
    parser.add_argument('--base_checkpoint', type=str, required=False, default=None, help='Optional path to base checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--verify', action='store_true', help='Verify merged checkpoint')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 1. 加载 checkpoint
        logger.info("="*60)
        logger.info("Step 1: Loading checkpoints")
        logger.info("="*60)

        lora_ckpt = load_checkpoint(args.lora_checkpoint)
        
        # 2. 确定合并源 (CRITICAL FIX)
        # 逻辑修改：如果 LoRA checkpoint 包含非 LoRA 参数（即它是一个完整的微调模型），
        # 我们必须直接使用它作为基础，而不是使用 base_checkpoint。
        # 只有当 LoRA checkpoint 仅包含 delta (sparse) 时，才使用 base_checkpoint 叠加。
        
        lora_params = lora_ckpt["params"]
        lora_flat = traverse_util.flatten_dict(lora_params, sep="/")
        
        # 检查是否存在非 LoRA 键 (例如 biases, layer_norm 等)
        non_lora_keys_in_lora = {k for k in lora_flat.keys() if "lora_" not in k}
        
        # 简单的启发式检查：如果包含超过 10 个非 LoRA 张量，我们假设它是完整的
        is_full_checkpoint = len(non_lora_keys_in_lora) > 10
        
        params_for_merge = None
        base_ckpt_for_verify = None # 用于最后的数值验证

        if is_full_checkpoint:
            logger.info(f"✓ Detected {len(non_lora_keys_in_lora)} non-LoRA parameters in LoRA checkpoint.")
            logger.info("ℹ️  Treating LoRA checkpoint as the SOURCE of TRUTH (base + fine-tuned weights).")
            logger.info("   Merging process will add LoRA deltas to these weights.")
            
            # 直接使用 LoRA checkpoint 的参数，这样保留了微调过的 bias/norm
            params_for_merge = lora_params
            
            # 如果提供了 base_checkpoint，仅提示用户我们忽略了它用于合并，但可能用于验证
            if args.base_checkpoint:
                logger.info(f"ℹ️  --base_checkpoint was provided but will be ignored for merging logic")
                logger.info(f"    because LoRA checkpoint is self-contained.")
                base_ckpt_for_verify = load_checkpoint(args.base_checkpoint)
            else:
                base_ckpt_for_verify = lora_ckpt # 如果没有 base，验证时对比自己（不完美但可用）

        else:
            # Sparse checkpoint 模式
            logger.info("ℹ️  LoRA checkpoint appears to contain ONLY LoRA deltas (sparse).")
            if args.base_checkpoint is None:
                logger.error("❌ Sparse LoRA checkpoint requires --base_checkpoint to be provided!")
                sys.exit(1)
            
            logger.info(f"Loading base checkpoint from {args.base_checkpoint}")
            base_ckpt = load_checkpoint(args.base_checkpoint)
            base_ckpt_for_verify = base_ckpt
            
            # 叠加模式：Base + LoRA Deltas
            base_flat = traverse_util.flatten_dict(base_ckpt["params"], sep="/")
            merged_flat = dict(base_flat)
            
            # 将 LoRA 参数覆盖进去
            for k, v in lora_flat.items():
                if "lora_" in k:
                    merged_flat[k] = v
            
            params_for_merge = traverse_util.unflatten_dict(merged_flat, sep="/")

        # 3. 合并 LoRA 权重
        logger.info("\n" + "="*60)
        logger.info("Step 2: Merging LoRA weights")
        logger.info("="*60)
        
        merged_params, merge_info = merge_lora_weights(params_for_merge)
        
        # 打印统计
        logger.info("-" * 60)
        logger.info(f"Total layers merged: {merge_info['num_merged']}")
        if merge_info['num_merged'] > 0:
            details = merge_info['details']
            avg_relative_change = np.mean([d['relative_change'] for d in details.values()])
            logger.info(f"Average relative change: {avg_relative_change:.2%}")
        logger.info("-" * 60)
        
        # 4. 保存
        logger.info("\n" + "="*60)
        logger.info("Step 3: Saving merged checkpoint")
        logger.info("="*60)
        
        checkpoint_to_save = deepcopy(lora_ckpt) if is_full_checkpoint else deepcopy(base_ckpt)
        checkpoint_to_save['params'] = merged_params
        checkpoint_to_save.setdefault('metadata', {})['lora_merged'] = True
        
        save_checkpoint(checkpoint_to_save, args.output)
        
        # 5. 验证
        if args.verify:
            verify_checkpoint(args.lora_checkpoint, args.output)
            
            # 数值验证
            try:
                logger.info("Performing numeric verification...")
                # 重新加载刚保存的
                merged_ck = load_checkpoint(args.output)
                merged_flat = traverse_util.flatten_dict(merged_ck["params"], sep="/")
                
                # 确定用于对比的 base
                # 这里的逻辑是：merged_weight ≈ base_weight + (A @ B)
                # 如果是 full checkpoint，base_weight 来自 lora_ckpt['params'] (冻结的那些)
                # 如果是 sparse，base_weight 来自 base_ckpt['params']
                
                if is_full_checkpoint:
                    verify_source_flat = traverse_util.flatten_dict(lora_ckpt["params"], sep="/")
                else:
                    verify_source_flat = traverse_util.flatten_dict(base_ckpt["params"], sep="/")

                # 获取 LoRA 参数用于计算 delta
                # 注意：LoRA 参数一定在 lora_ckpt 里
                lora_source_flat = traverse_util.flatten_dict(lora_ckpt["params"], sep="/")
                _, lora_stats = detect_lora_params(lora_source_flat)
                
                passed = 0
                total = 0
                worst_err = 0.0
                
                for key in lora_stats.keys():
                    # 构建 key (简化版逻辑，需匹配 merge_lora_weights 的解析)
                    if "/w" in key:
                        a_key, b_key = key.replace("/w", "/lora_a"), key.replace("/w", "/lora_b")
                        s_key = key.replace("/w", "/lora_scale")
                    elif "/kernel" in key:
                        a_key, b_key = key.replace("/kernel", "/lora_A"), key.replace("/kernel", "/lora_B")
                        # Handle mismatched cases (A vs a) if needed, usually covered by detect
                        if a_key not in lora_source_flat: a_key = key.replace("/kernel", "/lora_a")
                        if b_key not in lora_source_flat: b_key = key.replace("/kernel", "/lora_b")
                        s_key = key.replace("/kernel", "/lora_scale")
                    else:
                        a_key, b_key = key + "_lora_a", key + "_lora_b"
                        s_key = key + "_lora_scale"

                    if a_key not in lora_source_flat or key not in verify_source_flat:
                        continue

                    total += 1
                    base_w = verify_source_flat[key]
                    a = lora_source_flat[a_key]
                    b = lora_source_flat[b_key]
                    scale = float(lora_source_flat[s_key]) if s_key in lora_source_flat else 1.0
                    
                    expected = base_w + jnp.matmul(a, b) * scale
                    got = merged_flat[key]
                    
                    diff = float(jnp.linalg.norm(expected - got)) / (float(jnp.linalg.norm(expected)) + 1e-8)
                    if diff < 1e-4: passed += 1
                    worst_err = max(worst_err, diff)
                
                logger.info(f"Numeric Verify: {passed}/{total} passed. Worst rel err: {worst_err:.2e}")
                
            except Exception as e:
                logger.warning(f"Verification failed with error: {e}")

        logger.info("\n✓ LoRA merge completed successfully!")

    except Exception as e:
        logger.error(f"\n❌ Error during merge: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()