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
    force=True,  # ensure our config applies even if another handler was set up
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    加载 OpenPI checkpoint (支持 OCDBT 和传统格式)

    OpenPI 的 checkpoint 结构：
    - checkpoint_path/params/ (OCDBT format, 训练检查点的标准格式)
    - checkpoint_path/checkpoint (pickle format, 传统格式)
    """
    checkpoint_path = Path(checkpoint_path)

    # 优先尝试 OCDBT 格式 (训练检查点的标准格式)
    # 情况1: 用户直接指向 OCDBT 目录 (manifest.ocdbt 在当前目录)
    # 情况2: 用户指向父目录 (manifest.ocdbt 在 params/ 子目录)
    ocdbt_dirs = []
    if (checkpoint_path / "manifest.ocdbt").exists():
        ocdbt_dirs.append(checkpoint_path)
    if (checkpoint_path / "params" / "manifest.ocdbt").exists():
        ocdbt_dirs.append(checkpoint_path / "params")

    for params_dir in ocdbt_dirs:
        try:
            logger.info(f"Loading OCDBT checkpoint from {params_dir}")

            # 使用官方的 restore_params 工具
            params = restore_params(
                params_dir,
                restore_type=np.ndarray,
                dtype=None  # 保留原始 dtype
            )

            # 包装成 checkpoint 结构
            checkpoint = {"params": params}
            logger.info("✓ Loaded using OCDBT/Orbax format")
            return checkpoint

        except Exception as e:
            logger.warning(f"Failed to load as OCDBT from {params_dir}: {e}")

    # 回退到传统格式 (pickle/msgpack)
    possible_files = [
        checkpoint_path / 'checkpoint',
        checkpoint_path / 'checkpoint.msgpack',
        checkpoint_path,  # 可能直接是文件
    ]

    for ckpt_file in possible_files:
        if ckpt_file.exists() and ckpt_file.is_file():
            try:
                logger.info(f"Loading checkpoint from {ckpt_file}")

                # 尝试 pickle
                try:
                    with open(ckpt_file, 'rb') as f:
                        checkpoint = pickle.load(f)
                    logger.info("✓ Loaded using pickle format")
                    return checkpoint
                except:
                    pass

                # 尝试 msgpack
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
        f"Could not find valid checkpoint in {checkpoint_path}\n"
        f"Tried OCDBT at: {params_dir}\n"
        f"Tried legacy formats: {[str(f) for f in possible_files]}"
    )


def detect_lora_params(flat_params: Dict[str, Any]) -> Tuple[Set[str], Dict[str, int]]:
    """
    检测 LoRA 参数并统计信息

    支持多种 LoRA 命名约定:
    - path/w + path/lora_a + path/lora_b (标准格式)
    - path/kernel + path/lora_A + path/lora_B (传统格式)
    - path_lora_a + path_lora_b (扁平格式)

    Returns:
        lora_keys: 包含 LoRA 的参数 key 集合
        lora_stats: 统计信息 {layer_name: rank}
    """
    lora_keys = set()
    lora_stats = {}

    for key in flat_params.keys():
        # 检测小写 lora_a/lora_b 或大写 lora_A/lora_B
        if 'lora_a' in key or 'lora_A' in key:
            # 提取 base key
            # 尝试多种命名模式
            base_key = None

            if '/lora_a' in key:
                # 标准格式: path/lora_a -> path/w
                base_key_w = key.replace('/lora_a', '/w')
                base_key_kernel = key.replace('/lora_a', '/kernel')
                if base_key_w in flat_params:
                    base_key = base_key_w
                elif base_key_kernel in flat_params:
                    base_key = base_key_kernel
            elif '/lora_A' in key:
                # 传统格式: path/lora_A -> path/kernel
                base_key_kernel = key.replace('/lora_A', '/kernel')
                base_key_w = key.replace('/lora_A', '/w')
                if base_key_kernel in flat_params:
                    base_key = base_key_kernel
                elif base_key_w in flat_params:
                    base_key = base_key_w
            elif '_lora_a' in key:
                # 扁平格式: path_lora_a -> path
                base_key = key.replace('_lora_a', '')
                if base_key not in flat_params:
                    base_key = None

            if base_key:
                lora_keys.add(base_key)
                # 记录 rank
                rank = flat_params[key].shape[-1]  # lora_a: [in_dim, rank]
                lora_stats[base_key] = rank

    return lora_keys, lora_stats


def merge_lora_weights(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    合并 LoRA 权重到基础模型
    
    Returns:
        merged_params: 合并后的参数树
        merge_info: 合并统计信息
    """
    # 1. 展平参数树
    logger.info("Flattening parameter tree...")
    flat_params = traverse_util.flatten_dict(params, sep='/')
    
    # 2. 检测 LoRA 参数
    logger.info("Detecting LoRA parameters...")
    lora_keys, lora_stats = detect_lora_params(flat_params)
    
    if not lora_keys:
        logger.warning("⚠️  No LoRA parameters found in checkpoint!")
        logger.warning("This might be a base model without LoRA fine-tuning.")
        return params, {'num_merged': 0, 'details': {}}
    
    logger.info(f"Found {len(lora_keys)} layers with LoRA weights")
    
    # 3. 合并 LoRA 权重
    merged_flat = {}
    skip_keys = set()
    merge_info = {'num_merged': 0, 'details': {}}
    
    for key, value in flat_params.items():
        # 跳过已处理的 key
        if key in skip_keys:
            continue
        
        # 检查是否是需要合并的 kernel/w
        if key in lora_keys:
            # 获取 LoRA 组件 - 支持多种命名约定
            lora_a_key = None
            lora_b_key = None
            scale_key = None

            # 尝试不同的 LoRA 命名模式
            if '/w' in key:
                # 标准格式: path/w -> path/lora_a, path/lora_b
                lora_a_key = key.replace('/w', '/lora_a')
                lora_b_key = key.replace('/w', '/lora_b')
                scale_key = key.replace('/w', '/lora_scale')
            elif '/kernel' in key:
                # 传统格式: path/kernel -> path/lora_A 或 path/lora_a
                lora_a_key = key.replace('/kernel', '/lora_A')
                if lora_a_key not in flat_params:
                    lora_a_key = key.replace('/kernel', '/lora_a')
                lora_b_key = key.replace('/kernel', '/lora_B')
                if lora_b_key not in flat_params:
                    lora_b_key = key.replace('/kernel', '/lora_b')
                scale_key = key.replace('/kernel', '/lora_scale')
            else:
                # 扁平格式: path -> path_lora_a, path_lora_b
                lora_a_key = key + '_lora_a'
                lora_b_key = key + '_lora_b'
                scale_key = key + '_lora_scale'

            # 确保 LoRA 权重存在
            if lora_a_key not in flat_params or lora_b_key not in flat_params:
                logger.warning(f"⚠️  Missing LoRA weights for {key}, using base weight")
                merged_flat[key] = value
                continue
            
            # 加载权重
            base_weight = value
            lora_a = flat_params[lora_a_key]
            lora_b = flat_params[lora_b_key]

            # 检查维度
            # LoRA 在最后两个维度上工作:
            # - base_weight: [..., in_dim, out_dim]
            # - lora_a: [..., in_dim, rank]
            # - lora_b: [..., rank, out_dim]
            # 前面的维度必须匹配

            if len(base_weight.shape) < 2 or len(lora_a.shape) < 2 or len(lora_b.shape) < 2:
                logger.error(f"❌ Insufficient dimensions for {key}")
                logger.error(f"   Base: {base_weight.shape}")
                logger.error(f"   LoRA_a: {lora_a.shape}")
                logger.error(f"   LoRA_b: {lora_b.shape}")
                raise ValueError(f"Insufficient dimensions for {key}")

            # 检查前导维度是否匹配
            if base_weight.shape[:-2] != lora_a.shape[:-2] or base_weight.shape[:-2] != lora_b.shape[:-2]:
                logger.error(f"❌ Leading dimensions mismatch for {key}:")
                logger.error(f"   Base: {base_weight.shape}")
                logger.error(f"   LoRA_a: {lora_a.shape}")
                logger.error(f"   LoRA_b: {lora_b.shape}")
                raise ValueError(f"Leading dimensions mismatch for {key}")

            # 检查最后两个维度的兼容性
            expected_shapes = (
                base_weight.shape[-2] == lora_a.shape[-2] and  # in_dim 匹配
                base_weight.shape[-1] == lora_b.shape[-1] and  # out_dim 匹配
                lora_a.shape[-1] == lora_b.shape[-2]           # rank 一致
            )

            if not expected_shapes:
                logger.error(f"❌ Shape mismatch for {key}:")
                logger.error(f"   Base: {base_weight.shape}")
                logger.error(f"   LoRA_a: {lora_a.shape}")
                logger.error(f"   LoRA_b: {lora_b.shape}")
                raise ValueError(f"Shape mismatch for {key}")

            # 计算 LoRA delta: ΔW = a @ b
            # matmul 会自动处理批次维度并在最后两个维度上执行矩阵乘法
            lora_delta = jnp.matmul(lora_a, lora_b)

            # 应用 scale (如果存在)
            scale = 1.0
            if scale_key in flat_params:
                scale = float(flat_params[scale_key])
                lora_delta = lora_delta * scale
                skip_keys.add(scale_key)

            # 合并: W_new = W_base + ΔW
            merged_weight = base_weight + lora_delta

            # 计算变化幅度
            delta_norm = float(jnp.linalg.norm(lora_delta))
            base_norm = float(jnp.linalg.norm(base_weight))
            relative_change = delta_norm / (base_norm + 1e-8)

            # 记录信息
            rank = lora_a.shape[1]
            merge_info['details'][key] = {
                'rank': rank,
                'scale': scale,
                'delta_norm': delta_norm,
                'base_norm': base_norm,
                'relative_change': relative_change
            }

            logger.info(f"✓ Merged: {key}")
            logger.info(f"    Shape: {merged_weight.shape}, Rank: {rank}, "
                       f"Relative change: {relative_change:.2%}")

            merged_flat[key] = merged_weight
            merge_info['num_merged'] += 1

            # 标记 LoRA 权重为已处理
            skip_keys.add(lora_a_key)
            skip_keys.add(lora_b_key)
            
        elif 'lora_' not in key:
            # 非 LoRA 参数（如 bias, layernorm 等），直接保留
            merged_flat[key] = value
        # 'lora_A', 'lora_B', 'lora_scale' 会被自动跳过
    
    # 4. 重建参数树
    logger.info("Rebuilding parameter tree...")
    merged_params = traverse_util.unflatten_dict(merged_flat, sep='/')
    
    return merged_params, merge_info


def save_checkpoint(checkpoint: Dict[str, Any], output_path: Path):
    """以标准 Orbax/OCDBT 训练检查点目录保存合并后的 checkpoint。

    保存的布局与训练时 `openpi.training.checkpoints.save_state` 写出的
    `"params"` item 完全一致，从而可以被 `openpi.models.model.restore_params`
    直接加载：
      - 外层的 `"params"` 是 CheckpointManager 的 item 名（对应 `<step>/params` 目录）；
      - 内层的 `"params"` 是 OCDBT 中的 PyTree 根节点，下面才是 `PaliGemma` 等键。

    - 当 output_path 的最后一段是纯数字（例如 29999）时：
      使用 CheckpointManager 在其父目录创建一个与训练一致的 step 目录结构：
        <parent>/<step>/
          - _CHECKPOINT_METADATA
          - params/ (OCDBT: manifest.ocdbt, _METADATA, 数据分片等)

    - 其他情况：
      回退为在 output_path/params 下直接写入 OCDBT（无 _CHECKPOINT_METADATA）。
    """
    output_path = Path(output_path)

    # 仅保存用于推理的 params（与训练保存的可推理子集一致）。
    # 注意这里的结构：
    #   - CheckpointManager.save(items={"params": params_tree}) 中，外层 key "params"
    #     只是 item 名，对应生成 <step>/params 目录，不会参与 OCDBT 内部的 PyTree 结构；
    #   - params_tree 本身是 {"params": <pure params>}，与训练时保存完全一致，
    #     也与 restore_params 中的 wrap/unwrap 逻辑相匹配。
    pure_params = checkpoint["params"]
    params_tree = {"params": pure_params}

    if output_path.name.isdigit():
        # 将 output_path 视为一个 step 目录名，使用其父目录作为 checkpoint 根目录
        step = int(output_path.name)
        parent = output_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Saving merged checkpoint in Orbax format using CheckpointManager: parent={parent}, step={step}"
        )
        mngr = ocp.CheckpointManager(
            parent,
            item_handlers={
                # 仅保存用于推理的 params 项
                "params": ocp.PyTreeCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                # 与训练保持一致：目录已由我们创建，无需让管理器再创建
                create=False,
                async_options=ocp.AsyncOptions(timeout_secs=7200),
            ),
        )

        # CheckpointManager.save 会在 <parent>/<step>/ 下写入
        mngr.save(step, {"params": params_tree})
        # 确保保存完成，避免脚本提前退出
        mngr.wait_until_finished()
        logger.info(f"✓ Checkpoint saved successfully to {parent / str(step)} (OCDBT)")
    else:
        # 回退：直接在 output_path/params 下写 OCDBT，满足下游对 params 路径的读取
        params_dir = output_path / "params"
        params_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving merged checkpoint (params only) to {params_dir} in OCDBT format")
        # 使用 CheckpointManager 同样强制同步保存，避免脚本退出导致异步未完成。
        parent = params_dir.parent
        mngr = ocp.CheckpointManager(
            parent,
            item_handlers={
                "params": ocp.PyTreeCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                create=True,
                async_options=ocp.AsyncOptions(timeout_secs=7200),
            ),
        )
        # 选用一个固定 step（例如 0）写入 <parent>/0/params，然后移动到 params_dir
        tmp_step = 0
        mngr.save(tmp_step, {"params": params_tree})
        mngr.wait_until_finished()
        tmp_params = parent / str(tmp_step) / "params"
        # 移动生成的 OCDBT 目录到目标 params_dir
        import shutil
        if params_dir.exists():
            shutil.rmtree(params_dir)
        shutil.move(str(tmp_params), str(params_dir))
        # 清理空的 step 目录
        step_dir = parent / str(tmp_step)
        try:
            shutil.rmtree(step_dir)
        except Exception:
            pass
        logger.info(f"✓ Params saved successfully to {params_dir} (OCDBT)")


def verify_checkpoint(original_path: Path, merged_path: Path):
    """
    验证合并后的 checkpoint
    """
    logger.info("\n" + "="*60)
    logger.info("Verifying merged checkpoint...")
    logger.info("="*60)
    
    # 加载两个 checkpoint
    original = load_checkpoint(original_path)
    merged = load_checkpoint(merged_path)
    
    # 比较参数数量
    orig_params = traverse_util.flatten_dict(original['params'], sep='/')
    merged_params = traverse_util.flatten_dict(merged['params'], sep='/')
    
    # 统计
    orig_keys = set(k for k in orig_params.keys() if 'lora_' not in k)
    merged_keys = set(merged_params.keys())
    
    logger.info(f"Original params (non-LoRA): {len(orig_keys)}")
    logger.info(f"Merged params: {len(merged_keys)}")
    
    # 检查是否所有非 LoRA 参数都存在
    missing = orig_keys - merged_keys
    extra = merged_keys - orig_keys
    
    if missing:
        logger.warning(f"⚠️  Missing {len(missing)} parameters in merged checkpoint")
        for key in list(missing)[:5]:  # 只显示前5个
            logger.warning(f"   - {key}")
    
    if extra:
        logger.info(f"ℹ️  Extra {len(extra)} parameters (should be 0 if fully merged)")
    
    # 检查形状
    logger.info("\nChecking parameter shapes...")
    shape_match = True
    for key in orig_keys & merged_keys:
        if orig_params[key].shape != merged_params[key].shape:
            logger.error(f"❌ Shape mismatch for {key}:")
            logger.error(f"   Original: {orig_params[key].shape}")
            logger.error(f"   Merged: {merged_params[key].shape}")
            shape_match = False
    
    if shape_match:
        logger.info("✓ All parameter shapes match")
    
    # 检查是否还有 LoRA 参数
    remaining_lora = [k for k in merged_params.keys() if 'lora_' in k]
    if remaining_lora:
        logger.warning(f"⚠️  Found {len(remaining_lora)} remaining LoRA parameters:")
        for key in remaining_lora[:5]:
            logger.warning(f"   - {key}")
    else:
        logger.info("✓ No LoRA parameters remaining (fully merged)")
    
    logger.info("="*60)
    logger.info("Verification complete")
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Merge LoRA weights into base model for OpenPI checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python merge_lora.py \\
      --lora_checkpoint checkpoints/pi05_libero_finetuned/ \\
      --output checkpoints/pi05_libero_merged/

  # With verification
  python merge_lora.py \\
      --lora_checkpoint checkpoints/my_lora/ \\
      --output checkpoints/my_merged/ \\
      --verify
        """
    )
    
    parser.add_argument(
        '--lora_checkpoint',
        type=str,
        required=True,
        help='Path to LoRA fine-tuned checkpoint directory (can be the OCDBT params dir, or a directory containing params)'
    )

    parser.add_argument(
        '--base_checkpoint',
        type=str,
        required=False,
        default=None,
        help='Optional path to base checkpoint to merge into (use when LoRA checkpoint contains only LoRA deltas)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for merged checkpoint'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify merged checkpoint after saving'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 1. 加载 checkpoint
        logger.info("="*60)
        logger.info("Step 1: Loading checkpoints")
        logger.info("="*60)

        lora_ckpt = load_checkpoint(args.lora_checkpoint)
        if 'params' not in lora_ckpt:
            logger.error("❌ Invalid LoRA checkpoint format: 'params' key not found")
            sys.exit(1)

        base_ckpt = None
        if args.base_checkpoint is not None:
            base_ckpt = load_checkpoint(args.base_checkpoint)
            if 'params' not in base_ckpt:
                logger.error("❌ Invalid base checkpoint format: 'params' key not found")
                sys.exit(1)
            logger.info("✓ Loaded base checkpoint")
        else:
            logger.info("ℹ️  No base checkpoint provided; will assume LoRA checkpoint includes base weights")

        # 2. 选择用于合并的参数树
        #    - 如果提供 base，则在 base 上叠加 LoRA 组件
        #    - 否则，直接使用 LoRA ckpt（其应包含 base + LoRA）
        from copy import deepcopy
        if base_ckpt is not None:
            base_params = base_ckpt['params']
            lora_params = lora_ckpt['params']

            base_flat = traverse_util.flatten_dict(base_params, sep='/')
            lora_flat = traverse_util.flatten_dict(lora_params, sep='/')

            # 仅将 LoRA 相关键覆盖到 base 上
            for k, v in lora_flat.items():
                if 'lora_' in k:
                    base_flat[k] = v

            params_for_merge = traverse_util.unflatten_dict(base_flat, sep='/')
        else:
            params_for_merge = lora_ckpt['params']

        # 3. 合并 LoRA 权重
        logger.info("\n" + "="*60)
        logger.info("Step 2: Merging LoRA weights")
        logger.info("="*60)
        merged_params, merge_info = merge_lora_weights(params_for_merge)
        
        # 打印统计信息
        logger.info("\n" + "-"*60)
        logger.info("Merge Statistics:")
        logger.info("-"*60)
        logger.info(f"Total layers merged: {merge_info['num_merged']}")
        
        if merge_info['num_merged'] > 0:
            # 计算平均统计
            details = merge_info['details']
            avg_relative_change = np.mean([d['relative_change'] for d in details.values()])
            ranks = [d['rank'] for d in details.values()]
            
            logger.info(f"LoRA ranks used: {set(ranks)}")
            logger.info(f"Average relative change: {avg_relative_change:.2%}")
            logger.info(f"Max relative change: {max(d['relative_change'] for d in details.values()):.2%}")
            logger.info(f"Min relative change: {min(d['relative_change'] for d in details.values()):.2%}")
        logger.info("-"*60)
        
        # 4. 保存合并后的 checkpoint
        logger.info("\n" + "="*60)
        logger.info("Step 3: Saving merged checkpoint")
        logger.info("="*60)
        
        # 更新 checkpoint（尽量保留 base 的 metadata）
        if base_ckpt is not None:
            checkpoint_to_save = deepcopy(base_ckpt)
        else:
            checkpoint_to_save = deepcopy(lora_ckpt)
        checkpoint_to_save['params'] = merged_params
        
        # 可选：添加元数据
        if 'metadata' not in checkpoint_to_save:
            checkpoint_to_save['metadata'] = {}
        checkpoint_to_save['metadata']['lora_merged'] = True
        checkpoint_to_save['metadata']['merge_info'] = {
            'num_merged': merge_info['num_merged'],
            'source_lora_checkpoint': str(args.lora_checkpoint),
            'base_checkpoint': str(args.base_checkpoint) if args.base_checkpoint else None,
        }
        
        # 保存为标准 Orbax/OCDBT 训练检查点目录结构
        save_checkpoint(checkpoint_to_save, args.output)
        
        # 5. 验证（可选）
        if args.verify:
            # 当提供 base 时，用 base 作为数值对比基准
            verify_base = args.base_checkpoint if args.base_checkpoint else args.lora_checkpoint
            verify_checkpoint(args.lora_checkpoint, args.output)
            # 额外数值校验：检查 merged ≈ base + A@B
            try:
                logger.info("Performing numeric verification of merged weights...")
                lora_ck = load_checkpoint(args.lora_checkpoint)
                base_ck = load_checkpoint(verify_base)
                merged_ck = load_checkpoint(args.output)

                lora_flat = traverse_util.flatten_dict(lora_ck['params'], sep='/')
                base_flat = traverse_util.flatten_dict(base_ck['params'], sep='/')
                merged_flat = traverse_util.flatten_dict(merged_ck['params'], sep='/')

                # 检测 LoRA 目标层
                _, lora_stats = detect_lora_params(lora_flat)

                total = 0
                passed = 0
                worst_rel_err = 0.0
                worst_key = None

                for key in lora_stats.keys():
                    # 组装 LoRA 组件键
                    if '/w' in key:
                        a_key = key.replace('/w', '/lora_a')
                        b_key = key.replace('/w', '/lora_b')
                        s_key = key.replace('/w', '/lora_scale')
                    elif '/kernel' in key:
                        a_key = key.replace('/kernel', '/lora_A')
                        if a_key not in lora_flat:
                            a_key = key.replace('/kernel', '/lora_a')
                        b_key = key.replace('/kernel', '/lora_B')
                        if b_key not in lora_flat:
                            b_key = key.replace('/kernel', '/lora_b')
                        s_key = key.replace('/kernel', '/lora_scale')
                    else:
                        a_key = key + '_lora_a'
                        b_key = key + '_lora_b'
                        s_key = key + '_lora_scale'

                    if a_key not in lora_flat or b_key not in lora_flat:
                        continue

                    total += 1
                    base_w = base_flat[key]
                    a = lora_flat[a_key]
                    b = lora_flat[b_key]
                    s = float(lora_flat[s_key]) if s_key in lora_flat else 1.0
                    expected = base_w + jnp.matmul(a, b) * s
                    got = merged_flat[key]

                    # L2 相对误差
                    num = float(jnp.linalg.norm(expected - got))
                    denom = float(jnp.linalg.norm(expected) + 1e-8)
                    rel_err = num / denom
                    if rel_err < 1e-4:
                        passed += 1
                    if rel_err > worst_rel_err:
                        worst_rel_err = rel_err
                        worst_key = key

                logger.info(f"Numeric verify: passed {passed}/{total}, worst rel err {worst_rel_err:.2e} at {worst_key}")
                if passed != total:
                    logger.warning("⚠️  Some layers failed numeric verification; inspect logs above.")
            except Exception as ve:
                logger.warning(f"Numeric verification skipped due to error: {ve}")
        
        # 完成
        logger.info("\n" + "="*60)
        logger.info("✓ LoRA merge completed successfully!")
        logger.info("="*60)
        logger.info(f"Merged checkpoint saved to: {args.output}")
        logger.info(f"You can now use this checkpoint with realtime-vla:")
        logger.info(f"  python convert_from_jax.py \\")
        logger.info(f"      --jax_path {args.output} \\")
        logger.info(f"      --output converted.pkl")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ Error during merge: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
