"""Resolve portable task/model configs and launch one of the three training stages.

No shell expressions are executed. --dry-run resolves argv without importing
training libraries, checking local assets, creating files, or starting workers.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('actor', 'meter_sft', 'meter_rl')
METHODS = ('deltattg', 'grpo', 'dapo', 'gigpo', 'hgpo')
LOCKED = {
    'actor_rollout_ref.actor.policy_loss.loss_mode': 'vanilla',
    'actor_rollout_ref.actor.loss_agg_mode': 'token-mean',
    'algorithm.norm_adv_by_std_in_grpo': 'true',
}


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load(path):
    with Path(path).open(encoding='utf-8') as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a configuration mapping: {path}')
    return value


def scalar(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(',', ':'))
    return str(value)


class Resolver:
    def __init__(self, defaults, environment):
        self.defaults = defaults
        self.environment = environment
        self.resolved = {}
        self.stack = set()

    def get(self, key, fallback=None):
        if key in self.environment and (self.environment[key] != '' or (key not in self.defaults and fallback is None)):
            return str(self.environment[key])  # User paths are literal, never evaluated.
        if key in self.resolved:
            value = self.resolved[key]
        elif key in self.defaults:
            if key in self.stack:
                raise ValueError(f'Cyclic variable: {key}')
            self.stack.add(key)
            try:
                value = self.text(scalar(self.defaults[key]))
            finally:
                self.stack.remove(key)
            self.resolved[key] = value
        elif fallback is None:
            raise ValueError(f'Missing variable: {key}')
        else:
            value = ''
        return self.text(fallback) if value == '' and fallback is not None else value

    def text(self, text):
        # Resolve innermost ${VAR:-default} first. User values are inserted once
        # and are never expanded recursively as executable or configuration text.
        pieces = []
        position = 0
        while position < len(text):
            start = text.find('${', position)
            if start < 0:
                pieces.append(text[position:]); break
            pieces.append(text[position:start])
            depth, end = 1, start + 2
            while end < len(text) and depth:
                if text.startswith('${', end):
                    depth += 1; end += 2
                elif text[end] == '}':
                    depth -= 1; end += 1
                else:
                    end += 1
            if depth:
                raise ValueError(f'Unclosed variable in {text!r}')
            expr = text[start + 2:end - 1]
            key, separator, fallback = expr.partition(':-')
            if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
                raise ValueError(f'Unsupported interpolation: {expr}')
            pieces.append(self.get(key, fallback if separator else None))
            position = end
        return ''.join(pieces)


def layer(config, kind, task):
    # Phase/task profiles override common fields without leaking profile metadata.
    plain = {k: v for k, v in config.items() if k not in {'phases', 'tasks'}}
    plain = merge(plain, config.get('phases', {}).get(kind, {}))
    return merge(plain, config.get('tasks', {}).get(task, {}))


def build_plan(kind, task, model, *, extra_configs=(), settings=(), overrides=(), ablation=None, environment=None, method="deltattg", action_penalty_mode=None):
    if method not in METHODS or (kind != 'actor' and method != 'deltattg'):
        raise ValueError('Baseline methods apply only to actor training')
    if kind not in KINDS:
        raise ValueError(f'Unknown stage: {kind}')
    if action_penalty_mode is not None:
        if kind != 'actor':
            raise ValueError('--action-penalty-mode applies only to actor training')
        if action_penalty_mode not in ('off', 'on'):
            raise ValueError('action_penalty_mode must be off or on')
    for name in (task, model):
        if not re.fullmatch(r'[a-z0-9_]+', name):
            raise ValueError(f'Invalid task/model name: {name}')
    config = merge(load(ROOT / 'configs/launch/common.yaml'), load(ROOT / f'configs/launch/{kind}.yaml'))
    family = 'actor' if kind == 'actor' else 'meter'
    task_config = load(ROOT / f'configs/{family}/tasks/{task}.yaml')
    if kind != 'actor' and kind not in task_config.get('phases', {}):
        raise ValueError(f'{task} has no {kind} profile')
    config = merge(config, layer(task_config, kind, task))
    config = merge(config, layer(load(ROOT / f'configs/{family}/models/{model}.yaml'), kind, task))
    if kind == 'actor' and method != 'deltattg':
        config = merge(config, layer(load(ROOT / f'configs/actor/methods/{method}.yaml'), kind, task))
    for path in extra_configs:
        config = merge(config, layer(load(path), kind, task))
    if ablation:
        if kind != 'actor' or method != 'deltattg' or not re.fullmatch(r'[a-z0-9_]+', ablation):
            raise ValueError('Ablations apply only to actor training')
        spec = load(ROOT / f'configs/ablations/{ablation}.yaml')
        for entry in spec.get('hydra_overrides', []):
            key, value = entry.split('=', 1)
            config.setdefault('overrides', {})[key] = value
        if spec.get('meter_kind') == 'untrained_qwen3_4b_instruct_2507':
            config['env']['METER_MODEL'] = '${DELTATTG_STORAGE}/models/Qwen3-4B-Instruct-2507'
    env = dict(os.environ if environment is None else environment)
    for entry in settings:
        key, sep, value = entry.partition('=')
        if not sep or not re.fullmatch(r'[A-Z_][A-Z_0-9]*', key):
            raise ValueError('--set expects ENV_NAME=value')
        env[key] = value
    if action_penalty_mode is not None:
        env['USE_INVALID_ACTION_PENALTY'] = str(action_penalty_mode == 'on')
    env.update({'TASK': task, 'MODEL': model, 'KIND': kind, 'METHOD': method, 'ABLATION_SUFFIX': '-'+ablation if ablation else ''})
    env.setdefault('DELTATTG_ROOT', str(ROOT))
    env.setdefault('DELTATTG_STORAGE', env['DELTATTG_ROOT'])
    env.setdefault('DELTATTG_DATA', str(Path(env['DELTATTG_STORAGE']) / 'data'))
    resolver = Resolver(config.get('env', {}), env)
    resolved_env = {key: resolver.get(key) for key in config.get('env', {})}
    values = {key: resolver.text(scalar(value)) for key, value in config['overrides'].items()}
    for entry in overrides:
        if entry == '--':
            continue
        key, sep, value = entry.partition('=')
        if not sep or not re.fullmatch(r'\+{0,2}[\w.]+', key):
            raise ValueError(f'Expected a Hydra key=value override: {entry}')
        # A single final value per key, including + and ++ forms.
        for old in list(values):
            if old.lstrip('+') == key.lstrip('+'):
                del values[old]
        values[key] = value
    normalized = {key.lstrip('+'): value for key, value in values.items()}
    effective_penalty_mode = None
    if kind == 'actor':
        penalty_enabled = yaml.safe_load(normalized['actor_rollout_ref.actor.use_invalid_action_penalty'])
        if not isinstance(penalty_enabled, bool):
            raise ValueError('actor_rollout_ref.actor.use_invalid_action_penalty must be a boolean')
        effective_penalty_mode = 'on' if penalty_enabled else 'off'
        if action_penalty_mode is not None and effective_penalty_mode != action_penalty_mode:
            raise ValueError('--action-penalty-mode conflicts with the use_invalid_action_penalty override')
        expected_estimator = 'grpo' if method in ('deltattg', 'dapo') else method
        if normalized['algorithm.adv_estimator'] != expected_estimator:
            raise ValueError('Select the actor algorithm with --method')
        if normalized['credit_assignment.enabled'].lower() != str(method == 'deltattg').lower():
            raise ValueError('Baseline methods must disable DeltaTTG credit redistribution')
    locked = dict(LOCKED) if kind != 'meter_sft' else {}
    for key, expected in locked.items():
        if normalized.get(key, '').lower() != expected:
            raise ValueError(f'The unified recipe requires {key}={expected}')
    allowed_credit_keys = load(ROOT / 'configs/actor/deltattg.yaml')['credit_assignment']
    for key in normalized:
        if key.startswith('credit_assignment.') and key.split('.', 1)[1] not in allowed_credit_keys:
            raise ValueError(f'Unsupported credit assignment setting: {key}')
    command = [resolver.text(scalar(value)) for value in config['command']]
    if command[0] == 'python':
        command[0] = sys.executable
    def hydra_value(value):
        # Keep list/bool/number syntax; quote filesystem paths with Hydra delimiters.
        if value.startswith(('/', './', '../')) and any(c in value for c in ' ,={}[]'):
            return json.dumps(value)
        return value
    command += [key+'='+hydra_value(value) for key, value in values.items()]
    prepare = [[resolver.text(scalar(value)) for value in command] for command in config.get('prepare', [])]
    for command_item in prepare:
        if command_item[0] == 'python':
            command_item[0] = sys.executable
    runtime_env = {**env, **resolved_env}
    runtime_env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for key in config.get('unset_env', []):
        runtime_env.pop(key, None)
    return {
        'kind': kind, 'task': task, 'model': model, 'method': method, 'ablation': ablation,
        'action_penalty_mode': effective_penalty_mode,
        'command': command, 'prepare': prepare, 'overrides': values,
        'environment': resolved_env,
        'required_paths': [normalized.get(key, '') for key in config.get('required_paths', [])],
        'required_assets': [resolver.text(path) for path in config.get('required_assets', [])],
        'output_dir': normalized[config['output_key']],
    }, runtime_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=KINDS)
    parser.add_argument('--task', required=True, choices=('alfworld', 'papersearchqa'))
    parser.add_argument('--method', choices=METHODS, default='deltattg')
    parser.add_argument('--action-penalty-mode', choices=('off', 'on'),
                        help='Actor invalid-action penalty: off by default; on subtracts the configured penalty per invalid action')
    parser.add_argument('--model')
    parser.add_argument('--config', type=Path, action='append', default=[])
    parser.add_argument('--set', dest='settings', action='append', default=[])
    parser.add_argument('--ablation')
    parser.add_argument('--dry-run', action='store_true')
    args, overrides = parser.parse_known_args()
    model = args.model or ('qwen3_4b' if args.kind == 'actor' else 'qwen3_4b_instruct')
    try:
        plan, env = build_plan(args.kind, args.task, model, extra_configs=args.config,
            settings=args.settings, overrides=overrides, ablation=args.ablation, method=args.method,
            action_penalty_mode=args.action_penalty_mode)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return
        if any(not path or path == 'null' for path in plan['required_paths']):
            raise ValueError('Set all model/checkpoint and data paths before launching (see --dry-run)')
        # Preserve the allocated devices and their order when vLLM needs numeric IDs.
        devices = env.get('CUDA_VISIBLE_DEVICES', '')
        if 'GPU-' in devices:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
            mapping = [tuple(part.strip() for part in row.split(',', 1)) for row in rows.splitlines()]
            selected = []
            for assigned in devices.split(','):
                matches = [index for index, uuid in mapping if uuid.startswith(assigned.strip())]
                if len(matches) != 1:
                    raise ValueError(f'Cannot uniquely resolve allocated device: {assigned}')
                selected.extend(matches)
            env['CUDA_VISIBLE_DEVICES'] = ','.join(selected)
        for command in plan['prepare']:
            subprocess.run(command, check=True, cwd=ROOT, env=env)
        for value in plan['required_paths'] + plan['required_assets']:
            paths = yaml.safe_load(value) if value.startswith('[') else [value]
            for path in paths:
                if not Path(path).exists():
                    raise ValueError(f'Required asset does not exist: {path}')
        output = Path(plan['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        # Store the actual resolved launch plan; never dump the inherited environment.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (output / f'launch_config_{stamp}.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
        for key in ('WANDB_DIR',):
            if env.get(key):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
        os.chdir(ROOT)
        os.execvpe(plan['command'][0], plan['command'], env)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
