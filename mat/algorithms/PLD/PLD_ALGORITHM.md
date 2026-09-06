# PLD アルゴリズム実装メモ

このディレクトリの PLD 実装は、既存の MAT などで収集した rollout データを使い、分散実行可能なポリシーをオフラインで蒸留するための実装である。中心的な考え方は、全エージェントの共有観測から VAE でグローバルな潜在表現を作り、各エージェントのローカル観測だけからその潜在表現を予測させ、その予測潜在とローカル観測を使って教師行動を模倣する、という流れになっている。

## ファイル構成

- `VAE.py`
  - `MultiAgentVAE` を定義する。
  - 入力は全エージェント分の観測 `[B, N, obs_dim]`。
  - 出力は復元観測 `recon_x`、潜在分布の平均 `mu`、対数分散 `logvar`。
- `pld.py`
  - VAE 単体を `.npz` / `.pt` データから学習するためのスクリプト。
  - `GlobalObsDataset`、`load_global_obs`、`vae_loss`、CLI の `main()` を含む。
- `decentralized_policy.py`
  - ローカル観測から潜在表現を予測する `AgentObsEncoder`。
  - 観測と潜在表現から行動分布を出す `AgentPolicy`。
  - 潜在模倣損失と advantage weighted imitation loss を定義する。
- `pld_policy.py`
  - runner から使える policy wrapper。
  - `get_actions()` / `act()` / `get_values()` / `save()` / `restore()` を提供する。
- `pld_trainer.py`
  - rollout `.npz` を読み込んで、VAE と分散ポリシーを学習する trainer。
  - 実際の PLD 学習ロジックは主にここにある。

## 全体像

PLD はこの実装では次の 3 つのモデル部品に分かれる。

1. **グローバル VAE**
   - `share_obs` から全体状態を圧縮する。
   - `mu` が教師潜在表現として使われる。
2. **ローカル観測エンコーダ**
   - 各エージェントの `obs_i` だけから `z_i_pred` を予測する。
   - 実行時にグローバル観測を使わずに済むようにするための部品。
3. **分散ポリシー**
   - 各エージェントごとに `[obs_i, z_i_pred]` を入力し、行動分布を出す。
   - 離散行動では logits、連続行動では Gaussian の `mu` と `log_std` を出す。

学習時にはグローバル情報を使えるが、実行時には各エージェントのローカル観測だけを使う設計になっている。

## VAE

`MultiAgentVAE` は `[B, N, obs_dim]` の観測を受け取る。ここで `B` はバッチサイズ、`N` はエージェント数である。

エンコード処理は次の通り。

1. 各エージェントの観測を共有 MLP `agent_encoder` に通す。
   - 入力: `[B, N, obs_dim]`
   - 出力: `[B, N, agent_feature_dim]`
2. 全エージェントの特徴を flatten する。
   - 出力: `[B, N * agent_feature_dim]`
3. `mu_layer` と `logvar_layer` で潜在分布を作る。
   - `mu`: `[B, latent_dim]`
   - `logvar`: `[B, latent_dim]`

デコード処理は逆方向で、潜在変数 `z` から `[B, N, obs_dim]` を復元する。VAE 損失は標準的な

```text
VAE loss = reconstruction loss + KL(q(z|x) || N(0, I))
```

で、復元損失は設定により MSE または L1 が使われる。

## 分散ポリシー

`decentralized_policy.py` では、ローカル観測だけで動くモデルが定義されている。

`AgentObsEncoder` は各エージェントの観測を独立に処理する MLP である。

```text
obs_i -> z_i_pred
```

`AgentPolicy` はローカル観測と予測潜在を結合して行動分布を作る。

```text
[obs_i, z_i_pred] -> action distribution
```

離散行動の場合は `Categorical(logits)`、連続行動の場合は diagonal Gaussian として扱う。`PLDPolicy.get_actions()` では、利用可能行動 `available_actions` がある場合、選べない行動の logits を `-inf` にしてマスクする。

`PLDPolicy.get_values()` は常にゼロを返す。つまり、この PLD policy wrapper は critic を持たず、PPO のような value 更新をする設計ではない。runner インターフェースに合わせるために value 形状だけ返している。

## 学習データ

`PLDTrainer` は `--pld_dataset_dir` 以下の `.npz` ファイルをすべて読む。必須キーは次の 3 つ。

- `obs`
- `actions`
- `advantages`

任意キーとして以下も使われる。

- `share_obs`
- `active_masks`
- `available_actions`

`share_obs` がない場合は `obs` が代用される。データ形状は `[T, R, N, D]`、`[C, T, R, N, D]`、`[M, N, D]` を受け付け、内部では `[M, N, D]` に flatten される。

このリポジトリにある `dataset/smac_eval_rollout_0_steps_0.npz` は、確認時点では以下のような shape を持っている。

```text
obs:               (212073, 1, 8, 268)
share_obs:         (212073, 1, 8, 318)
actions:           (212073, 1, 8, 1)
advantages:        (212073, 1, 8, 1)
available_actions: (212073, 1, 8, 15)
```

`PLDTrainer._load_dataset()` はこの 4 次元形式を `[212073, 8, D]` のように潰して使う。

## 損失関数

PLD の policy 学習で使われる損失は主に 2 つである。

### 1. 潜在模倣損失

VAE の `mu` を教師潜在として使う。

```text
share_obs -> VAE -> mu
obs_i -> AgentObsEncoder -> z_i_pred
```

`mu` は `[B, latent_dim]` なので、各エージェントに複製して `[B, N, latent_dim]` にする。その上で `z_i_pred` と比較する。

```text
L_latent = mean((z_i_pred - mu)^2)
```

実装上は `latent_mimic_loss()` が MSE または cosine loss に対応しているが、`PLDTrainer` からはデフォルトの MSE で呼ばれている。

### 2. Advantage weighted imitation loss

教師行動 `actions` を模倣する損失である。ただし各サンプルは advantage で重み付けされる。

離散行動では cross entropy:

```text
L_policy = CE(policy_logits, teacher_action) * weight(advantage)
```

連続行動では Gaussian negative log likelihood:

```text
L_policy = -log pi(a_teacher | obs, z_pred) * weight(advantage)
```

重みは `--pld_use_exp_weight` により変わる。

- `False`: `max(advantage, 0)` を使う。
- `True`: `exp(clamp(advantage / temperature, -20, 20))` を使う。

最終的な policy 側の損失は次の形。

```text
total_loss = pld_latent_coef * L_latent + pld_policy_coef * L_policy
```

`pld_vae_coef` も設定値として存在するが、現在の `PLDTrainer.train()` では policy 学習時の `total_loss` には使われていない。

## 学習フロー

`PLDTrainer.train(buffer)` は引数として `buffer` を受け取るが、現在の実装では trainer 内で読み込んだ offline dataset を使い、渡された buffer は使わない。

処理の流れは次の通り。

1. `pld_dataset_dir` から `.npz` を読み込む。
2. `pld_vae_path` が存在する場合、その checkpoint を VAE にロードし、`freeze_vae=True` にする。
3. VAE checkpoint がない場合、`_train_vae_only()` で `share_obs` の再構成を学習する。
4. VAE の `mu` を教師潜在として生成する。
5. `AgentObsEncoder(obs)` で各エージェントの予測潜在 `z_pred` を作る。
6. `latent_mimic_loss(z_pred, mu)` を計算する。
7. `AgentPolicy(obs, z_pred)` で行動分布を作る。
8. `advantage_weighted_imitation_loss()` で教師行動を模倣する。
9. `policy.optimizer.step()` でローカルエンコーダとポリシーを更新する。

VAE を学習する場合は先に VAE のみを学習し、その後は `torch.no_grad()` で VAE を使って教師潜在を作る。policy 更新時に VAE は更新されない。

## 実行時の挙動

`PLDPolicy.get_actions()` は runner から呼ばれる実行用 API である。

1. `obs` を `[batch, num_agents, obs_dim]` に reshape する。
2. `pld_disable_obs_encoder` が `False` なら `AgentObsEncoder` で `z_pred` を作る。
3. `pld_disable_obs_encoder` が `True` なら `z_pred` はゼロベクトルになる。
4. `AgentPolicy(obs, z_pred)` で行動分布を作る。
5. 離散行動では `Categorical` から sample、または deterministic 時に argmax を取る。
6. 連続行動では Gaussian から `rsample()`、または deterministic 時に平均 `mu` を使う。

このため、実行時に VAE は使われない。VAE は学習時に教師潜在を作るためのモデルである。

## 主要ハイパーパラメータ

`mat/config.py` で定義されている PLD 関連引数は以下である。

- `--pld_latent_dim`: VAE とローカルエンコーダの潜在次元。
- `--pld_agent_feature_dim`: VAE 内で各エージェント観測を埋め込む特徴次元。
- `--pld_hidden_dim`: ローカルエンコーダと policy MLP の hidden 次元。
- `--pld_num_layers`: ローカルエンコーダと policy MLP の層数。
- `--pld_use_layernorm`: MLP に LayerNorm を入れる。
- `--pld_lr`: PLD モジュールの learning rate。
- `--pld_batch_size`: offline dataset の minibatch size。
- `--pld_epochs`: 1 update あたりの epoch 数。
- `--pld_recon_loss`: VAE 復元損失。`mse` または `l1`。
- `--pld_latent_coef`: 潜在模倣損失の重み。
- `--pld_policy_coef`: 行動模倣損失の重み。
- `--pld_use_exp_weight`: advantage 重みに指数関数を使う。
- `--pld_temperature`: 指数重みの温度。
- `--pld_dataset_dir`: offline rollout `.npz` の置き場。
- `--pld_updates`: PLD update 回数。
- `--pld_vae_path`: 事前学習済み VAE checkpoint の path。
- `--pld_freeze_vae`: VAE を固定する指定。
- `--pld_disable_obs_encoder`: ローカル観測エンコーダを無効化し、ゼロ潜在を使う。

## 注意点

- `PLDTrainer` は offline dataset を直接使うため、通常の PPO/MAT のように rollout buffer から on-policy 更新する実装ではない。
- `PLDPolicy` は critic を持たず、value はゼロを返す。value loss を使うアルゴリズムではなく、runner 互換のための戻り値である。
- `pld_vae_coef` は config にあるが、現在の `PLDTrainer.train()` の最終 policy loss には入っていない。
- `freeze_vae` は checkpoint ロード時に `True` にされるが、policy 学習側ではそもそも VAE を `torch.no_grad()` で使っているため、policy 更新時には VAE は更新されない。
- `mat/runner/shared/smac_runner.py` の `PLDSMACHRunner` には `run()` が 2 回定義されている。Python では後に定義された `run()` が有効になるため、先に書かれている「offline PLD training runner」の `run()` は上書きされている。現在のままだと、意図した offline 専用 update loop が使われていない可能性がある。
- `mat/algorithms/PLD/pld.py` の `from VAE import MultiAgentVAE` は、パッケージ実行時には import path の問題が出る可能性がある。他ファイルと同じく `from mat.algorithms.PLD.VAE import MultiAgentVAE` の方が安定する。

## まとめ

この PLD 実装は、グローバル情報を持つ教師データから、ローカル観測だけで動ける分散ポリシーを作るための蒸留アルゴリズムである。VAE は全体観測を圧縮して教師潜在を作り、各エージェントは自分の観測からその潜在を近似する。最終的な policy は、予測潜在とローカル観測を使って教師行動を advantage weighted imitation で模倣する。

