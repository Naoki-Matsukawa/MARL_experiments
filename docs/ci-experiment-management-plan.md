# CI導入・実験管理整備の検討メモ

作成日: 2026-08-09
対象: Multi-Agent-Transformer リポジトリ

## 進捗（2026-08-09）

フェーズ1（環境ごとのDockerイメージ整備）を `feature/ci-docker-infra` ブランチで実施済み。

- SMAC / football / mpe / vmas / robotarium / ma_mujoco の6環境: ビルド成功、
  実際に短時間の学習を回して動作確認済み（過程で見つかった実アプリのバグ
  `n_agents`未設定・`wandb.log`のuse_wandb未ガード・`eval_faulty_node`の
  Noneハンドリングも合わせて修正）。
- jaxmarl_robotarium: ビルド・env構築までは成功。学習ループの実行時に
  vendored `3rdparty/JaxMARL-Robotarium` 内のbarrier-certificate計算で
  `cuSolver internal error` が再現する（GPU/メモリを変えても再現、CPU
  フォールバックは学習スクリプト側で禁止されている）。原因はこのリポジトリの
  コードではなく jaxlib 0.4.38 と本クラスタのGPU/ドライバ組み合わせ側の
  可能性が高く、今回は未解決のまま。詳細は `docker/README.md` を参照。
- marbler / dexteroushandenvs: 非公開パッチ・IsaacGymのEULA制約のため保留。
- 依存管理は当初案のrequirements.txtではなく、環境ごとの
  `pyproject.toml` + `uv.lock`（`docker/<env>/`）に統一。
- `3rdparty/MARBLER`, `robotarium_python_simulator`, `JaxMARL-Robotarium`
  はgit submodule化済み。

## 0. これは何か

「複数のシミュレーション環境 × GPUクラスタでの実験を、push契機で自動実行できるようにしたい」という要望について、
(a) 現状の認識を実装済みコードから確認し、
(b) 要望を機能単位に分解し、
(c) 実装コストとROIの見積りに基づく導入順序を提案する。

このメモは提案であり、まだ何も実装していない。着手前にセクション5の要確認事項をすり合わせる。

## 1. 現状認識（コードから確認できた事実）

### 1.1 実行環境・依存関係
- シミュレーション環境は `mat/envs/` 配下に少なくとも8種類同居している: `starcraft2` (SMAC), `football`, `mpe`, `vmas`, `robotarium`, `marbler`, `jaxmarl_robotarium`, `ma_mujoco`, `dexteroushandenvs`。
- 依存関係はリポジトリ直下の `pyproject.toml` / `requirements.txt` に**全環境ぶんが1枚岩で固定**されている（例: `torch==1.10.2`, CUDA 10.2世代）。環境ごとの依存分離は無い。
- 例外的に SMAC 用は `Dockerfile.smac`（PyTorch 2.3.1 / CUDA 12.1ベース、`docker/requirements-smac.txt` を使用）で新しい依存を分離している。football 用にも `dockerfile` / `football/Dockerfile` があるが、こちらは手動運用（自動ビルド・自動プッシュの仕組みはなし）。
- 他の環境（mpe, vmas, robotarium, marbler, jaxmarl_robotarium, ma_mujoco, dexteroushandenvs）には Dockerfile が存在しない。

### 1.2 実験の起動・GPU割当（想定より進んでいた点）
- `mat/scripts/run_yaml.py` という自作のYAMLベースランチャーが既にあり、以下が実装済み:
  - `extends` によるYAML継承、`presets`/`base`/`matrix`/`seeds` によるパラメータスイープ生成
  - `resources.gpu_ids` / `resources.num_gpus` によるGPUプール指定、`cuda_visible_devices: auto` 指定時は `nvidia-smi` の空きメモリ順に自動割当（`detect_gpu_pool` / `configured_gpu_pool`）
  - `--dry-run` でコマンド確認、`--index` でSlurm array taskごとの実行対象を絞り込み
- Slurm連携もすでに二系統ある:
  - `run_yaml_slurm.sh`（`uv run python` 直接実行、`dgx-a100-40g` パーティション想定）
  - `run_yaml_apptainer_slurm.sh`（Docker imageを `apptainer build` で `.sif` に変換して `apptainer exec --nv` 実行、`dgx-a100-80g` パーティション想定）
  - `submit_yaml_array.sh` がYAMLの `slurm:` / `resources:` セクションを読み、Slurm array jobとして一括投入
- つまり「YAMLでconfigを書く」「Slurmに投げる」「GPUをある程度自動選択する」という土台は**すでに存在する**。ゼロから作る必要はない。

### 1.3 実行管理・記録
- 実験管理は `wandb`（`mat/runner/shared/base_runner.py` で `wandb.init` / `wandb.log`）＋タイムスタンプ名の出力ファイル（`dump/`, `results/` 等）が中心。
- git運用は現状 `main` ブランチ中心で、実験ごとのコード分岐は基本コミット追加・YAML追加で同一ブランチ上に積み増している（複数ブランチ自体は存在するが、実験バリエーション管理には使われていない）。

### 1.4 CI/CDの現状
- `.github/` ディレクトリは存在せず、GitHub Actions等のCIは未導入。
- GitHubリモート（`git@github.com:eto56/mat.git`）は存在するので、push契機のワークフローは技術的に組める。
- push→実行の自動化は無く、現状は「ラップトップからsshでクラスタに入り、`uv run` や `sbatch` を手動実行」という運用と理解している。

### 1.5 計算資源（ユーザー確認済み）
- Slurmクラスタ（`dgx-a100-40g` / `dgx-a100-80g` などmil研と見られるパーティション）に加えて、Slurm管理外の単体GPUマシンにもsshで直接アクセスして使っている。→ **ジョブの行き先が「Slurm」と「非Slurmのsshホスト」の2系統ある**前提で設計する必要がある。

## 2. 要望の分解

| # | 要望 | 現状 |
|---|---|---|
| 1 | pushしたコードのバージョンを、新しく書いたconfig通りに自動実行 | ランチャー（`run_yaml.py`）はあるが、起動はすべて手動 |
| 2 | 適切なGPUクラスタ内の適切なGPU単位に自動割当 | 単一ノード内のGPU自動選択は実装済み。クラスタ横断（Slurm / 非Slurmホストの使い分け）の自動振り分けは無い |
| 3 | Slurmにも対応 | 実装済み（`run_yaml_slurm.sh`, `run_yaml_apptainer_slurm.sh`, `submit_yaml_array.sh`） |
| 4 | 環境ごとにDockerイメージを用意し、その中で研究コードを実行 | SMAC/footballのみ部分実装。イメージのビルド・配布は手動 |
| 5 | 毎回イメージをビルドしない | 未実装（レジストリへのpush/pull・キャッシュの仕組みが無い） |

## 3. 実装コストに見合うか（総評）

- **要望3・一部の要望2（単一ノード内GPU割当）はすでにほぼ実装済み**。新規実装コストはドキュメント整理程度で、追加投資はほぼ不要。
- **要望4・5（環境ごとのDockerイメージ＋ビルドキャッシュ）は投資対効果が高い**。SMACの構成（Dockerfile → GHCR等へpush → Slurm側はApptainerでpull&sif化）を他環境にも横展開するだけなので、既存パターンの繰り返し適用で済む。CIで「変更があった環境のイメージだけビルド」すれば手動ビルドの手間もほぼ消える。
- **要望1（push→自動実行）は費用対効果を要検討**。GitHub Actions の self-hosted runner をGPUマシン（Slurm / 非Slurmホスト双方）に常駐させる構成は技術的には可能だが、
  - 大学の共有計算機ポリシー上、runnerプロセスを常駐させて良いかは要確認（他ユーザーとの共有環境のため）。
  - 研究フェーズ（config試行錯誤が頻繁）では「pushのたびに数時間かかるジョブが自動起動する」のはリスクの方が大きい。
  - → **今回は「特定ブランチ/タグへのpushのみ起動」という設計にする（ユーザー確認済み）**。これなら普段の開発pushとは分離でき、誤爆リスクを抑えつつ自動化のメリットを得られる。
- **要望2のうち「クラスタ横断の自動振り分け」（Slurm vs 非SlurmのGPUホストのどちらに投げるか）は新規実装が必要**。ここは既存のSlurmスケジューラに任せられない部分なので、YAML側に `target: slurm | ssh-host` のような明示指定を持たせる薄い仕組みで十分（自動最適配置のような高度なスケジューラを自作するのは過剰投資）。

**結論**: フルスコープを一度に作るのは過剰。「Docker化＋イメージキャッシュ（要望4・5）」から着手し、「特定ブランチpushでの自動実行（要望1・2の一部）」をその上に薄く載せる、という順序が最もコストに見合う。

## 4. 実装順序の提案

### フェーズ0: 既存資産の棚卸し・ドキュメント化（低コスト）
- `run_yaml.py` / Slurmスクリプト群 / `docker/README.md` の使い方を1つの `docs/` にまとめる（このファイルの隣に置く想定）。
- 目的: 次のフェーズに入る前に「再発明」を避ける。

### フェーズ1: 環境ごとのDockerイメージ整備 + レジストリキャッシュ（ROI最大）
- 各シミュレーション環境用に `docker/Dockerfile.<env>` と `docker/requirements-<env>.txt` を用意（`Dockerfile.smac` のパターンを横展開）。
- GitHub Actions で、変更のあった `docker/Dockerfile.<env>` に対応する環境のイメージだけを `docker buildx` + layer cache付きでビルドし、GHCR（`ghcr.io/eto56/mat-<env>`）にpush。
- Slurm側は既存の `apptainer build ... docker-daemon://` の代わりに `docker://ghcr.io/...` から直接pullする形に変更し、手元でのdocker buildを不要にする。
- **これで「毎回イメージをビルドしない」が達成される**（変更が無ければCIも走らない／pullはレイヤキャッシュが効く）。

### フェーズ2: 依存関係の環境ごと分離
- ルート直下の一枚岩 `pyproject.toml` から、環境ごとの `requirements-<env>.txt` への切り出しをフェーズ1と合わせて進める（フェーズ1のDockerfileが要求するため、実質同時進行になる）。

### フェーズ3: 特定ブランチ/タグpushでの自動実行（要望1・2）
- 例: `experiment/*` ブランチや `run-*` タグへのpushをトリガーに、GitHub Actions が
  1. pushされたコミットのconfig差分（`mat/scripts/configs/**`）を検出
  2. YAMLの `target:`（`slurm` / 特定sshホスト名）に応じて `submit_yaml_array.sh`（Slurm）または ssh経由で `run_yaml.py`（非Slurmホスト）を起動
  - 通常の `main` へのpushには一切反応しないため、開発フローと実験実行フローが分離される。
- self-hosted runner常駐の可否（大学計算機ポリシー）は着手前に確認必須。不可なら「GitHub Actionsからsshでリモート起動するだけ」の軽量構成に倒す。

### フェーズ4: 実験管理の仕上げ
- wandbのproject/tag命名規則の統一、実行時のgit commit hashとconfigのwandb.config記録（再現性の担保）。
- 出力先をタイムスタンプ命名から `run_id`（wandb run id等）ベースに揃え、`dump/` 以下の散らかりを整理。

## 5. 着手前に確認したいこと

1. self-hosted GitHub Actions runnerを、mil研Slurmのログインノード／非SlurmのGPUホストに常駐させることが計算機利用ポリシー上問題ないか。
2. コンテナイメージの置き場所は GHCR（`ghcr.io`）でよいか、大学内に別途プライベートレジストリがあるか。
3. フェーズ3のトリガーとするブランチ/タグの命名規則（例: `experiment/*` プレフィックス、`run-YYYYMMDD-*` タグなど）の希望有無。
4. 非SlurmのGPUホストは何台・何GPU構成か（フェーズ3の `target:` 設計に必要）。

## 6. まとめ

- Slurm対応・単一ノード内GPU自動選択（要望2の一部・要望3）は**すでに実装済み**であり、追加投資は最小限でよい。
- 「環境ごとのDockerイメージ化＋ビルドキャッシュ」（要望4・5）が最もROIが高く、既存のSMAC用構成を横展開するだけなので次に着手すべき。
- 「push即自動実行」（要望1）はフルオートだと研究フェーズでは事故りやすいため、**特定ブランチ/タグpush限定**というスコープに絞ることで、コストとリスクを抑えつつ導入する。
- クラスタ横断（Slurm / 非Slurmホスト）の振り分けだけは新規実装が必要だが、YAMLに `target` を明示させる程度の薄い仕組みで十分。
