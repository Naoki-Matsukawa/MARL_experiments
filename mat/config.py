import argparse
# import dotenv
import os



def get_config():
    """
    The configuration parser for hyper-parameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyper-parameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["mat", "mat_dec"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """

 


    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mat', choices=["mat", "mat_dec", "mat_encoder", "mat_decoder", "mat_gru", "r_mappo", "pld"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default="matsukawa-naoki555-university-of-tokyo",help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--sight_range", type=float, default=9.0,
                        help="Local observation sight range for SMAC-style environments.")
    parser.add_argument("--mask_attack_by_sight", action='store_true', default=False,
                        help="Limit SMAC attack availability by sight range as well as shooting range.")
    parser.add_argument("--randomize_enemy_position", action='store_true', default=False,
                        help="Randomize initial enemy positions in supported SMAC environments.")
    parser.add_argument("--enemy_position_jitter", type=float, default=0.0,
                        help="Maximum initial enemy position jitter radius.")
    parser.add_argument("--enemy_position_jitter_mode", type=str, default="group", choices=["group", "unit"],
                        help="Enemy jitter mode: group translates all enemies together; unit jitters each enemy.")
    parser.add_argument("--enemy_position_jitter_attempts", type=int, default=20,
                        help="Number of attempts to sample valid enemy jitter positions.")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=2,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')

    # PLD parameters
    parser.add_argument("--pld_latent_dim", type=int, default=32, help="Latent dimension for PLD VAE/encoder.")
    parser.add_argument("--pld_agent_feature_dim", type=int, default=64, help="Agent feature dim for PLD VAE.")
    parser.add_argument("--pld_hidden_dim", type=int, default=256, help="Hidden dim for PLD policy/encoder.")
    parser.add_argument("--pld_num_layers", type=int, default=2, help="Number of layers for PLD policy/encoder.")
    parser.add_argument("--pld_use_layernorm", action='store_true', default=False, help="Use LayerNorm in PLD MLPs.")
    parser.add_argument("--pld_lr", type=float, default=3e-4, help="Learning rate for PLD modules.")
    parser.add_argument("--pld_batch_size", type=int, default=512, help="Minibatch size for PLD training.")
    parser.add_argument("--pld_epochs", type=int, default=10, help="Epochs per update for PLD training.")
    parser.add_argument("--pld_recon_loss", type=str, default="mse", choices=["mse", "l1"], help="VAE recon loss.")
    parser.add_argument("--pld_vae_coef", type=float, default=1.0, help="Weight for VAE loss.")
    parser.add_argument("--pld_latent_coef", type=float, default=1.0, help="Weight for latent mimic loss.")
    parser.add_argument("--pld_policy_coef", type=float, default=1.0, help="Weight for policy imitation loss.")
    parser.add_argument("--pld_use_exp_weight", action='store_true', default=False, help="Use exp weighting for advantages.")
    parser.add_argument("--pld_temperature", type=float, default=1.0, help="Temperature for advantage weighting.")
    parser.add_argument("--pld_log_std_min", type=float, default=-5.0, help="Min log std for continuous PLD policy.")
    parser.add_argument("--pld_log_std_max", type=float, default=2.0, help="Max log std for continuous PLD policy.")
    parser.add_argument("--pld_dataset_dir", type=str,
                        default="mat/algorithms/PLD/dataset", help="Directory containing MAT npz rollouts for PLD.")
    parser.add_argument("--pld_updates", type=int, default=1, help="Number of offline PLD updates per run.")
    parser.add_argument("--pld_vae_path", type=str, default=None, help="Path to a pretrained VAE checkpoint.")
    parser.add_argument("--pld_freeze_vae", action='store_true', default=False, help="Freeze VAE during PLD training.")
    parser.add_argument("--pld_disable_obs_encoder", action='store_true', default=False,
                        help="Disable PLD obs encoder and use zero latent instead.")
    
    
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=100, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")
    parser.add_argument("--save_debug_render", action='store_true', default=False,
                        help="Save a headless top-down debug render for supported environments.")
    parser.add_argument("--debug_render_format", type=str, default="gif", choices=["gif"],
                        help="Debug render output format.")
    parser.add_argument("--debug_render_episodes", type=int, default=1,
                        help="Number of eval episodes to capture for debug rendering.")
    parser.add_argument("--debug_render_interval", type=int, default=1,
                        help="Capture one debug render frame every N env steps.")
    parser.add_argument("--debug_render_fps", type=int, default=8,
                        help="Frames per second for saved debug render GIFs.")
    parser.add_argument("--debug_render_show_sight", action='store_true', default=False,
                        help="Draw agent sight circles in debug renders.")
    parser.add_argument("--debug_render_dir", type=str, default="debug_renders",
                        help="Directory for debug render outputs, relative to the run directory unless absolute.")
    parser.add_argument("--save_replay", action='store_true', default=False,
                        help="Save StarCraft II replays for supported environments.")
    parser.add_argument("--replay_dir", type=str, default="",
                        help="StarCraft II replay output directory.")
    parser.add_argument("--replay_prefix", type=str, default="",
                        help="StarCraft II replay filename prefix.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")


    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)


    # noise rate to obs
    parser.add_argument("--final_noise_rate", type=float, default=0.0, help="the noise rate to obs, default 0.0")
    parser.add_argument("--eval_noise_rate", type=float, default=0.0, help="the noise rate to obs during evaluation, default 0.0")
    parser.add_argument("--gradual" , action='store_true', default=False, help="whether to use gradual noise rate to obs, default False")

    parser.add_argument("--noise_std", type=float, default=0, help="the std of noise added to eval obs, default 0")

    # distillation parameters
    parser.add_argument("--distillation", action="store_true", default=False, help="Enable knowledge distillation to train student policies.")
    parser.add_argument("--student_kl_coef", type=float, default=1.0, help="Coefficient for student KL divergence loss during distillation.")
    parser.add_argument("--student_rl_coef", type=float, default=0.1, help="Coefficient for student policy gradient loss during distillation.")
    parser.add_argument("--student_rl_coef_start", type=float, default=None, help="Starting coefficient for student policy gradient schedule (defaults to student_rl_coef unless linear scheduling is enabled).")
    parser.add_argument("--student_rl_linear_schedule", action="store_true", default=False, help="Linearly increase student_rl_coef from start to target over the training steps.")
    parser.add_argument("--student_rl_schedule_steps", type=float, default=None, help="Number of env steps to ramp student_rl_coef; defaults to num_env_steps when not set.")
    parser.add_argument("--student_value_coef", type=float, default=0.5, help="Coefficient for student value function loss during distillation.")
    parser.add_argument("--student_aux_value_coef", type=float, default=1, help="Coefficient for student auxiliary value function loss during distillation.")
    parser.add_argument("--student_clip_param", type=float, default=0.2, help="Clipping parameter for student PPO-style loss.")
    parser.add_argument("--student_log_std_init", type=float, default=0.0, help="Initial log std for continuous student policies.")
    parser.add_argument("--student_use_mlp", action="store_true", default=False, help="Use MLP in student policy network.")
    return parser
