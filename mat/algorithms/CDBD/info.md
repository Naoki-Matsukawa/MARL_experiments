    Algorithm: Online Critic-Latent Belief Distillation

Input:
    Env
    Pretrained centralized teacher critic C_T
    Student recurrent actor π_S
    Projection head g
    Hyperparameters γ, λ_GAE, ε_PPO, λ_belief

Teacher critic interface:
    C_T(global_obs) -> (teacher_value, teacher_latent)

Student interface:
    RNN(local_obs_i, hidden_i) -> hidden_i
    Actor(hidden_i) -> action_dist_i
    Projection(hidden_i) -> student_latent_i


1. Freeze teacher critic C_T

    for param in C_T.parameters:
        param.requires_grad = False


2. Repeat for each training iteration

    Initialize rollout buffer B

    Reset environment
    Get local observations {o_i,0} and global observation O_0
    Initialize hidden states {h_i,0}

    for t = 0, ..., T-1:

        # ----- Student rollout -----

        for each agent i:

            h_i,t = RNN(o_i,t, h_i,t-1)

            dist_i,t = Actor(h_i,t)

            a_i,t ~ dist_i,t

            logp_i,t = log dist_i,t(a_i,t)

            z_S_i,t = Projection(h_i,t)

        Execute joint action a_t = (a_1,t, ..., a_N,t)

        Observe reward r_t, done_t
        Observe next local observations {o_i,t+1}
        Observe next global observation O_t+1


        # ----- Teacher annotation -----

        with no_grad:

            V_T,t, z_T,t = C_T(O_t)


        # ----- Store transition -----

        Store into B:
            local_obs_t        = {o_i,t}
            global_obs_t       = O_t
            hidden_t           = {h_i,t}
            actions_t          = {a_i,t}
            old_logprobs_t     = {logp_i,t}
            reward_t           = r_t
            done_t             = done_t
            teacher_value_t    = V_T,t
            teacher_latent_t   = z_T,t

        if done_t:
            reset hidden states
            reset environment
        else:
            o_i,t = o_i,t+1
            O_t = O_t+1


3. Compute advantages using teacher values

    For each timestep t in B:

        δ_t = r_t + γ * V_T,t+1 * (1 - done_t) - V_T,t

    Compute GAE:

        A_t = δ_t + γ * λ_GAE * (1 - done_t) * A_t+1

    Normalize advantages if needed.


4. Update student for K epochs

    for each minibatch M from B:

        for each sample t and agent i in M:

            # Recompute student outputs
            h_i,t = RNN(o_i,t, h_i,t-1)
            dist_i,t = Actor(h_i,t)
            new_logp_i,t = log dist_i,t(a_i,t)

            z_S_i,t = Projection(h_i,t)

            # PPO policy loss
            ratio_i,t = exp(new_logp_i,t - old_logprobs_i,t)

            L_PPO_i,t =
                - min(
                    ratio_i,t * A_t,
                    clip(ratio_i,t, 1 - ε_PPO, 1 + ε_PPO) * A_t
                )

            # Belief distillation loss
            # Important: project hidden state, not action
            L_belief_i,t =
                || z_S_i,t - stopgrad(z_T,t) ||^2

        L_policy = mean_i,t L_PPO_i,t

        L_belief = mean_i,t L_belief_i,t

        L_total = L_policy + λ_belief * L_belief

        Update student RNN, Actor, and Projection head
        Do not update teacher critic


5. Execution

    At test time, discard teacher critic and projection head.

    For each agent i:

        h_i,t = RNN(o_i,t, h_i,t-1)

        a_i,t ~ Actor(h_i,t)

    Use only local observations.