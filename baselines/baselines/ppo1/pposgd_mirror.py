from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    pos_rews = np.zeros(horizon, 'float32')
    neg_pens = np.zeros(horizon, 'float32')
    avg_vels = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "pos_rews" : pos_rews, "neg_pens": neg_pens, "avg_vels":avg_vels}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, envinfo = env.step(ac)
        if "pos_rew" in envinfo:
            pos_rews[i] = envinfo["pos_rew"]
        if "neg_pen" in envinfo:
            neg_pens[i] = envinfo["neg_pen"]
        if "avg_vel" in envinfo:
            avg_vels[i] = envinfo["avg_vel"]
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            broke = False
            if 'broke_sim' in envinfo:
                if envinfo['broke_sim']:
                    broke = True
            if not broke:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
            else:
                t -= (cur_ep_len+1)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        sym_loss_weight = 0.0,
        return_threshold = None, # termiante learning if reaches return_threshold
        op_after_init = None,
        init_policy_params = None,
        policy_scope=None,
        max_threshold=None,
        positive_rew_enforce = False,
        reward_drop_bound = None,
        min_iters = 0,
        ref_policy_params = None,
         rollout_length_thershold = None 
        ):

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    if policy_scope is None:
        pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
        oldpi = policy_func("oldpi", ob_space, ac_space)  # Network for old policy
    else:
        pi = policy_func(policy_scope, ob_space, ac_space)  # Construct network for new policy
        oldpi = policy_func("old"+policy_scope, ob_space, ac_space)  # Network for old policy

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    sym_loss = sym_loss_weight * U.mean(tf.square(pi.mean - pi.mirrored_mean))    # mirror symmetric loss
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) + sym_loss # PPO's pessimistic surrogate (L^CLIP)

    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent, sym_loss]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent", "sym_loss"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()

    if init_policy_params is not None:
        cur_scope = pi.get_variables()[0].name[0:pi.get_variables()[0].name.find('/')]
        orig_scope = list(init_policy_params.keys())[0][0:list(init_policy_params.keys())[0].find('/')]
        for i in range(len(pi.get_variables())):
            assign_op = pi.get_variables()[i].assign(init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            U.get_session().run(assign_op)
            assign_op = oldpi.get_variables()[i].assign(init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            U.get_session().run(assign_op)

    if ref_policy_params is not None:
        ref_pi = policy_func("ref_pi", ob_space, ac_space)
        cur_scope = ref_pi.get_variables()[0].name[0:ref_pi.get_variables()[0].name.find('/')]
        orig_scope = list(ref_policy_params.keys())[0][0:list(ref_policy_params.keys())[0].find('/')]
        for i in range(len(ref_pi.get_variables())):
            assign_op = ref_pi.get_variables()[i].assign(
                ref_policy_params[ref_pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            U.get_session().run(assign_op)
        env.env.env.ref_policy = ref_pi

    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    max_thres_satisfied = max_threshold is None
    adjust_ratio = 0.0
    prev_avg_rew = -1000000
    revert_parameters = {}
    variables = pi.get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        revert_parameters[variables[i].name] = cur_val
    revert_data = [0, 0, 0]
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()

        if reward_drop_bound is not None:
            lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            revert_iteration = False
            if np.mean(
                    rewbuffer) < prev_avg_rew - reward_drop_bound:  # detect significant drop in performance, revert to previous iteration
                print("Revert Iteration!!!!!")
                revert_iteration = True
            else:
                prev_avg_rew = np.mean(rewbuffer)
            logger.record_tabular("Revert Rew", prev_avg_rew)
            if revert_iteration:  # revert iteration
                for i in range(len(pi.get_variables())):
                    assign_op = pi.get_variables()[i].assign(revert_parameters[pi.get_variables()[i].name])
                    U.get_session().run(assign_op)
                episodes_so_far = revert_data[0]
                timesteps_so_far = revert_data[1]
                iters_so_far = revert_data[2]
                continue
            else:
                variables = pi.get_variables()
                for i in range(len(variables)):
                    cur_val = variables[i].eval()
                    revert_parameters[variables[i].name] = np.copy(cur_val)
                revert_data[0] = episodes_so_far
                revert_data[1] = timesteps_so_far
                revert_data[2] = iters_so_far

        if positive_rew_enforce:
            rewlocal = (seg["pos_rews"], seg["neg_pens"], seg["rew"])  # local values
            listofrews = MPI.COMM_WORLD.allgather(rewlocal)  # list of tuples
            pos_rews, neg_pens, rews = map(flatten_lists, zip(*listofrews))
            if np.mean(rews) < 0.0:
                #min_id = np.argmin(rews)
                #adjust_ratio = pos_rews[min_id]/np.abs(neg_pens[min_id])
                adjust_ratio = np.max([adjust_ratio, np.mean(pos_rews) / np.abs(np.mean(neg_pens))])
                for i in range(len(seg["rew"])):
                    if np.abs(seg["rew"][i] - seg["pos_rews"][i] - seg["neg_pens"][i]) > 1e-5:
                        print(seg["rew"][i], seg["pos_rews"][i] , seg["neg_pens"][i])
                        print('Reward wrong!')
                        abc
                    seg["rew"][i] = seg["pos_rews"][i] + seg["neg_pens"][i] * adjust_ratio
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))
        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        if reward_drop_bound is None:
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("Iter", iters_so_far)
        if positive_rew_enforce:
            if adjust_ratio is not None:
                logger.record_tabular("RewardAdjustRatio", adjust_ratio)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        if max_threshold is not None:
            print('Current max return: ', np.max(rewbuffer))
            if np.max(rewbuffer) > max_threshold:
                max_thres_satisfied = True
            else:
                max_thres_satisfied = False

        return_threshold_satisfied = True
        if return_threshold is not None:
            if not(np.mean(rewbuffer) > return_threshold and iters_so_far > min_iters):
                return_threshold_satisfied = False
        rollout_length_thershold_satisfied = True
        if rollout_length_thershold is not None:
            rewlocal = (seg["avg_vels"], seg["rew"])  # local values
            listofrews = MPI.COMM_WORLD.allgather(rewlocal)  # list of tuples
            avg_vels, rews = map(flatten_lists, zip(*listofrews))
            if not(np.mean(lenbuffer) > rollout_length_thershold and np.mean(avg_vels) > 0.5 * env.env.env.final_tv):
                rollout_length_thershold_satisfied = False
        if rollout_length_thershold is not None or return_threshold is not None:
            if rollout_length_thershold_satisfied and return_threshold_satisfied:
                break

    return pi, np.mean(rewbuffer)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
