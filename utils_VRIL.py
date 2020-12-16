import h5py, os, time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



def hfile_to_df(hfile, tbin='frame'):
    # sampling interval
    all_cols = []
    all_data = []
    condition_map = ['ad1s', 'adol50', 'ad50', 'ad100', 'online', 'ep1s', 'ol50']
    tL = hfile['out/mt'].shape[1]
    cond_data = hfile['out/designmat/condition']
    for k in ['condition', 'Y', 'goal', 'mt', 'subj', 'framerate']:
        if k in hfile['out']:
            entry = np.array(hfile[f'out/{k}'], dtype=np.float)
            if k == 'mt':
                entry = entry[1, :]
        else:
            entry = np.array(hfile[f'out/designmat/{k}'], dtype=np.float)
            if k == 'condition':
                all_data.append(entry)
                all_cols.append('condition_raw')
                entry = np.array([condition_map[int(tj) - 1] for tj in entry[0]])
        all_data.append(entry)
        exceptions = ['mt', 'condition']
        if (entry.shape[0] > 1) and (k not in exceptions):
            all_cols = all_cols + [k + '_theta', k + '_rho']
        else:
            all_cols.append(k)
        # print(k, len(all_cols), all_cols)
    first_df = pd.DataFrame(np.vstack(all_data).T, columns=all_cols)
    for c in all_cols:
        if c != 'condition':
            first_df[c] = first_df[c].astype(np.float)
    # Here implements skip downsampling
    v_real = np.array(hfile['out/designmat/v'], dtype=np.float)
    v_real[np.isnan(v_real)] = get_defaults()['default_theta']
    if tbin == 'frame':
        spoints = np.around(100 / 6 * np.arange(1, 19)).astype(np.int)
        tags_t = [f"v_theta_{sp}_frame" for sp in np.arange(1, 19)]
        tags_r = [f"v_rho_{sp}_frame" for sp in np.arange(1, 19)]
    else:
        tT = v_real.shape[0]
        spoints = np.arange(tbin, tT + 1, tbin)
        tags_t = [f"v_theta_{sp}_ms" for sp in spoints]
        tags_r = [f"v_rho_{sp}_ms" for sp in spoints]
    v_samp = v_real[spoints - 1, :, :]
    v_merge = np.vstack((v_samp[:, 0, :], v_samp[:, 1, :])).T
    vframe = pd.DataFrame(v_merge, columns=tags_t + tags_r)
    return pd.concat((first_df, vframe), axis=1)
    # return first_df, vframe


def trial_vector_time_lag(vec, t):
    """ Takes in vector and shift it by t (pad with False, "" or nan in according to data dtype)
    :param vec: input vector (number, str or bool)
    :param t: shift lag (integer)
    :return: oarr: np.ndarray: shifted array
    @test
    """
    if t == 0:
        return vec
    dtype = vec.dtype
    if np.issubdtype(dtype, np.bool_):
        oarr = np.zeros(len(vec), dtype=dtype)
    elif np.issubdtype(dtype, np.number):
        oarr = np.full(len(vec), np.nan, dtype=np.float)
    elif np.issubdtype(dtype, np.str_):
        oarr = np.full(len(vec), "", dtype=dtype)
    else:
        raise NotImplementedError(f"Unhandled dtype {dtype}")
    if t < 0:
        oarr[-t:] = vec[:t]
    else:
        oarr[:-t] = vec[t:]
    return oarr


def get_defaults():
    return {'default_theta': 0, 'default_rho': 0}


def df_lag_by_groups(df, group_args, lag_cols, lagnum=1):
    # only support backward lag so far
    if len(group_args) == 1:
        garg = group_args[0]
        for ig in np.unique(df[garg]):
            subdf = df[df[garg] == ig]
            for lc in lag_cols:
                for ln in range(1, lagnum + 1):
                    df.loc[df[garg] == ig, f'{lc}_{ln}lag'] = trial_vector_time_lag(
                        subdf[lc].values, -ln)
            df.loc[df[garg] == ig, 'trial'] = np.arange(len(subdf))
        return df
    else:
        garg = group_args[0]
        vals = [df_lag_by_groups(df[df[garg] == ig].copy(),
                                 group_args[1:], lag_cols,
                                 lagnum) for ig in np.unique(df[garg])]
        for vvv in vals:
            print(len(vvv), np.unique(vvv['condition']))
        # print(len(vals[0]))
        return pd.concat([df_lag_by_groups(df[df[garg] == ig].copy(),
                                           group_args[1:], lag_cols,
                                           lagnum) for ig in np.unique(df[garg])],
                         axis=0, ignore_index=True)


def take_slice_by_id(df, slice_n=20):
    # only support backward lag so far
    return pd.concat([df[df['subj'] == ig].iloc[-20:]
                      for ig in np.unique(df['subj'])], axis=0, ignore_index=True)


def df_train_test_split(df, frac=0.8):
    df = df.reset_index(drop=True)
    train_set = df.sample(frac=0.8, replace=False)
    test_set = df.drop(train_set.index)
    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)


def VRIL(dt=0.005, df=None, showplots=False, algorithm=[]):
    # Generates time series for the visual rotation implicit learning (VRIL) tasks,
    # will randomly generate iti
    # Here functions are autograd.numpy but should be the same
    # TUNABLES
    df = df.dropna().reset_index()
    mean_iti = 0  # Randomly sample not very important
    down_min = 0.02
    down_max = 0.1
    prop_noise = np.array([3, 5])  # std for theta and rho respectively

    def TTS(T):
        # Convert Time to Steps
        return int(round(T / dt))

    def add_ramp_to_final(start, end, vfinal, x, downtime=None, addnoise=False):
        # Modified from RHY in JazNets package
        # vfinal assumed to be 1darray in this case
        nv = len(vfinal)
        vfinal = vfinal.reshape((nv, 1))
        slope = vfinal / (end - start)
        x[start:end, :] = (slope * np.arange(end - start)).T
        real_end = end
        if downtime is not None:
            downsteps = TTS(downtime)
            # print(end, downsteps, down_time, len(x))
            assert end + downsteps <= len(x), 'OUT OF BOUNDS'
            x[end:(end + downsteps), :] = (vfinal - vfinal / downsteps * np.arange(downsteps)).T
            real_end = downsteps + end
        if addnoise:
            x[start:real_end] = x[start:real_end] + get_noise(real_end - start)
        return x

    def polar_to_xy(trho):
        theta, rho = trho
        return np.array([np.cos(theta * np.pi / 180) * rho,
                         np.sin(theta * np.pi / 180) * rho])

    # Hard Wired
    indx = np.random.randint(len(df))
    # print(len(df), indx)
    trial1_time = df.loc[indx, 'mt_1lag'] / 1000  # 0.3 # trial_duration
    trial2_time = df.loc[indx, 'mt'] / 1000
    trial1_steps = TTS(trial1_time)
    trial2_steps = TTS(trial2_time)
    iti_steps = TTS(mean_iti)
    total_steps = trial1_steps + trial2_steps + 2 * iti_steps
    # print(trial1_steps, trial2_steps, iti_steps, total_steps)
    # Default to frame operation
    lt_edges = np.arange(1, 19) * 0.1 / 6

    # down_time = np.random.random() * (down_max - down_min) + down_min
    down_time = None
    vclamp_mem = np.full((total_steps, 2), get_defaults()['default_theta'])
    prop_mem = np.zeros((total_steps, 2))
    motor_out = np.zeros((total_steps, 2))

    # Later change to movement time
    trial2_start = trial1_steps + iti_steps
    get_noise = lambda t: np.random.multivariate_normal(prop_noise, np.eye(2), t)
    lag1 = df.loc[indx, ['Y_theta_1lag', 'Y_rho_1lag']].values
    lag0 = df.loc[indx, ['Y_theta', 'Y_rho']].values
    # print(lag0, polar_to_xy(lag0))

    prop_mem = add_ramp_to_final(0, trial1_steps, lag1,
                                 prop_mem, down_time, True)
    motor_out = add_ramp_to_final(trial2_start, trial2_start + trial2_steps,
                                  lag0, motor_out, down_time, False)
    # start time, if time is less than or equal to the next bin edge, get its value
    # print(dt, trial1_steps, trial1_time, trial2_steps, trial2_time)
    ilt = 0
    for ts in range(trial1_steps):
        curr_time = min((ts + 1) * dt, 0.3)
        if curr_time <= lt_edges[ilt]:
            vclamp_mem[ts] = df.loc[indx, [f'v_theta_{ilt + 1}_frame',
                                           f'v_rho_{ilt + 1}_frame']].values
        else:
            # print(lt_edges[ilt], ilt, ts, (ts+1) * dt, curr_time)
            ilt += 1
            vclamp_mem[ts] = df.loc[indx, [f'v_theta_{ilt + 1}_frame',
                                           f'v_rho_{ilt + 1}_frame']].values

    targ_idx = np.arange(trial2_start, trial2_start + trial2_steps)
    targets = motor_out
    inputs = np.hstack((vclamp_mem, prop_mem))
    hints = targets / 2

    if showplots:
        plt.figure(figsize=(15, 15))
        plt.plot(inputs)
        plt.plot(targets)
        plt.plot(hints)
        plt.legend(['vclamp_theta', 'vclamp_rho',
                    'prop_theta', 'prop_rho',
                    'motor_theta', 'motor_rho', 'hints_0', 'hints_1'])
        plt.title(df.loc[indx, 'condition'])
        plt.show()

    inps_and_targs = {'inps': inputs, 'targs': targets, 'hints': hints,
                      'targ_idx': targ_idx, 'theta': [],
                      'condition': df.loc[indx, 'condition']}
    return inps_and_targs


def VRILTR(dt=0.005, df=None, showplots=False, algorithm=[]):
    # Generates time series for the visual rotation implicit learning (VRIL) tasks,
    # will randomly generate iti
    # Here functions are autograd.numpy but should be the same
    # TUNABLES
    df = df.dropna().reset_index()
    mean_iti = 0  # Randomly sample not very important
    down_min = 0.02
    down_max = 0.1
    prop_noise = np.array([3, 3])  # std for theta and rho respectively

    def TTS(T):
        # Convert Time to Steps
        return int(round(T / dt))

    def add_ramp_to_final(start, end, vfinal, x, downtime=None, addnoise=False):
        # Modified from RHY in JazNets package
        # vfinal assumed to be 1darray in this case
        nv = len(vfinal)
        vfinal = vfinal.reshape((nv, 1))
        slope = vfinal / (end - start)
        x[start:end, :] = (slope * np.arange(end - start)).T
        real_end = end
        if downtime is not None:
            downsteps = TTS(downtime)
            # print(end, downsteps, down_time, len(x))
            assert end + downsteps <= len(x), 'OUT OF BOUNDS'
            x[end:(end + downsteps), :] = (vfinal - vfinal / downsteps * np.arange(downsteps)).T
            real_end = downsteps + end
        if addnoise:
            x[start:real_end] = x[start:real_end] + get_noise(real_end - start)
        return x

    def polar_to_xy(trho):
        theta, rho = trho
        return np.array([np.cos(theta * np.pi / 180) * rho,
                         np.sin(theta * np.pi / 180) * rho])

    # Hard Wired
    indx = np.random.randint(len(df))
    # print(len(df), indx)
    trial1_time = df.loc[indx, 'mt_1lag'] / 1000  # 0.3 # trial_duration
    trial2_time = df.loc[indx, 'mt'] / 1000
    trial1_steps = TTS(trial1_time)
    trial2_steps = TTS(trial2_time)
    iti_steps = TTS(mean_iti)
    total_steps = trial1_steps + trial2_steps + 2 * iti_steps
    # print(trial1_steps, trial2_steps, iti_steps, total_steps)
    # Default to frame operation
    lt_edges = np.arange(1, 19) * 0.1 / 6

    # down_time = np.random.random() * (down_max - down_min) + down_min
    down_time = None
    vclamp_mem = np.full((total_steps, 2), get_defaults()['default_theta'])
    prop_mem = np.zeros((total_steps, 2))
    motor_out = np.zeros((total_steps, 2))

    # Later change to movement time
    trial2_start = trial1_steps + iti_steps
    get_noise = lambda t: np.random.multivariate_normal(prop_noise, np.eye(2), t)
    lag1 = df.loc[indx, ['Y_theta_1lag', 'Y_rho_1lag']].values
    lag0 = df.loc[indx, ['Y_theta', 'Y_rho']].values
    # print(lag0, polar_to_xy(lag0))

    prop_mem = add_ramp_to_final(0, trial1_steps, polar_to_xy(lag1),
                                 prop_mem, down_time, True)
    motor_out = add_ramp_to_final(trial2_start, trial2_start + trial2_steps,
                                  polar_to_xy(lag0), motor_out, down_time, False)
    # start time, if time is less than or equal to the next bin edge, get its value
    # print(dt, trial1_steps, trial1_time, trial2_steps, trial2_time)
    ilt = 0
    for ts in range(trial1_steps):
        curr_time = min((ts + 1) * dt, 0.3)
        if curr_time <= lt_edges[ilt]:
            vclamp_mem[ts] = polar_to_xy(df.loc[indx, [f'v_theta_{ilt + 1}_frame',
                                                       f'v_rho_{ilt + 1}_frame']].values)
        else:
            # print(lt_edges[ilt], ilt, ts, (ts+1) * dt, curr_time)
            ilt += 1
            vclamp_mem[ts] = polar_to_xy(df.loc[indx, [f'v_theta_{ilt + 1}_frame',
                                                       f'v_rho_{ilt + 1}_frame']].values)

    targ_idx = np.arange(trial2_start, trial2_start + trial2_steps)
    targets = motor_out
    inputs = np.hstack((vclamp_mem, prop_mem))
    hints = targets / 2

    if showplots:
        plt.figure(figsize=(15, 15))
        plt.plot(inputs)
        plt.plot(targets)
        plt.plot(hints)
        plt.legend(['vclamp_x', 'vclamp_y',
                    'prop_x', 'prop_y',
                    'motor_x', 'motor_y', 'hints_0', 'hints_1'])
        plt.title(df.loc[indx, 'condition'])
        plt.show()

    inps_and_targs = {'inps': inputs, 'targs': targets, 'hints': hints,
                      'targ_idx': targ_idx, 'theta': [],
                      'condition': df.loc[indx, 'condition'],
                      'steps': [trial1_steps, trial2_steps]}
    return inps_and_targs


def VRIL_ID(dt=0.005, df=None, showplots=False, algorithm=[]):
    # Generates time series for the visual rotation implicit learning (VRIL) tasks,
    # will randomly generate iti
    # Here functions are autograd.numpy but should be the same
    # TUNABLES
    df = df.dropna().reset_index()
    unique_ID = np.unique(df['subj'])
    # Hard Wired
    indx = np.random.randint(len(unique_ID))
    subf = df[df['subj'] == unique_ID[indx]]
    fs = np.arange(1, 19)
    v_select = [f'v_theta_{ii}_frame' for ii in fs] + [f'v_rho_{ii}_frame' for ii in fs]
    p_inlags = ['Y_theta_1lag', 'Y_rho_1lag', 'mt_1lag']
    outlags = ['Y_theta', 'Y_rho']

    inps = subf[v_select + p_inlags].values
    targets = subf[outlags].values
    hints = targets / 2
    targ_idx = np.arange(len(targets))
    if showplots:
        plt.figure(figsize=(15, 15))
        plt.plot(inps[:, -3:])
        plt.plot(targets)
        plt.plot(hints)
        plt.legend(['prop_theta', 'prop_rho', 'Y_theta', 'Y_rho', 'hint0', 'hint1'])
        plt.title(df.loc[indx, 'condition'])
        plt.show()

    inps_and_targs = {'inps': inps, 'targs': targets, 'hints': hints,
                      'targ_idx': targ_idx}
    return inps_and_targs


def RNNtest(rnn, InpsAndTargsFunc, testdelay=0, **kwargs):
    '''
    Function that tests a trained network. Relevant parameters in p start with 'test'
    Inputs:
        InpsAndTargsFunc: function used to generate time series (same as in train)
        testdelay: Amount of time to wait between plots (useful for debugging)
        **kwargs: arguments passed to InpsAndTargsFunc
    '''
    p = rnn.p
    rnn.initialize_act()
    print('Initializing', end="")
    for i in range(p['test_init_trials']):
        print('.', end="")
        inps_and_targs = InpsAndTargsFunc(dt=p['dt'], **kwargs)
        rnn.run(inps_and_targs['inps'])
    print('')

    E_out = 0  # Running squared error
    V_targ = 0  # Running variance of target
    print('Testing: %g trials' % p['test_trials'])
    all_targs_thetas = []
    all_targs_rhos = []
    all_activities = []
    all_outputs_thetas = []
    all_outputs_rhos = []
    all_steps = []
    inp_num = []
    inp_cond = []
    all_inps = []
    all_stepN = []
    for idx in range(p['test_trials']):
        print('.', end="")
        inps_and_targs = InpsAndTargsFunc(dt=p['dt'], **kwargs)
        inp = inps_and_targs['inps']
        targ = inps_and_targs['targs']
        all_stepN.append(inps_and_targs['steps'])
        targ_idx = inps_and_targs['targ_idx']
        all_steps.append(np.arange(len(targ)) * p['dt'])
        all_inps.append(inp)
        all_targs_thetas.append(targ[:, 0].ravel())
        all_targs_rhos.append(targ[:, 1].ravel())
        out = rnn.run(inp, record_flag=1)
        all_outputs_thetas.append(out[0][:, 0].ravel())
        all_outputs_rhos.append(out[0][:, 1].ravel())
        # print(out[1].shape)
        inp_num.append(np.full(len(targ), idx))
        inp_cond.append(np.full(len(targ), inps_and_targs['condition']))
        all_activities.append(out[1])
        # print(out[1])

        out = out[0]
        # all_activities.append(out[1][np.newaxis, :, :])
        E_out = E_out + np.trace(
            np.dot(np.transpose(out[targ_idx] - targ[targ_idx]), out[targ_idx] - targ[targ_idx])) / \
                targ[targ_idx].shape[1]
        V_targ = V_targ + np.trace(np.dot(np.transpose(targ[targ_idx]), targ[targ_idx])) / \
                 targ[targ_idx].shape[1]
        time.sleep(testdelay)
    print('')
    E_norm = E_out / V_targ
    print('Normalized error: %g' % E_norm)

    pdDF = {'targ_x': np.concatenate(all_targs_thetas),
            'targ_y': np.concatenate(all_targs_rhos),
            'output_x': np.concatenate(all_outputs_thetas),
            'output_y': np.concatenate(all_outputs_rhos),
            'steps': np.concatenate(all_steps),
            'num': np.concatenate(inp_num),
            'cond': np.concatenate(inp_cond)}
    pdDF = pd.DataFrame(pdDF)
    final_outs = np.vstack([[tt[-1] for tt in all_outputs_thetas],
                            [rr[-1] for rr in all_outputs_rhos]])
    final_targs = np.vstack([[tt[-1] for tt in all_targs_thetas],
                             [rr[-1] for rr in all_targs_rhos]])
    savedict = {'df': pdDF,
                'final_outs': final_outs,
                'final_targs': final_targs,
                'activity': all_activities,
                'inps': all_inps,
                'step': np.array(all_stepN)}
    return E_norm, savedict


def plot_multiple_test(rnn, newdf, condition, nrow=5, saveopt=None):
    p = rnn.p
    p['test_trials'] = nrow
    subdf = newdf[newdf['condition'] == condition].reset_index()
    Enorm, savedict= RNNtest(rnn, VRILTR, testdelay=0, df=subdf)
    pdf = savedict['df']
    fig, axes = plt.subplots(nrows=nrow, ncols=2, figsize=(20, 20))
    unique_num = np.unique(pdf['num'])
    randinx = np.random.randint(len(unique_num), size=nrow)
    for i, ir in enumerate(unique_num[randinx]):
        #fig, axes = plt.subplots(nrows=)
        axes[i, 0].plot(pdf.loc[pdf['num'] == ir, 'steps'].values,
                pdf.loc[pdf['num']==ir, 'targ_x'].values)
        axes[i, 0].plot(pdf.loc[pdf['num'] == ir, 'steps'].values,
                pdf.loc[pdf['num']==ir, 'output_x'].values)
        axes[i, 0].legend(['target_x', 'output_x'])

        axes[i, 1].plot(pdf.loc[pdf['num'] == ir, 'steps'].values,
                pdf.loc[pdf['num']==ir, 'targ_y'].values)
        axes[i, 1].plot(pdf.loc[pdf['num'] == ir, 'steps'].values,
                pdf.loc[pdf['num']==ir, 'output_y'].values)
        axes[i, 1].legend(['target_y', 'output_y'])
    fig.suptitle(f'EMPJ Testing in {condition} condition')
    if saveopt is not None:
        if not os.path.exists(saveopt):
            os.makedirs(saveopt)
        fig.savefig(os.path.join(saveopt, f'empj_test_{condition}'))

def plot_test_average_results(rnn, newdf, saveopt=None, niter=5):
    p = rnn.p
    conditions = np.unique(newdf['condition'])
    p['test_trials'] = niter
    alldfs = []
    for cond in conditions:
        subdf = newdf[newdf['condition'] == cond].reset_index()
        Enorm, savedict= RNNtest(rnn, VRILTR, testdelay=0, df=subdf)
        resid = savedict['final_outs']-savedict['final_targs']
        erros = np.concatenate([np.full(resid.shape[1], vv) for vv in ['x_err', 'y_err']])
        #print(erros.shape, resid.ravel().shape)
        currdf = pd.DataFrame(np.vstack((resid.ravel(order='C'), erros)).T,
                              columns=['RNN-target', 'type'])
        currdf['condition'] = cond
        alldfs.append(currdf)
    alldfs = pd.concat(alldfs, axis=0, ignore_index=True)
    alldfs['RNN-target'] = alldfs['RNN-target'].astype(np.float)
    g = sns.catplot(x='RNN-target', row='type', y='condition', kind='bar', data=alldfs, ci=95)
    if saveopt is not None:
        plt.savefig(os.path.join(saveopt, 'average_error'))



def view_ring(savedict):
    ACT = savedict['activity']
    X = ACT
    activities = ACT
    allsteps = savedict['step']

    X = np.vstack(X)
    pca = PCA(n_components=3)
    pca.fit(X)

    fig = plt.figure(figsize=[15, 15])
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(activities)):
        act = activities[i]
        proj = pca.transform(act)
        traj = proj[:, 0:3]
        dstart = int(allsteps[i, 0])

        # norm_mem = matplotlib.colors.Normalize(vmin=0, vmax=dstart)
        mem_indx = np.arange(dstart)
        motor_indx = np.arange(dstart, len(act))
        ts_mem = mem_indx / len(act)

        ts_motor = motor_indx / len(act)
        memc = np.vstack((1 - ts_mem, np.full_like(ts_mem, 0.1), ts_mem)).T
        memc = [tuple(np.around(memc[i], 3)) for i in range(len(memc))]
        # memc =
        memc = [0.9, 0.2, 0.2]
        # ax.plot3D(traj[:,0],traj[:,1],traj[:,2],'o',color=c,alpha=0.1)
        ax.plot3D(traj[mem_indx, 0], traj[mem_indx, 1], traj[mem_indx, 2], 'o',
                  color=memc, alpha=.3)
        motorc = np.vstack((ts_motor, np.full_like(ts_motor, 0.1), 1 - ts_motor)).T
        motorc = [tuple(np.around(motorc[i], 3)) for i in range(len(motorc))]
        motorc = [0.3, 0.3, 0.9]
        ax.plot3D(traj[motor_indx, 0], traj[motor_indx, 1], traj[motor_indx, 2], 'o',
                  color=motorc, alpha=.3)
        if i == 0:
            last_start = ax.scatter3D(traj[0, 0], traj[0, 1], traj[0, 2], color=[0, 0, 1], label='mem_start')
            start = ax.scatter3D(traj[dstart, 0], traj[dstart, 1], traj[dstart, 2], color=[1, 0, 0],
                                 label='motor_start')
            finish = ax.scatter3D(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=[0, 0, 0], label='finish')
        else:
            last_start = ax.scatter3D(traj[0, 0], traj[0, 1], traj[0, 2], color=[0, 0, 1])
            start = ax.scatter3D(traj[dstart, 0], traj[dstart, 1], traj[dstart, 2], color=[1, 0, 0])
            finish = ax.scatter3D(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=[0, 0, 0])
        # Plotting the first start position:
        # if i==0:
        #    ax.scatter3D(traj[0,0],traj[0,1],traj[0,2],'bx', s=100)

    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_zlabel('PC3', fontsize=15)
    ax.view_init(-70, 90)
    ax.legend()

    # ax.set_zlim3d([-10, 10])

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    return pca, ax


def plot_pca_by_condition(rnn, newdf, condition, niter=30, saveopt=None):
    rnn.p['test_trials'] = niter
    subdf = newdf[newdf['condition'] == condition].reset_index()
    Enorm, savedict = RNNtest(rnn, VRILTR, testdelay=0, df=subdf)
    view_ring(savedict)
    if saveopt is not None:
        plt.title(f'RNN activity PCA {condition} Condition')
        plt.savefig(os.path.join(saveopt, f'PCA_RNN_activity_{condition}.png'))


