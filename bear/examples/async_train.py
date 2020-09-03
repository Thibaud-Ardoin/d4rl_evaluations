import ray
from bear_hdf5_d4rl import *
from glob import glob

print(' * Bear asynchronous training script * ')

# python async_train.py --env='flow-ring-random-v0' 

@ray.remote
def launch_train(dir, variant):
    rand = np.random.randint(0, 100000)
    setup_logger(os.path.join('BEAR_launch', str(rand)), snapshot_mode='gap', variant=variant, base_log_dir='./data/'+dir)
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)
    experiment(variant)

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='BEAR-runs')
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v0')
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--qf_lr', default=5e-5, type=float)
    parser.add_argument('--policy_lr', default=5e-5, type=float)
    parser.add_argument('--mmd_sigma', default=50, type=float)
    parser.add_argument('--kernel_type', default='gaussian', type=str)
    parser.add_argument('--target_mmd_thresh', default=0.05, type=float)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    variant = dict(
        algorithm="BEAR",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E5),
        buffer_filename=None, #halfcheetah_101000.pkl',
        load_buffer=True,
        env_name=args.env,
        dataset=args.dataset,
        algorithm_kwargs=dict(
            num_epochs=300,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            num_actions_sample=args.num_samples,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,

            # BEAR specific params
            mode='auto',
            kernel_choice=args.kernel_type,
            policy_update_style='0',
            mmd_sigma=args.mmd_sigma,
            target_mmd_thresh=args.target_mmd_thresh,

        ),
    )
    
    ray.init()
    result_ids = []
    data_dir = '/home/ubuntu/.d4rl/datasets/remote/flow-ring-v0-idm-noise'
    noise_type=['1', '2+5', '5', '7+5']
    datasets=[data_dir + noise_type[k] + '-clean.hdf5' for k in range(4)]

    for i,dataset in enumerate(datasets):
        print('Go for data: ', dataset)
        print('Also called dataset N', i)
        variant['dataset']=dataset

        for j in range(4):
            print('Start the run N',j)
            result_ids.append(launch_train.remote(dir='async_noise_'+noise_type[i], variant=variant))


    results = ray.get(result_ids)
    print('Result of the multi lunch : ', results)
