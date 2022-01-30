import click
import numpy as np
import pickle

import logger
from misc_util import set_global_seeds
import config as config
from rollout import RolloutWorker
import os
import serial
import time
import binascii
from joint_array_publisher import JointArrayPub

@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=1)
#@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts):
    query = 'roslaunch aubo_i5_moveit_config moveit_planning_execution.launch robot_ip:=192.168.1.101'
    os.system(query)
    publisher_to_moveit_object = JointArrayPub()
    ser = serial.Serial(port='/dev/ttyUSB0',baudrate=115200,timeout=1, parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)


    #ser.write("\t\x10\x03è\x00\x03\x06\x00\x00\x00\x00\x00\x00s0")
    ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")
    data_raw = ser.readline()
    print(data_raw)
    data = binascii.hexlify(data_raw)
    print ("Response 1 ", data)
    time.sleep(0.01)
 
    ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")
    data_raw = ser.readline()
    print(data_raw)
    data = binascii.hexlify(data_raw)
    print ("Response 2 ", data)
    time.sleep(1)
    
    print ("Open gripper")
    ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")
    data_raw = ser.readline()
    print(data_raw)
    data = binascii.hexlify(data_raw)
    print ("Response 4 ", data)
    time.sleep(2)

    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    #env_name = policy.info['env_name']
    env_name = 'AuboReach-v0'
    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'rollout_batch_size': 1,
        'T': params['T'],
    }
    #eval_params = {
     #   'exploit': True,
      #  'use_target_net': params['test_with_polyak'],
       # 'compute_Q': True,
        #'rollout_batch_size': 1,
        #'render': bool(render),
    #}

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    #evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()

    print ("Close gripper")
    ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29")
    data_raw = ser.readline()
    print(data_raw)
    data = binascii.hexlify(data_raw)
    print ("Response 3 ", data)
    time.sleep(2)
    #action = [-3.0617, -0.9315,-2.139, -1.2847]
    action = [0,0,0,0]
    publisher_to_moveit_object.pub_joints_to_moveit(action)


if __name__ == '__main__':
    main()