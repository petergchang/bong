import os


learning_rates = [1,2,3]
agents = ['fg-bong', 'fg-blr']
grid_search_params = [(lr, agent) for lr in learning_rates for agent in agents]

for index, (lr, agent) in enumerate(grid_search_params):
    main_name = '/teamspace/studios/this_studio/bong/bong/experiments/main.py'
    job_name = f'bong-{index}'
    output_dir = f'/teamspace/studios/this_studio/jobs/{job_name}/work'
    cmd = f'python {main_name} --agent {agent} --learning_rate {lr} --dir {output_dir}'
    print('running ', cmd)
    os.system(cmd)
