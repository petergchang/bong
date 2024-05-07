from lightning_sdk import Studio, Machine

# reference to the current studio
studio = Studio()

# use the jobs plugin
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']


learning_rates = [1,2,3]
agents = ['fg-bong', 'fg-blr']
grid_search_params = [(lr, agent) for lr in learning_rates for agent in agents]

for index, (lr, agent) in enumerate(grid_search_params):
    main_name = '/teamspace/studios/this_studio/bong/bong/experiments/main.py'
    cmd = f'python {main_name} --agent {agent} --learning_rate {lr}'
    job_name = f'bong-{index}'
    print('starting job', job_name, 'running', cmd)
    #job_plugin.run(cmd, machine=Machine.A10G, name=job_name)
    job_plugin.run(cmd, name=job_name)
    # results stored in /teamspace/jobs/job_name/work/xxx
