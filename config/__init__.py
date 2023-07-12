from configtree import Loader, Walker, Updater


def walk_configs(config_path):
    walk = None
    update = Updater(namespace={'None': None})
    load = Loader(walk=walk, update=update)
    print(f'walking {config_path}')
    main_configs = load(config_path)
    main_configs = main_configs.rare_copy()

    return main_configs


