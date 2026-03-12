import os
import pkg_resources  # for accessing package data files


def load_database_files():
    """
    Loads all relevant ligand-receptor and signaling databases from the installed package's database directory.
    Returns a dictionary of file paths by species and database name.
    """
    
    base_path = pkg_resources.resource_filename('CellNeighborEX', 'database')

    database_files = {
        'human': {
            'omnipath': os.path.join(base_path, 'omnipath_human.csv'),
            'cellchat': os.path.join(base_path, 'cellchat_human.csv'),
            'celltalk': os.path.join(base_path, 'celltalk_human.csv')
        },
        'mouse': {
            'omnipath': os.path.join(base_path, 'omnipath_mouse.csv'),
            'cellchat': os.path.join(base_path, 'cellchat_mouse.csv'),
            'celltalk': os.path.join(base_path, 'celltalk_mouse.csv')
        },
        'rat': {
            'omnipath': os.path.join(base_path, 'omnipath_rat.csv')
        }
    }

    return database_files