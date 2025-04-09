''' Usefull functions to save data'''

import os
import hashlib
import random
import string



def generate_random_hash():
    """Generate a random hash name."""
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    return hashlib.sha1(random_string.encode()).hexdigest()[:10]

def write_dict_to_file(f, dictionary, level=2):
    """Helper function to write a dictionary to the file with indentation."""
    indent = ' ' * level
    for key, value in dictionary.items():
        if isinstance(value, dict):  # Nested dictionary
            f.write(f"{indent}{key}:\n")
            write_dict_to_file(f, value, level + 2)
        else:
            f.write(f"{indent}{key}: {value}\n")


def create_folder_and_files(objmcmc,output_dir):
    """Create a folder with a random hash name and three files inside it."""
    folder_name = generate_random_hash()
    dest_unique_name = os.path.join(output_dir, folder_name)
    os.makedirs(dest_unique_name, exist_ok=True)
    
    file_name = f"{folder_name}_characteristique.txt"
    file_path = os.path.join(dest_unique_name, file_name)
    
    with open(file_path, 'w') as f:
        f.write("Object Model:\n")
        objmodel=objmcmc.prior
        f.write(f"  Name: {objmodel.name}\n")
        f.write(f"  Type: {objmodel.type}\n")
        f.write("  Parameters:\n")
        write_dict_to_file(f, objmodel.param, level=4)
        f.write(f"  Type of Graph: {objmodel.typegraph}\n")
        f.write("Settings:\n")
        for key, value in objmcmc.settings.items():
            if isinstance(value, dict):  # Nested dictionary
                f.write(f"  {key}:\n")
                write_dict_to_file(f, value, level=4)
            else:
                f.write(f"  {key}: {value}\n")
    f.close() 
    return folder_name