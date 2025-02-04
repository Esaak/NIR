import numpy as np



def get_coords_from_np(data, num_coords):
    coords = []
    for i in range(num_coords):
        coord = np.unique(data[:, i])
        coords.append(coord)

    return coords

def get_matrix(data, coords_size):
    return np.reshape(data, coords_size) # z, y, x

def get_slice_from_3d(data, distance, coord_name):
    match coord_name:
        case "x":
            return data[:,:, distance]
        case "y":
            return data[:, distance, :]
        case "z":
            return data[distance, :, :]
        
def get_slice_from_2d(data, distance, coord_name):
    match coord_name:
        case 1:
            return data[:, distance]
        case 2:
            return data[distance, :]