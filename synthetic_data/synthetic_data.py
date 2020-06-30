import numpy as np
import scipy as sp
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import math
from os import listdir
import glob
from multiprocessing import Pool
from functools import partial
import subprocess
import os
import time
import random
import pickle
import time


# ----------------- PARAMETERS ---------------------
num_data = 1000    # number of data to generate


# .exe Parameters
program = "brain" # name of exe file
N = 4 # number of threads
model = "RD" # Reaction-diffusion model
profiler = 1 # Code profiling
verbose = 1 # Simulation printouts
adaptive = 0 # Adaptive grid refinement
PatFileName = "Atlas/anatomy_dat/" # Location of patient anatomy file
vtk_dump = 1 # dumping of output
channel = ['0'] # Channels to keep during vtu to npz conversion


# Tumor Parameters
Dw_range = (0.0003, 0.0013) # Diffusivity in white matter [cm^2/day]
rho_range = (0.005, 0.13) # Proliferation rate [1/day]
Tend_range = (50, 1000) # Final simulation time [day]
icx_range = (0.4, 0.8) # tumor initial location
icy_range = (0.4, 0.8)
icz_range = (0.2, 0.6)
uth_range = (0.3, 1.0) # tumor threshold

#----------------------- Modified vtu2npz Functions-----------------------------
def read_grid_vtk(data):
    # Get the coordinates of nodes in the mesh
    nodes_vtk_array= data.GetPoints().GetData()
    vertices = vtk_to_numpy(nodes_vtk_array)
    #The "Velocity" field is the vector in vtk file
    numpy_array = []
    for i in channel:
        vtk_array = data.GetPointData().GetArray('channel'+i)
        numpy_array.append(vtk_to_numpy(vtk_array))

    return vertices, np.array(numpy_array)

# This function is modified to save the thresholded values
def extract_VTK(filename, u_th):
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()

    vertices, numpy_array = read_grid_vtk(vtk_data)
    numpy_data[x, y, z, :] = numpy_array.T
    # Apply the thresholding
    thr_numpy_data = np.zeros_like(numpy_data)
    thr_numpy_data[numpy_data > u_th] = 1

    path, filename = os.path.split(filename)
    file_name = path.replace("vtu_data", "npz_data")+"/"+filename
    file_name = file_name.replace(".vtu",".npz")

    np.savez_compressed(file_name, data=numpy_data, thr_data=thr_numpy_data)
#------------------------------------------------------------------------------


# Generate dataset
for i in range(num_data):

    # Sample parameters from the given range
    Dw = round(random.uniform(Dw_range[0], Dw_range[1]), 8)
    rho = round(random.uniform(rho_range[0], rho_range[1]), 7)
    Tend = random.randrange(Tend_range[0], Tend_range[1])
    icx = round(random.uniform(icx_range[0], icx_range[1]), 5)
    icy = round(random.uniform(icy_range[0], icy_range[1]), 5)
    icz = round(random.uniform(icz_range[0], icz_range[1]), 5)
    uth = round(random.uniform(uth_range[0], uth_range[1]), 5)

    # Create folders to store the data
    vtu_path = os.path.join("Dataset/vtu_data", str(i))
    npz_path = os.path.join("Dataset/npz_data", str(i))
    os.mkdir(vtu_path)
    os.mkdir(npz_path)

    # Run the .exe file with specified parameters(dumpfreq = Tend)
    args = "./{} -nthreads {} -model {} -profiler {} -verbose {} -adaptive {} \
    -PatFileName {} -vtk {} -dumpfreq {} -Dw {} -rho {} -Tend {} -icx {} \
    -icy {} -icz {}".format(program, N, model, profiler, verbose, adaptive, \
    PatFileName, vtk_dump, Tend, Dw, rho, Tend, icx, icy, icz).split()
    subprocess.run(args)

    # Move the simulation results to vtu folder
    args_data = ("mv Data_0001.vtu " + vtu_path).split()
    args_txt = ("mv supro_params.txt " + npz_path).split()
    subprocess.run(args_data)
    subprocess.run(args_txt)

    # Convert vtu files to npz
    files_cfd = []
    for filename in sorted(glob.glob(os.path.join(vtu_path,'*.vtu'))):
        files_cfd.append(filename)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(files_cfd[0])
    reader.Update()
    vtk_data = reader.GetOutput()
    vertices, numpy_array = read_grid_vtk(vtk_data)
    bounds_cfd = vtk_data.GetBounds()
    H = np.unique(vertices[:,0]).shape[0]
    W = np.unique(vertices[:,1]).shape[0]
    D = np.unique(vertices[:,2]).shape[0]
    factor = 128
    numpy_data = np.zeros((H, W, D, len(channel)))
    x, y, z = zip(*list(map(tuple, np.uint16(factor*vertices))))
    thr_extract_VTK = partial(extract_VTK, u_th=uth)
    pool = Pool(8)
    pool.map(thr_extract_VTK, files_cfd)
    pool.close()

    # Dump the parameter tags
    tag_dict = {"Dw":Dw, "rho":rho, "Tend":Tend, "icx":icx, "icy":icy, \
    "icz":icz, "uth":uth}
    pickle_path = npz_path + "/" + "parameter_tag.pkl"
    with open(pickle_path, 'wb') as handle:
        pickle.dump(tag_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("************************************************************************")
    print(str(i+1), "th data has been generated!")
    print(str((i+1)*100/num_data), "%")
    print("************************************************************************")
