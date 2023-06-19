import pywapor
# User inputs.
timelim = ["2021-07-01", "2021-07-11"]
latlim = [28.9, 29.7]
lonlim = [30.2, 31.2]
project_folder = r"/my_first_ETLook_run/"

# Download and prepare input data.
ds_in  = pywapor.pre_et_look.main(project_folder, latlim, lonlim, timelim)

# Run the model.
ds_out = pywapor.et_look.main(ds_in)
