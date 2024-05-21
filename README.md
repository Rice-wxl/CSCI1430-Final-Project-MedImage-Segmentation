# CSCI1430-Final-Project-MedImage-Segmentation

Link to the writeup for this project: [CS1430 Final Project Writeup](CS1430FinalProjectWriteup.pdf)

Running OSCAR Script: torch.cuda.is_available() = False

Container: search example container on oscar ccv

interact -q gpu

mkdir $PWD/pyenv

export PYTHONUSERBASE=$PWD/pyenv
export APPTAINER_CACHEDIR=/tmp
export APPTAINER_TMPDIR=/tmp
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PATH=$PATH:$PWD/pyenv/bin

apptainer run --nv -W $PWD /oscar/runtime/software/external/ngc-containers/pytorch.d/x86_64.d/pytorch-24.03-py3

pip install <pkg name> --user
