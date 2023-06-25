FROM takumaynd/focal-mujoco210:latest

# explicitly specify arch list for A4000
# See https://github.com/pytorch/extension-cpp/issues/71#issuecomment-1137310884
# This seems like a better solution, tho: https://github.com/pytorch/extension-cpp/issues/71#issuecomment-1061880626
# RUN TORCH_CUDA_ARCH_LIST="8.6+PTX" FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

# This plain form works after changing the docker default runtime to nvidia
# RUN pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

# params_proto and its dependencies
RUN pip install params_proto termcolor pandas


RUN pip install scipy h5py matplotlib mplot3d-dragger pyglet
RUN pip install opencv-python

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     swig
# RUN pip install gym[box2d]==0.22
# RUN pip install pygame
# ENV SDL_VIDEODRIVER="dummy"

RUN pip install filelock

RUN pip install git+https://github.com/deepmind/dm_control.git@mujoco_2.1.0_eol#egg=dm_control

RUN pip install colorlog

# It's gonna build in the first call
RUN python -c 'import mujoco_py'
RUN pip install -I gym==0.21.0

RUN pip install git+https://github.com/geyang/gym-dmc.git@ed2946bd91a6618cbc307308815ec244d1fc2eeb#egg=gym-dmc
# dm_control fails with a newer numpy
RUN pip install -I numpy==1.23.1

# NOTE: Without this, singularity fails with
# OSError: [Errno 30] Read-only file system: b'/usr/local/lib/python3.9/dist-packages/mujoco_py/generated/mujocopy-buildlock'
RUN chmod a+rwx -R /usr/local/lib/python3.9/dist-packages/mujoco_py/generated

# deepmind control suite reads from this env var
ENV MJLIB_PATH="/usr/local/etc/mujoco/mujoco210/bin/libmujoco210.so"

RUN pip install scikit-image pfrl
RUN pip install tabulate
