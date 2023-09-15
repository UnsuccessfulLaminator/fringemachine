# Fringemachine

This tool controls a very specific lab setup, wherein an arduino connected to an I2C DAC shifts
out voltage values in a cycle in response to a trigger signal from an Allied Vision USB3 camera.
The voltage controls a piezo crystal attached to a mirror, varying the path length of one side
of a double-slit pair. The interference pattern produced can be phase-shifted in this way.

To control the camera, my rust binding of the Vimba C library is used,
[https://github.com/UnsuccessfulLaminator/vimba-rs](vimba-rs).

## Calibrate

The first subcommand is `fringemachine calibrate`, which tries to find a cycle of 3 voltages
that produce fringes separated equally in phase, i.e. with the first image as a reference, the
next two should have phase shifts of 2pi/3 and 4pi/3. This is accomplished by holding the 1st
voltage constant and sweeping through ranges for the 2nd and 3rd.

```
Usage: fringemachine calibrate [OPTIONS] <V0> <V1_RANGE> <V2_RANGE>

Arguments:
  <V0>        Value of cycle voltage 0
  <V1_RANGE>  Range of values to try for cycle voltage 1
  <V2_RANGE>  Range of values to try for cycle voltage 2

Options:
  -i, --iterations <ITERATIONS>  Number of calibration iterations [default: 3]
  -h, --help                     Print help
```

With the piezo driver set as it is at the time of writing, voltage ranges should be very small,
for example:

```
fringemachine calibrate 0 1..1.01 2.5..2.51 -i 2
```

will keep the 1st cycle voltage at 0 V, while sweeping the 2nd from 1 V to 1.01 V and the 3rd from
2.5 V to 2.51 V. Ten samples are used for each range, so there are 100 cycles tested in total. 2
iterations will be used in the above command.

## Acquire

Having calibrated, we can acquire video. The video can either be displayed or saved, and its
contents can either be raw frames or wrapped phase images processed from every 3 raw frames. In the
latter case, the video will have 1/3rd of the raw framerate.

```
Usage: fringemachine acquire [OPTIONS] <V0> <V1> <V2>

Arguments:
  <V0>  Value of cycle voltage 0
  <V1>  Value of cycle voltage 1
  <V2>  Value of cycle voltage 2

Options:
  -p, --phase          If set, frames will be processed into wrapped phase and the video will have one third of the camera's framerate. If not set, video will be raw
  -o, --output <FILE>  Video output destination. If not specified, video will be played live
  -h, --help           Print help
```

For example, if we found a good cycle of 0 V, 1.006 V, 2.503 V from the previous calibration
command, we can acquire wrapped phase video to a file called "out.mp4" with this command:

```
fringemachine acquire 0 1.006 2.503 -o out.mp4 -p
```
