use vimba::prelude::*;
use std::io::{self, Write};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::fs::File;
use std::ops::Range;
use std::f64::consts::TAU;
use std::process::{Command, Stdio, ChildStdin};
use std::path::{Path, PathBuf};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::par_azip;
use ndrustfft::{R2cFftHandler, FftHandler, ndfft_r2c, ndfft, Complex};
use serialport::{self, SerialPort};
use anyhow::{self, Context, bail};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json;



fn parse_range(s: &str) -> Result<Range<f64>, String> {
    let parts: Vec<&str> = s.split("..").collect();

    if parts.len() != 2 {
        Err("Range must be of form START..END".to_string())
    }
    else {
        let start: f64 = parts[0].parse().map_err(|_| "Invalid float for start".to_string())?;
        let end: f64 = parts[1].parse().map_err(|_| "Invalid float for end".to_string())?;

        Ok(start..end)
    }
}

#[derive(Parser)]
struct Args {
    #[arg(short, long, value_name = "FILE")]
    /// Hardware configuration file, containing camera and serial parameters
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: ProgramCommand
}

#[derive(Subcommand)]
enum ProgramCommand {
    /// Find a cycle of voltages that produces 3 images whose fringes are equally
    /// separate in phase.
    Calibrate {
        /// Value of cycle voltage 0
        v0: f64,

        #[arg(value_parser = parse_range)]
        /// Range of values to try for cycle voltage 1
        v1_range: Range<f64>,

        #[arg(value_parser = parse_range)]
        /// Range of values to try for cycle voltage 2
        v2_range: Range<f64>,

        #[arg(short, long, default_value_t = 3)]
        /// Number of calibration iterations
        iterations: usize
    },
    
    /// Acquire video with the fringes cycled using specified voltage values
    Acquire {
        /// Value of cycle voltage 0
        v0: f64,

        /// Value of cycle voltage 1
        v1: f64,

        /// Value of cycle voltage 2
        v2: f64,

        #[arg(short, long, action)]
        /// If set, frames will be processed into wrapped phase and the video will have
        /// one third of the camera's framerate. If not set, video will be raw.
        phase: bool,

        #[arg(short, long, value_name = "FILE")]
        /// Video output destination. If not specified, video will be played live.
        output: Option<PathBuf>
    }
}

#[derive(Deserialize, Clone, Copy)]
struct Region {
    x: usize, y: usize, w: usize, h: usize
}

#[derive(Deserialize)]
struct Config {
    camera_name: Option<String>,
    framerate: f64,
    exposure_micros: f64,
    gain_db: f64,
    gamma: f64,
    roi: Option<Region>,
    trigger_line: String,

    serial_device: String,
    serial_baud: u32
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config_path = args.config.unwrap_or(PathBuf::from("config"));
    let config_file = File::open(config_path).context("Failed to read file \"config\"")?;
    let config: Config = serde_json::from_reader(config_file).context("Bad config file")?;

    // Set up hardware
    let mut cam = setup_camera(&config)?;
    let mut port = setup_mcu(&config)?;
    
    let cam_width = cam.get_feature_int("Width")? as usize;
    let cam_height = cam.get_feature_int("Height")? as usize;
    let roi = config.roi.unwrap_or(Region { x: 0, y: 0, w: cam_width, h: cam_height });

    if roi.w > cam_width { bail!("ROI width larger than camera frame width"); }
    if roi.h > cam_height { bail!("ROI height larger than camera frame height"); }
    
    match args.command {
        ProgramCommand::Calibrate { v0, v1_range, v2_range, iterations } => {
            calibrate(&mut port, &mut cam, v0, v1_range, v2_range, roi, iterations)
        },
        ProgramCommand::Acquire { v0, v1, v2, phase, output } => {
            let fps = if phase { config.framerate/3. } else { config.framerate };
            let mut ffmpeg = setup_ffmpeg(fps, roi.w, roi.h, output.as_deref())?;

            acquire(&mut port, &mut cam, &mut ffmpeg, v0, v1, v2, roi, phase)
        }
    }
}

fn acquire(
    port: &mut impl Write, cam: &mut Camera, out: &mut impl Write,
    v0: f64, v1: f64, v2: f64,
    roi: Region,
    phase: bool
) -> anyhow::Result<()> {
    write!(port, "m cycle\n")?; // Change to cycle mode
    write!(port, "l0003\n")?;   // Cycle length of 3
    write!(port, "c")?;         // Begin setting cycle values
    
    // Set cycle values
    for v in [v0, v1, v2] {
        let dac_value = (v*65536./5.) as u16;

        port.write_all(&dac_value.to_be_bytes())?;
    }
    
    let (x0, x1) = (roi.x, roi.x+roi.w);
    let (y0, y1) = (roi.y, roi.y+roi.h);

    let (tx, rx) = mpsc::channel();
    
    cam.start_streaming(move |frame| {
        let data = frame.unpack_data_to_u16().unwrap();
        let arr = Array::from_shape_vec((frame.height, frame.width), data).unwrap();
        let arr = arr.slice(s![y0..y1, x0..x1]).mapv(|x| x as f64);

        tx.send(arr).unwrap();
        StreamContinue(true)
    }, 16)?;
    
    if phase {
        for (frame0, frame1, frame2) in rx.iter().tuples() {
            let mut phase = Array2::zeros(frame0.dim());
            
            par_azip!((p in &mut phase, &i0 in &frame0, &i1 in &frame1, &i2 in &frame2) {
                let numer = 3f64.sqrt()*(i1-i2);
                let denom = 2.*i0-i1-i2;
                
                *p = numer.atan2(denom);
            });

            let out_bytes: Vec<u8> = phase.iter()
                .map(|&x| ((x/TAU)+0.5)*4095.)
                .flat_map(|x| (x as u16).to_le_bytes())
                .collect();

            out.write_all(&out_bytes)?;
        }
    }
    else {
        for frame in rx.iter() {
            let out_bytes: Vec<u8> = frame.iter()
                .flat_map(|&x| (x as u16).to_le_bytes())
                .collect();

            out.write_all(&out_bytes)?;
        }
    }

    cam.stop_streaming()?;

    Ok(())
}

fn calibrate(
    port: &mut impl Write, cam: &mut Camera,
    v0: f64, mut v1_range: Range<f64>, mut v2_range: Range<f64>,
    roi: Region,
    iterations: usize
) -> anyhow::Result<()> {
    let mut cycles = Array3::zeros((10, 10, 3));
    let mut errors = Array2::zeros((10, 10));

    for i in 0..iterations {
        let (v1, v2) = find_best_cycle(
            port, cam,
            v0, v1_range.clone(), v2_range.clone(), roi,
            cycles.view_mut(), errors.view_mut()
        )?;

        println!("Iteration {i} - Best cycle 0, {v1}, {v2}");

        show_surface(cycles.slice(s![.., .., 1]), cycles.slice(s![.., .., 2]), errors.view())?;

        let dv1 = v1_range.end-v1_range.start;
        let dv2 = v2_range.end-v2_range.start;

        v1_range = (v1-dv1/4.)..(v1+dv1/4.);
        v2_range = (v2-dv2/4.)..(v2+dv2/4.);
    }

    Ok(())
}

fn show_surface(xs: ArrayView2::<f64>, ys: ArrayView2::<f64>, zs: ArrayView2::<f64>)
-> io::Result<()> {
    let mut file = File::create("data")?;

    azip!((x in xs, y in ys, z in zs) {
        write!(file, "{x} {y} {z}\n").unwrap();
    });

    drop(file);

    Command::new("gnuplot").stderr(Stdio::null()).args(["-c", "plot"]).status()?;

    Ok(())
}

fn find_best_cycle(
    port: &mut impl Write, cam: &mut Camera,
    v0: f64, v1_range: Range<f64>, v2_range: Range<f64>,
    roi: Region,
    mut cycles: ArrayViewMut3::<f64>, mut errors: ArrayViewMut2::<f64>
) -> anyhow::Result<(f64, f64)> {
    let v1s = Array1::<f64>::linspace(v1_range.start, v1_range.end, cycles.dim().0);
    let v2s = Array1::<f64>::linspace(v2_range.start, v2_range.end, cycles.dim().1);
    
    // After this, every cycle_values[i, j, ..] is a unique cycle of [v0, v1, v2]
    cycles.slice_mut(s![.., .., 0]).fill(v0);
    cycles.slice_mut(s![.., .., 1..]).assign(&meshgrid(&v1s.view(), &v2s.view()));
    
    // Transmit cycle values to MCU, set it to cycle mode and move to index 0,
    // wait for a second in case it's still processing the serial data.
    set_cycle_values(port, &cycles.view())?;
    write!(port, "m cycle\n")?;
    write!(port, "i0000\n")?;
    thread::sleep(Duration::from_secs(1));
    
    // Channel to send frames from camera thread to main thread
    let (tx, rx) = mpsc::channel();
    
    // Region of interest (ROI) bounds
    let (x0, x1) = (roi.x, roi.x+roi.w);
    let (y0, y1) = (roi.y, roi.y+roi.h);
    
    // This stream closure converts every raw frame into a 2D array cropped to
    // the ROI and sends it to the main thread
    cam.start_streaming(move |frame| {
        let data = frame.unpack_data_to_u16().unwrap();
        let arr = Array::from_shape_vec((frame.height, frame.width), data).unwrap();
        let arr = arr.slice(s![y0..y1, x0..x1]).mapv(|x| x as f64);

        tx.send(arr).unwrap();
        StreamContinue(true)
    }, 16)?;
    
    // FFT handlers are contexts for efficient FFTs. Also need the frequencies
    let mut fft_handler_x = R2cFftHandler::<f64>::new(x1-x0);
    let mut fft_handler_y = FftHandler::<f64>::new(y1-y0);
    let freqs = rfft2_freqs((x1-x0)/2+1, y1-y0);
    let freq_mags = freqs.map_axis(Axis(2), |freq| freq[0].hypot(freq[1]));
    
    // Will be filled with phase values corresponding to individual voltages
    // in each cycle
    let mut phases = Array3::<f64>::zeros(cycles.raw_dim());

    for (frame, phase) in rx.iter().zip(phases.iter_mut()) {
        let mut fft = rfft2(&frame.view(), &mut fft_handler_x, &mut fft_handler_y);
        
        azip!((&f in &freq_mags, c in &mut fft) {
            if f < 1./50. || f > 1./5. {
                *c = Complex { re: 0., im: 0. };
            }
        });
        
        let argmax = fft.indexed_iter()
            .max_by(|(_, c0), (_, c1)| c0.norm_sqr().total_cmp(&c1.norm_sqr()))
            .unwrap()
            .0;

        *phase = fft[argmax].arg().rem_euclid(TAU);
    }

    cam.stop_streaming()?;
    
    // Each error value measures how unevenly spaced the 3 phases in a cycle are
    // compared to [0, 2pi/3, 4pi/3]
    azip!((mut p in phases.rows_mut(), err in &mut errors) {
        p.mapv_inplace(|p| p.rem_euclid(TAU));
        p -= p.iter().copied().reduce(f64::min).unwrap();

        let err1 = (p[1]-(TAU/3.)).powf(2.);
        let err2 = (p[2]-(2.*TAU/3.)).powf(2.);

        *err = err1+err2;
    });
    
    // Index (i, j) of the minimum error
    let argmin = errors.indexed_iter().min_by(|(_, e0), (_, e1)| e0.total_cmp(e1)).unwrap().0;
    let best_cycle = cycles.slice(s![argmin.0, argmin.1, ..]);

    Ok((best_cycle[1], best_cycle[2]))
}

fn rfft2(
    arr: &ArrayView2<f64>,
    handler_x: &mut R2cFftHandler<f64>, handler_y: &mut FftHandler<f64>
) -> Array2<Complex<f64>> {
    let mut temp = Array2::<Complex<f64>>::zeros((arr.nrows(), arr.ncols()/2+1));
    let mut fft = Array2::<Complex<f64>>::zeros(temp.raw_dim());

    ndfft_r2c(&arr, &mut temp, handler_x, 1);
    ndfft(&temp, &mut fft, handler_y, 0);

    fft
}

fn rfft2_freqs(output_width: usize, output_height: usize) -> Array3::<f64> {
    let x_freqs = Array::linspace(0., 0.5, output_width);
    let y_freqs = Array::range(0., 1., 1./output_height as f64)
        .mapv(|f| (f+0.5).rem_euclid(1.)-0.5);

    meshgrid(&x_freqs.view(), &y_freqs.view())
}

fn meshgrid(xs: &ArrayView1::<f64>, ys: &ArrayView1::<f64>) -> Array3::<f64> {
    let mut out = Array3::<f64>::zeros((ys.len(), xs.len(), 2));

    for mut row in out.slice_mut(s![.., .., 0]).rows_mut() { row.assign(&xs); }
    for mut col in out.slice_mut(s![.., .., 1]).columns_mut() { col.assign(&ys); }

    out
}

fn set_cycle_values(port: &mut impl Write, values: &ArrayView3::<f64>) -> io::Result<()> {
    write!(port, "l{:04x}\n", values.len())?;
    write!(port, "c")?;

    for v in values.iter() {
        let dac_value = (v*65536./5.) as u16;

        port.write_all(&dac_value.to_be_bytes())?;
    }

    Ok(())
}

fn setup_ffmpeg(
    framerate: f64, width: usize, height: usize, output: Option<&Path>
) -> io::Result<ChildStdin> {
    let size = format!("{width}x{height}");
    let fps = framerate.to_string();
    let mut ffmpeg_args = vec![];
    let mut cmd = "ffplay";

    ffmpeg_args.extend([
        "-framerate", &fps,          // Input options
        "-f", "rawvideo",            //  |
        "-pixel_format", "gray12le", //  |
        "-video_size", &size,        // <+
        "-i", "-",                   // Read from stdin
    ]);

    if let Some(path) = output {
        let path = path.to_str().expect("Invalid UTF-8 in output path");

        ffmpeg_args.extend([
            "-vcodec", "h264", // Output format
            path,              // Write to a file
            "-y"               // Overwrite if the file exists
        ]);

        cmd = "ffmpeg";
    }

    let child = Command::new(cmd)
        .args(ffmpeg_args)
        .stdin(Stdio::piped()) // Want to open a pipe so we can write data
        .stderr(Stdio::null()) // Discard all the gunk from stderr
        .spawn()?;
    
    Ok(child.stdin.expect("FFmpeg process has no stdin!"))
}

fn setup_mcu(config: &Config) -> serialport::Result<Box<dyn SerialPort>> {
    let port = serialport::new(&config.serial_device, config.serial_baud)
        .timeout(std::time::Duration::from_millis(100))
        .open()?;
    
    thread::sleep(Duration::from_secs(2)); // Wait for device to reset

    Ok(port)
}

fn setup_camera(config: &Config) -> vimba::Result<Camera> {
    let vimba = Vimba::new()?;
    let cameras = vimba.list_cameras()?;

    let cam_id = if let Some(cam_id) = &config.camera_name {
        if !cameras.iter().any(|c| c.id == *cam_id) {
            panic!("Camera {} not found!", cam_id);
        }

        &cam_id
    }
    else {
        if cameras.len() == 0 { panic!("No cameras found!"); }

        &cameras[0].id
    };
    
    let cam = vimba.open_camera(cam_id, AccessMode::FULL)?;

    cam.set_feature_enum("PixelFormat", "Mono12")?;
    cam.set_feature_bool("AcquisitionFrameRateEnable", true)?;
    cam.set_feature_float("AcquisitionFrameRate", config.framerate)?;
    cam.set_feature_float("ExposureTime", config.exposure_micros)?;
    cam.set_feature_float("Gain", config.gain_db)?;
    cam.set_feature_float("Gamma", config.gamma)?;
    cam.set_feature_enum("LineSelector", &config.trigger_line)?;
    cam.set_feature_enum("LineMode", "Output")?;
    cam.set_feature_enum("LineSource", "ExposureActive")?;
    cam.set_feature_bool("LineInverter", true)?;
    
    Ok(cam)
}
