use clap::{Arg, ArgMatches, Command, builder::PossibleValuesParser};

use poisson::{Builder, Type, algorithm::{Bridson, Ebeida}};

use rand::{Rng, seq::SliceRandom, SeedableRng, rng};
use rand::rngs::SmallRng;

use nalgebra::Vector2;

use image::{ImageBuffer, Rgb};

use lab::Lab;

use fnv::FnvHasher;

use std::hash::Hasher;
use std::str::FromStr;

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Algo {
    Ebeida,
    Bridson
}

impl FromStr for Algo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ebeida" => Ok(Algo::Ebeida),
            "bridson" => Ok(Algo::Bridson),
            _ => Err(format!("Invalid algorithm: {}", s))
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Style {
    Plain,
    Colorful,
    Dot
}

impl FromStr for Style {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "plain" => Ok(Style::Plain),
            "colorful" => Ok(Style::Colorful),
            "dot" => Ok(Style::Dot),
            _ => Err(format!("Invalid style: {}", s))
        }
    }
}

fn main() {
    let app = Command::new("Poisson visualisation")
        .author("delma")
        .version("0.1.0")
        .about("Visualisation for poisson library")
        .arg(
            Arg::new("OUTPUT")
                .help("Output file that's generated")
                .required(true)
                .index(1)
        )
        .arg(
            Arg::new("SEED")
                .help("Seed for the generation")
                .index(2)
        )
        .arg(
            Arg::new("radius")
                .short('r')
                .value_name("RADIUS")
                .help("Radius of the disks")
        )
        .arg(
            Arg::new("width")
                .short('w')
                .value_name("WIDTH")
                .help("Width of the generated image")
        )
        .arg(
            Arg::new("height")
                .short('h')
                .value_name("HEIGHT")
                .help("Height of the generated image")
        )
        .arg(
            Arg::new("style")
                .short('s')
                .value_name("STYLE")
                .help("Style for the disks")
                .value_parser(PossibleValuesParser::new(["plain", "colorful", "dot"]))
        )
        .arg(
            Arg::new("algo")
                .short('a')
                .help("Algorithm that's used to generate image")
                .value_name("ALGO")
                .value_parser(PossibleValuesParser::new(["ebeida", "bridson"]))
        );
    visualise(app.get_matches());
}

fn visualise(m: ArgMatches) {
    let width: u32 = m.get_one::<String>("width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let height: u32 = m.get_one::<String>("height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let radius: f32 = m.get_one::<String>("radius")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02);
    let algo = m.get_one::<String>("algo")
        .and_then(|s| Algo::from_str(s).ok())
        .unwrap_or(Algo::Ebeida);
    let style = m.get_one::<String>("style")
        .and_then(|s| Style::from_str(s).ok())
        .unwrap_or(Style::Plain);
    let name = m.get_one::<String>("OUTPUT").unwrap();
    let master_rng = m.get_one::<String>("SEED").map(|s| {
        let mut fnv = FnvHasher::with_key(0);
        for b in s.bytes() {
            fnv.write_u8(b);
        }
        SmallRng::seed_from_u64(fnv.finish())
    }).unwrap_or_else(|| SmallRng::from_rng(&mut rng()));

    let mut style_rng = master_rng.clone();

    let builder = Builder::<_, Vector2<f32>>::with_radius(radius, Type::Normal);
    let points = if algo == Algo::Ebeida {
        builder.build(master_rng, Ebeida).generate()
    } else {
        builder.build(master_rng, Bridson).generate()
    };

    let mut ps = points.clone();
    ps.shuffle(&mut style_rng);

    let mut image = ImageBuffer::new(width, height);
    for p in points {
        let pp = ps.pop().unwrap();
        let col = Rgb(Lab {
            l: style_rng.random::<f32>() * 80. + 10.,
            a: pp.x * 256. - 128.,
            b: pp.y * 256. - 128.
        }.to_rgb());

        let x = p.x * width as f32;
        let y = p.y * height as f32;
        let (rx, ry) = if style == Style::Dot {
            (0.2 * radius * width as f32, 0.2 * radius * height as f32)
        } else {
            (radius * width as f32, radius * height as f32)
        };
        for xx in -rx as i32..rx as i32 {
            for yy in -ry as i32..ry as i32 {
                let xx = xx as f32;
                let yy = yy as f32;
                let xxx = (x + xx) as i32;
                let yyy = height as i32 - (y + yy) as i32;
                if xxx < 0 || xxx >= width as i32 {
                    // Outside of the picture horizontally
                    continue;
                }
                if yyy < 0 || yyy >= height as i32 {
                    // Outside of the picture vertically
                    continue;
                }
                if xx * xx / (rx * rx) + yy * yy / (ry * ry) > 1. {
                    // Outside of the disk
                    continue;
                }
                let xxx = xxx as u32;
                let yyy = yyy as u32;
                if style == Style::Colorful {
                    image[(xxx, yyy)] = col;
                } else {
                    image[(xxx, yyy)] = Rgb([255, 255, 255]);
                }
                if style == Style::Plain && (xx == 0. || yy == 0.) {
                    image[(xxx, yyy)] = Rgb([255, 0, 0]);
                }
            }
        }
    }
    image.save(name).unwrap();
}
