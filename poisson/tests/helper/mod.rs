#![allow(unused)]
use poisson::{algorithm, Builder, Float, Type, Vector};

use rand::distr::StandardUniform;
use rand_distr::Distribution;
use rand::{rngs::SmallRng, SeedableRng};

use num_traits::NumCast;

use alga::general::AbstractField;
use alga::linear::{FiniteDimVectorSpace, NormedSpace};

use std::fmt::Debug;

pub fn print_v<F: Float, V: Vector<F>>(v: V) -> String {
    let mut result = "(".to_owned();
    for i in 0..V::dimension() {
        result.push_str(&format!("{}, ", v[i].to_f64().unwrap()));
    }
    if V::dimension() != 0 {
        result.pop();
    }
    result.push(')');
    result
}

#[derive(Clone, Copy)]
pub enum When {
    Always,
    Sometimes,
    Never,
}

pub fn test_with_samples<T>(samples: usize, relative_radius: f64, seeds: u32, ptype: Type)
where
    T: Debug + Vector<f64> + Copy,
    StandardUniform: Distribution<T>,
{
    test_with_samples_prefilled(
        samples,
        relative_radius,
        seeds,
        ptype,
        |_| |_| None::<T>,
        When::Always,
    );
}

pub fn test_with_samples_prefilled<'r, T, F, I>(
    samples: usize,
    relative_radius: f64,
    seeds: u32,
    ptype: Type,
    mut prefiller: F,
    valid: When,
) where
    T: 'r + Debug + Vector<f64> + Copy,
    F: FnMut(f64) -> I,
    I: FnMut(Option<T>) -> Option<T>,
    StandardUniform: Distribution<f64>,
    StandardUniform: Distribution<T>,
{
    test_algo(
        samples,
        relative_radius,
        seeds,
        ptype,
        &mut prefiller,
        valid,
        algorithm::Ebeida,
    );
    test_algo(
        samples,
        relative_radius,
        seeds,
        ptype,
        &mut prefiller,
        valid,
        algorithm::Bridson,
    );
}

fn test_algo<'r, T, F, I, A>(
    samples: usize,
    relative_radius: f64,
    seeds: u32,
    ptype: Type,
    prefiller: &mut F,
    valid: When,
    algo: A,
) where
    T: 'r + Debug + Vector<f64> + Copy,
    F: FnMut(f64) -> I,
    I: FnMut(Option<T>) -> Option<T>,
    A: algorithm::Creator<f64, T>,
    StandardUniform: Distribution<f64>,
    StandardUniform: Distribution<T>,
{
    use self::When::*;
    for i in 0..seeds {
        let mut prefilled = vec![];
        let mut seed = [0u8; 32];
        for j in 0..16 {
            let primes = [3, 7, 13, 19, 29, 37, 43, 53, 61, 71, 79, 89, 101, 107, 113, 131];
            let offsets = [2741, 2729, 2713, 2707, 2693, 2687, 2677, 2663, 2657, 2633, 2609, 2591, 2557, 2549, 2539, 2521];
            seed[j] = (i * primes[j] + offsets[j]) as u8;
            seed[j + 16] = ((i * primes[j] + offsets[j]) >> 8) as u8;
        }
        let rand = SmallRng::from_seed(seed);
        let mut poisson_iter = Builder::with_samples(samples, relative_radius, ptype)
            .build(rand, algo)
            .into_iter();
        let mut poisson = vec![];
        let mut prefill = (prefiller)(poisson_iter.radius());
        let mut last = None;
        let mut does_prefill = false;
        loop {
            while let Some(p) = (prefill)(last) {
                does_prefill = true;
                match valid {
                    Always => assert!(
                        poisson_iter.stays_legal(p),
                        "All prefilled should be accepted by the '{:?}' algorithm. \
                         {} was rejected.",
                        algo,
                        print_v(p)
                    ),
                    Never => assert!(
                        !poisson_iter.stays_legal(p),
                        "All prefilled should be rejected by the '{:?}' algorithm. \
                         {} was allowed even though {:?} was last to be generated.",
                        algo,
                        print_v(p),
                        last.map(print_v)
                    ),
                    _ => {}
                }
                prefilled.push(p);
                poisson_iter.restrict(p);
            }
            if let Some(pp) = poisson_iter.next() {
                last = Some(pp);
                poisson.push(pp);
            } else {
                break;
            }
        }
        let radius = poisson_iter.radius();
        let poisson_type = poisson_iter.poisson_type();
        let poisson = poisson.into_iter().chain(
            if let Always = valid {
                prefilled
            } else {
                vec![]
            }
            .into_iter(),
        );
        test_poisson(poisson, radius, poisson_type, algo, does_prefill);
    }
}

pub fn test_poisson<F, I, T, A>(poisson: I, radius: F, poisson_type: Type, algo: A, does_prefill: bool)
where
    I: Iterator<Item = T>,
    F: Float,
    T: Debug + Vector<F> + Copy,
    A: algorithm::Creator<F, T>,
{
    use poisson::Type::*;
    let dim = T::dimension();
    let mut vecs = vec![];
    let mut hints = vec![];
    {
        let mut iter = poisson.into_iter();
        while let Some(v) = iter.next() {
            if let (low, Some(high)) = iter.size_hint() {
                hints.push((low, high));
            } else {
                panic!(
                    "There wasn't hint for {}th iteration for the '{:?}' algorithm.",
                    hints.len(),
                    algo
                );
            }
            vecs.push(v);
        }
    }
    let len = hints.len();
    for (n, (l, h)) in hints.into_iter().enumerate() {
        let remaining = len - (n + 1);
        assert!(l <= remaining, "For the '{:?}' algorithm the lower bound of hint should be smaller than or equal to actual: {} <= {}", algo, l, remaining);
        assert!(h >= remaining, "For the '{:?}' algorithm the upper bound of hint should be larger than or equal to actual: {} >= {}", algo, h, remaining);
    }

    if !does_prefill {
        for v in &vecs {
            for n in 0..T::dimension() {
                assert!(v[n] >= F::cast(0));
                assert!(v[n] < F::cast(1));
            }
        }
    }

    let vecs = match poisson_type {
        Perioditic => {
            let mut vecs2 = vec![];
            for n in 0..3i64.pow(dim as u32) {
                let mut t = T::zero();
                let mut div = n;
                for i in 0..T::dimension() {
                    let rem = div % 3;
                    div /= 3;
                    t[i] = NumCast::from(rem - 1).unwrap();
                }
                for v in &vecs {
                    vecs2.push(*v + t);
                }
            }
            vecs2
        }
        Normal => vecs,
    };

    //TODO: Figure out how to check if distribution is maximal.
    assert_legal_poisson(&vecs, radius, algo);
}

pub fn assert_legal_poisson<F, T, A>(vecs: &Vec<T>, radius: F, algo: A)
where
    F: Float,
    T: Debug + Vector<F> + Copy,
    A: algorithm::Creator<F, T>,
{
    for &v1 in vecs {
        for &v2 in vecs {
            if v1 == v2 {
                continue;
            }
            let dist = (v1 - v2).norm();
            assert!(dist > radius * F::cast(2),
                    "Poisson-disk distribution requirement not met while generating using the '{:?}' algorithm: There exists 2 vectors with \
                     distance to each other of {} which is smaller than smallest allowed one {}. \
                     The samples: [{:?}, {:?}]",
                    algo,
                    dist.to_f64().unwrap(),
                    radius.to_f64().unwrap() * 2.,
                    v1,
                    v2);
        }
    }
}
