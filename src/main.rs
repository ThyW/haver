mod bloom_filter;

use bloom_filter::BloomFilter;
use clap::Parser as _;
use std::{
    collections::HashSet,
    fs::File,
    hash::{BuildHasher, Hasher},
    io::{self, stdout, BufRead, BufReader, Write},
    path::PathBuf,
};

#[derive(Default, PartialEq, Eq)]
struct NoHasher(u64);

impl Hasher for NoHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for (index, item) in bytes.iter().enumerate().take(8) {
            self.0 ^= u64::from(*item) << (index * 8);
        }
    }
}

impl BuildHasher for NoHasher {
    type Hasher = NoHasher;

    fn build_hasher(&self) -> Self::Hasher {
        NoHasher::default()
    }
}

#[derive(clap::Parser)]
#[command(version, about)]
struct Args {
    file: PathBuf,
    #[arg(short = 'e', default_value_t = 0.001)]
    err_prob: f64,
    #[arg(short = 'i', default_value_t = 10000000)]
    insertions: usize,
    #[arg(short = 'q')]
    quiet: bool,
    #[arg(short = 's')]
    short: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let mut map: HashSet<u64, NoHasher> = HashSet::with_hasher(NoHasher::default());
    let mut bloom_filter = BloomFilter::new(args.err_prob, args.insertions);

    let file = File::open(args.file)?;
    let mut writer = stdout().lock();

    let breader = BufReader::new(file);
    let mut err_count = 0;
    let mut total_count = 0;

    for line in breader.lines().map_while(|x| x.ok()) {
        total_count += 1;
        let (hash, _) = line.split_at(line.find('\t').unwrap());
        let hash = hash.trim();
        let hash = u64::from_str_radix(hash, 16).unwrap();
        let c1 = map.insert(hash);
        let c2 = bloom_filter.insert_raw((hash, (hash ^ 12658332951230890439u64).rotate_left(1)));
        if c1 && !c2 {
            if !args.quiet && !args.short {
                writeln!(writer, "error hash: {hash:016x}")?;
            }
            err_count += 1;
        }
    }

    if args.short {
        writeln!(writer, "{total_count} {err_count}")?;
        return Ok(());
    }
    writeln!(writer, "Processed hashes: {total_count}")?;
    writeln!(writer, "False positives: {err_count}")?;
    writeln!(
        writer,
        "Expected error probabilty: {:.5}%",
        args.err_prob * 100.,
    )?;
    writeln!(
        writer,
        "Real error rate: {:.5}%",
        (err_count as f64 / total_count as f64) * 100.,
    )?;
    Ok(())
}
