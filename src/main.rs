mod bloom_filter;

use bloom_filter::BloomFilter;
use std::{
    collections::HashSet,
    env::args,
    fs::File,
    hash::{BuildHasher, Hasher},
    io::{self, stdout, BufRead, BufReader, Write},
    process::exit,
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

fn main() -> io::Result<()> {
    let mut args = args();
    if args.len() != 4 {
        eprintln!("usage: haver <bloom-filter-file> <err-prob> <insertions>");
        exit(1);
    }
    let (f1, err, toks): (String, f64, usize) = (
        args.nth(1).unwrap(),
        args.next().unwrap().parse().unwrap(),
        args.next().unwrap().parse().unwrap(),
    );

    let mut map: HashSet<u64, NoHasher> = HashSet::with_hasher(NoHasher::default());
    let mut bloom_filter = BloomFilter::new(err, toks);

    let file = File::open(f1)?;
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
            writeln!(writer, "error hash: {hash:016x}")?;
            err_count += 1;
        }
    }

    writeln!(writer, "Processed hashes: {total_count}")?;
    writeln!(writer, "False positives: {err_count}")?;
    writeln!(writer, "Expected error probabilty: {:.5}%", err * 100.,)?;
    writeln!(
        writer,
        "Real error rate: {:.5}%",
        (err_count as f64 / total_count as f64) * 100.,
    )?;
    Ok(())
}
