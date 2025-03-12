# Ha(sh)Ver(ifier)

This is a small and simple tool used to check which hashes in the inserted into a bloom filter are marked as false positives.

## Usage

```
$ haver <onion-output-file> <error-probability> <insertions>
```

- `<onion-output-file>` is a file created by `onion-rs` when run with `--show-hashes`.
- `<error-probability>, <insertions>` a float and an integer, should match the values used by `onion-rs` when creating the file from the previous argument.
