use std::time::Instant;
use std::iter;
use algorithms::{ num_identical_pairs_combinatorics_refactored, num_identical_pairs_array };

// NOTE: profiling is very inaccurate and no conclusions about performance should be made
// from execution times
fn main() {
    for vec in [
        vec![1, 2, 3, 1, 2, 3],
        vec![1; 100_000],
        iter::repeat(0..100).take(10_000).flatten().collect::<Vec<i32>>(),
        iter::repeat(0..10_000).take(100).flatten().collect::<Vec<i32>>(),
    ] {
        let now = Instant::now();
        {
            println!("calling num_identical_pairs_combinatorics_refactored: {}", num_identical_pairs_combinatorics_refactored(&vec));
        }
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);

        let now = Instant::now();
        {
            println!("calling num_identical_pairs_array: {}", num_identical_pairs_array(&vec));
        }
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);

        println!("-----------------------------------------------------------------------------------------");
    }
}