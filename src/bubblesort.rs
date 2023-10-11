use super::Sorter;

pub struct BubbleSort;

impl Sorter for BubbleSort {
    fn sort<T>(&self, slice: &mut [T])
    where
        T: Ord,
    {
        let mut swapped = true;
        while swapped {
            swapped = false;
            for i in 0..slice.len() - 1 {
                if slice[i] > slice[i + 1] {
                    // internal use of memswap
                    slice.swap(i, i + 1);
                    swapped = true;
                }
            }
        }
    }
}

#[test]
fn bubblesort_basic() {
    let mut input = vec![4, 2, 3, 5, 1];
    BubbleSort.sort(&mut input);
    assert_eq!(input, &[1, 2, 3, 4, 5]);
}
