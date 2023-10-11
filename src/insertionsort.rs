use super::Sorter;

pub struct InsertionSort<'a> {
    variation: &'a str
}

impl<'a> Sorter for InsertionSort<'a> {
    // In place
    // Stable
    // Time O(N^2)
    // Space O(1)
    // Invariant: The subarray from 0 to the element before the current one is sorted.
    fn sort<T>(&self, slice: &mut [T])
    where
        T: Ord,
    {
        match self.variation {
            "binary_search" => {
                // Save a little time by not comparing log_2(M) elements instead of M elements each time
                // where M is the length of the sorted portion of our slice
                for unsorted in 1..slice.len() {
                    // find where to insert unsorted element using binary search
                    // when binary_search() can't find the value it returns Err(i) where i is the index
                    // is where you could insert and keep the elements sorted
                    let (Ok(i) | Err(i)) = slice[..unsorted].binary_search(&slice[unsorted]);
                    // equivalent to:
                    // let i = match slice[..unsorted].binary_search(&slice[unsorted]) {
                    //     Ok(i) => i,
                    //     Err(i) => i,
                    // };

                    // rotate_right on slice will move all elements to the right by specified amount and
                    // wrap around. By rotating right by 1 on slice[i..=unsorted], we move the unsorted
                    // element to the ith index the binary_search() found for us, and we shift all the
                    // elements that need shifting over by one which is perfect.
                    slice[i..=unsorted].rotate_right(1);
                }
            },
            _ => {
                for unsorted in 1..slice.len() {
                    let mut i = unsorted;
                    while i > 0 && slice[i - 1] > slice[i] {
                        slice.swap(i - 1, i);
                        i -= 1;
                    }
                }
            }
        }
    }
}

#[test]
fn insertionsort_basic() {
    let mut input = vec![4, 2, 3, 5, 1];
    InsertionSort { variation: "basic" }.sort(&mut input);
    assert_eq!(input, &[1, 2, 3, 4, 5]);
}

#[test]
fn insertionsort_binary_search() {
    let mut input = vec![4, 2, 3, 5, 1];
    InsertionSort { variation: "binary_search" }.sort(&mut input);
    assert_eq!(input, &[1, 2, 3, 4, 5]);
}