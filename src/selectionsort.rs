use super::Sorter;

pub struct SelectionSort<'a> {
    variation: &'a str,
}

impl<'a> Sorter for SelectionSort<'a> {
    // In-place: Yes
    // Stable: Only with O(N) extra space (using linked lists), or as a
    // variant of Insertion Sort instead of swapping the two items
    // Time: O(N^2)
    // Space: basic implementation is O(1)
    // Type: Selection
    // Concept: Find the smallest element of A, swap it with A[0], then find the
    // next smallest and swap it with A[1], etc. till the whole array is sorted
    // Invariant: all elements to the left of the current one should be sorted
    fn sort<T>(&self, slice: &mut [T])
    where
        T: Ord,
    {
        match self.variation {
            "cor_alternative" => {
                for unsorted in 0..slice.len() {
                    let smallest_in_rest =  slice[unsorted..]
                    .iter()
                    .enumerate()
                    // NOTE: min_by_key walks through the iterator and if it finds
                    // a minimum value (could be empty) it returns an Ok((usize, &T)) 
                    // aka an Ok with a reference to the (index: usize, value: &T) tuple created by enumerate().
                    // The way it determines a minimum is via the closure.
                    // We wish for our closure to convert a reference to a index value tuple into just its value
                    // so that only the value determines the minimum (not say minimum being
                    // determined by index or index + value or something else).
                    //
                    // There are some issues with lifetimes though:
                    // 
                    // Let t be the current element that min_by_key is going through:
                    // t is a reference to tuple that the iterator produced via the
                    // enumerate method. So it is of type &(usize, &T).
                    // 
                    // Incorrect: .min_by_key(|(_, v)| v) spits our "lifetime may not live long
                    // enough returning this value requires that `'1` must outlive `'2`"
                    //
                    // This is equivalent to saying:
                    //
                    // .min_by_key(|t| &t.1)
                    // 
                    // The issue there is that &t.1 is a reference to the tuple and not into the
                    // slice. It's a reference to a reference. We can't return it since the
                    // tuple will no longer be valid after the iteration ends. '1 will not outlive
                    // '2
                    //
                    // What we need to do is:
                    // 
                    // .min_by_key(|t| t.1)
                    // 
                    // which is accomplished by:
                    //
                    // .min_by_key(|&(_, v)| v)
                    //
                    // since the & here dereferences the tuple, thus allowing v to actually reference the slice
                    .min_by_key(|&(_, v)| v)
                    // adjust since the index we get back from min_by_key is the index for the unsorted
                    // portion of the slice only, not the full thing
                    .map(|(i, _)| unsorted + i)
                    .expect("slice is non-empty");

                    if unsorted != smallest_in_rest {
                        slice.swap(unsorted, smallest_in_rest);
                    }
                }
            },
            "cor" => {
                for unsorted in 0..slice.len() {
                    let mut smallest_in_rest = unsorted;
                    for i in (unsorted + 1)..slice.len() {
                        if slice[i] < slice[smallest_in_rest] {
                            smallest_in_rest = i;
                        }
                    }
                    if unsorted != smallest_in_rest {
                        slice.swap(unsorted, smallest_in_rest);
                    }
                }
            },
            _ => {
                for i in 0..slice.len() {
                    // find smallest element of remaining unsorted elements
                    let mut smallest_unsorted_index = i;
                    for j in (i + 1)..slice.len() {
                        match slice[j] < slice[smallest_unsorted_index] {
                            true => {
                                smallest_unsorted_index = j;
                            }
                            _ => (),
                        }
                    }
                    // swap smallest element with current element (unsorted)
                    slice.swap(i, smallest_unsorted_index);
                }
            }
        }
    }
}

#[test]
fn selectionsort_basic() {
    let mut input = vec![4, 2, 3, 5, 1];
    SelectionSort { variation: "basic" }.sort(&mut input);
    let expected = vec![1, 2, 3, 4, 5];
    assert_eq!(input, expected);
}

#[test]
fn selectionsort_basic_empty() {
    let mut input:Vec<i32> = vec![];
    SelectionSort { variation: "basic" }.sort(&mut input);
    let expected = vec![];
    assert_eq!(input, expected);
}

#[test]
fn selectionsort_cor() {
    let mut input = vec![4, 2, 3, 5, 1];
    SelectionSort { variation: "cor" }.sort(&mut input);
    let expected = vec![1, 2, 3, 4, 5];
    assert_eq!(input, expected);
}

#[test]
fn selectionsort_cor_empty() {
    let mut input:Vec<i32> = vec![];
    SelectionSort { variation: "cor" }.sort(&mut input);
    let expected = vec![];
    assert_eq!(input, expected);
}

#[test]
fn selectionsort_cor_alternative() {
    let mut input = vec![4, 2, 3, 5, 1];
    SelectionSort { variation: "cor_alternative" }.sort(&mut input);
    let expected = vec![1, 2, 3, 4, 5];
    assert_eq!(input, expected);
}

#[test]
fn selectionsort_cor_alternative_empty() {
    let mut input:Vec<i32> = vec![];
    SelectionSort { variation: "cor_alternative" }.sort(&mut input);
    let expected = vec![];
    assert_eq!(input, expected);
}