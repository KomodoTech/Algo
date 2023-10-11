use super::Sorter;
use rand::Rng;
pub struct QuickSort<'a> {
    variation: &'a str,
}

/// partition A[p..r] into possibly empty A[p..q-1] and A[q+1..r]
/// such that each element of A[p..q-1] is less than or equal to A[q]
/// which is itself less than or equal to A[q+1..r].
/// 
/// Initial pivot index q is chosen at random.
/// In this version, this is acheived by immediately moving the pivot
/// to the end of the array by swapping it with the last element, then
/// using two pointers (left, current) to partition. The current pointer
/// will simply be pointing to the current element we are comparing as we
/// iterate through our array. At each iteration all elements with indices
/// less than or equal to the left pointer will be less than or equal to the
/// pivot value. So each time A[current] <= pivot, we will be swapping
/// A[current] with A[left + 1] and incrementing left. This will effectively
/// move a value that needs to be in the left partition to the boundary of
/// the current left partition and move the partition boundary up by one to
/// reflect the change. We end the iteration when current hits the pivot/end
/// of the array, at which point we swap the pivot with A[left+1], inserting
/// it in its proper place (one to the right of the left boundary). The element
/// we swapped it with is guaranteed to be part of the right partition since
/// we saw all the elements and chose not to place it in the left partition.
fn partition_random_pivot_end<T>(slice: &mut [T])
    where T: Ord
{
    let end = slice.len()-1;
    let initial_pivot_index: usize = rand::thread_rng().gen_range(0..slice.len());
    slice.swap(initial_pivot_index, end);

    let mut left = 0;

    for current in 0..end {
        if slice[current] <= slice[end] {
            slice.swap(current, left);
            left += 1;
        }
    }

    slice.swap(left, end);
}

impl<'a> Sorter for QuickSort<'a> {
    // In-Place: Yes
    // Stable: No
    // Time: O(n^2)
    // Space: O(lg(n))
    // Type: Paritioning (Divide and Conquer)
    // Concept:
    //  Divide: Partition A[p..r] into possibly empty A[p..q-1] and A[q+1..r]
    //      such that each element of A[p..q-1] is leq to A[q], which is itself
    //      leq than A[q+1..r]. Computed index q as part of this partitioning.
    //  Conquer: Sort A[p..q-1] and A[q+1..r] by recursively calling quicksort
    //  Combine: No additional work to combine since everything is in the right order
    //
    // Notes: Despite the worst case of O(n^2), it's often a really good choice
    // because it's very efficient on average (expected running time of O(n*lg(n)))
    // It works well even in virtual-memory environments (good cache locality).
    // Randomized version of the algorithm to get good average runtime.

    fn sort<T>(&self, slice: &mut [T])
    where
        T: Ord,
    {
        match self.variation {
            "cor" => {
                todo!()
            },
            _ => {
                // PARTITION(A[p..r])
                // Can't clone needs to produce slices
                // QUICKSORT(A[p..q-1])
                // QUICKSORT(A[q+1..r])
            }
        }
    }
}
