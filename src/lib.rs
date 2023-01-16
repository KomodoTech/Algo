use std::collections::HashMap;

// 1512 Number of Good Pairs
// Given an array of integers nums, return the number of good pairs.
// A pair (i, j) is called good if nums[i] == nums[j] and i < j.
//
// Constraints:
// 1 <= nums.length <= 100
// 1 <= nums[i] <= 100
//
// Time O(N^2)
// SUM(N-1 + N-2 + ... + 1) = N * (N-1 - 1 + 1)/2 = N*(N-1)/2 which is an element of O(N^2)
// Space O(1)
// Type: Two pointers
pub fn num_identical_pairs_brute(nums: Vec<i32>) -> i32 {
    if nums.len() < 2 { return 0; }
    let mut count = 0;

    for i in 0..nums.len()-1 {
        for j in i+1..nums.len() {
            if nums[i] == nums[j] { count += 1;}
        }
    }
    count
}
// Standard 
// Iterate O(N) and check cache O(1) lookup as you go, if it is already there, increment count.
// Otherwise add to cache O(1) insert
//
// Time O(N)
// Space O(N)
// Type: Frequency Counter
pub fn num_identical_pairs_frequency_counter(nums: Vec<i32>) -> i32 {
    let mut frequency_counter: HashMap<i32, i32> = HashMap::new();

    nums.iter().fold(0, |count, &key| {
        match frequency_counter.get_mut(&key) {
            Some(value) => {
                *value += 1;
                count + *value - 1
            },
            None =>  {
                frequency_counter.insert(key, 1);
                count
            },
        }
    })
}
/// One of the keys is to realize that the concept of counting "good" pairs versus
/// all pairs, is the difference between calculating combinations versus counting the permutations. 
/// 
/// 
/// If nums is [1, 2, 3, 1, 1, 3, 1, 1], and you want to know how many "good" and "total" pairs of 1s
/// there are:
/// 
/// Total Pairs Of 1s:
/// There are 20 total pairs you can form with 5 ones. You choose one of the 5 ones as the first element
/// in the pair and then you have 4 choices for the second one. This is a permutation calculation.
/// 
/// nPr = n!/(n-r)!
/// 
/// where n is the number of 1s (5), and r is the number of elements we are selecting (pairs so 2). 
/// So 5P2 = 5!/(5-2)! = 5!/3! = 5*4 = 20
/// 
/// Those pairs are (using the indices of the ones):
/// 
/// (0, 3), (3, 0), (4, 0), (6, 0), (7, 0)
/// (0, 4), (3, 4), (4, 3), (6, 3), (7, 3)
/// (0, 6), (3, 6), (4, 6), (6, 4), (7, 4)
/// (0, 7), (3, 7), (4, 7), (6, 7), (7, 6)
/// 
/// Bad Pairs Of 1s:
/// 
/// Which of those are bad pairs?
/// 
///         (3, 0), (4, 0), (6, 0), (7, 0)
///                 (4, 3), (6, 3), (7, 3)
///                         (6, 4), (7, 4)
///                                 (7, 6)
/// 
/// Half of them in fact.
/// 
/// And the other half are the "good" pairs:
/// (0, 3)
/// (0, 4), (3, 4)
/// (0, 6), (3, 6), (4, 6)
/// (0, 7), (3, 7), (4, 7), (6, 7)
/// 
/// Good Pairs Of 1s:
/// 
/// You can see from looking at the good and the bad pairs, that really they are the
/// same elements but the order is just swapped. If you want to find the number of
/// good pairs directly, you can just find the "selection of items (1s) from the set of
/// distinct members (each 1 is distinct in our problem), such that the order of selection
/// does not matter (unlike permutations)" aka calculate the combination.
/// 
/// In our example, we would calculate nCr for n = 5 (number of 1s), and r = 2 (pairs):
/// 
/// nCr = n! / ((n-r)!*r!)
/// 
/// 5C2 = 5! / ((3)!*2!) = 20/2 = 10 
/// 
/// You can also see that in the problem n is variable but we're always looking at pairs so r = 2.
/// Hence, nC2 = n! / ((n-2)!*2!)  =  (n * (n-1))/2
/// 
/// This is correct, but if you're not completely convinced, another way to look at our example is:
/// You would probably naturally start at the first 1, and you would see that there are 4 other
/// 1s after it that it could pair with. Then you would look at the second 1 and see that it can
/// pair with 3 other 1s, etc. Your count of "good" pairs would be 4 + 3 + 2 + 1 = 10. 
/// 
/// If you did this for any n, you would get the arithmetic sum (9 year old Gauss punishment):
/// SUM from i = 1 to i = n-1 of i:
/// 1 + 2 + 3 + ... + (n-1) = n * (n-1 - 1 + 1) / 2 = (n * (n-1))/2
/// 
/// Same result!
/// 
/// So back to the problem at hand:
/// Start by populating a HashMap with each number in nums, and the number of occurences O(N).
/// Next we initialize a counter variable to 0
/// Next we want to iterate through each entry in the HashMap and for each value
/// (the number of duplicates for that num/key (e.g.  we had 5 1's so the key is 1 and the value is 5)),
/// we want to calculate the number of combinations: vC2 (where v is the value), and add it to our counter.
/// Finally we just return the counter
pub fn num_identical_pairs_combinatorics(nums: Vec<i32>) -> i32 {
    let mut frequency_counter: HashMap<i32, i32> = HashMap::new();

    // Calculate frequencies:
    nums.iter().for_each(|num| {
        match frequency_counter.get_mut(num) {
            Some(count) => {
                *count += 1;
            },
            None =>  {
                frequency_counter.insert(*num, 1);
            },
        }
    });

    // Count combinations:
    frequency_counter.iter().fold(0, |counter, (_key, value)| {
        let num_combinations = (*value) * (*value - 1) / 2;
        counter + num_combinations
    })
}

// use pass by ref to avoid moving ownership and keep option to reuse variable
// signature of &[T] instead of &Vec<T> because Vec<T> implements AsRef<[T]>, so this functions accepts both
// &Vec<T> and &[T]
pub fn num_identical_pairs_combinatorics_refactored(nums: &[i32]) -> u64 {
    let mut frequency_counter = HashMap::new();
    for num in nums {
        *frequency_counter.entry(num).or_insert(0) += 1;
    }
    frequency_counter.values().fold(0, |counter, value| {
        counter + (*value) * (*value - 1) / 2
    })
}

// Use array instead of HashMap for storing frequencies
// Works because of constraints on values stored in nums being positive
// so that you can use the value as index of array
pub fn num_identical_pairs_array(nums: &[i32]) -> u64 {
    const MAX_UNIQUE: usize = 100_000;
    let mut frequency_counter:[Option<u64>; MAX_UNIQUE] = [None; MAX_UNIQUE];
    for num in nums {
        match frequency_counter[*num as usize] {
            Some(freq) => {
                frequency_counter[*num as usize] = Some(freq + 1);
            },
            None => {
                frequency_counter[*num as usize] = Some(1);
            }
        }
    }
    frequency_counter
    .iter()
    .filter(|op| op.is_some())
    .fold(0, |counter, freq| {
        counter + (freq.unwrap()) * (freq.unwrap() - 1) / 2
    })
}

// 1470 Shuffle the Array
// Given the array nums consisting of 2n elements in the form [x1,x2,...,xn,y1,y2,...,yn].
// Return the array in the form [x1,y1,x2,y2,...,xn,yn].
//
// Constraints:
// 1 <= n <= 500
// nums.length == 2n
// 1 <= nums[i] <= 10^3
// Time O(N)
// Space O(N)
fn shuffle_brute(nums: Vec<i32>, n:i32) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(2*(n as usize));
    for i in 0..n as usize {
        out.push(nums[i]);
        out.push(nums[i + (n as usize)]);
    }
    out
}
/// Example:
/// nums = [x0, x1, x2, x3, y0, y1, y2, y3]
/// We want to get the following out at the end:
/// 
/// [x0, y0, x1, y1, x2, y2, x3, y3]
/// In order to make unpacking as straightforward as possible, we would want the step
/// before unpacking to look like this:
/// 
/// [x0*q+x0, y0*q+x1, x1*q+x2, y1*q+x3, x2*q+y0, y2*q+y1, x3*q+y2, y3*q+y3]
/// 
/// So the original value will always be used as the r_i and for the b_i we need a
/// generalizeable pattern to find the index:
/// 
/// index pattern:
/// [i/2, N + ((i+1)/2), i/2, N + ((i+1)/2), i/2, N + ((i+1)/2), i/2, N + ((i+1)/2)]
/// 
/// There are really two patterns here depending on the parity of the index
/// NOTE: you'll be reaching back to previous elements though so you need to take %q to make
/// sure that you're grabbing the right value whether or not it has already been modified
//
// Time O(N)
// Space O(1)
fn shuffle_charlies_grandparents(mut nums: Vec<i32>, n:i32) -> Vec<i32> {
    // q > r based off of max value stored in nums being 10^3
    let q = 1001;
    // Packing
    for i in 0..2*(n as usize) {
        // grab original value whether or not it was modified
        let r = nums[i] % q;
        match i%2 {
            0 => {
                let b = nums[i/2] % q;
                nums[i] = b*q + r;
            },
            1 => {
                let b = nums[n as usize + ((i-1)/2)] % q;
                nums[i] = b*q + r;
            },
            _ => panic!()
        }
    }
    // Unpacking
    nums.iter_mut().map(|num| *num/q).collect()
}
/// This approach packs all the nums[i] into nums[i+n] in the highest order bits
// This is way less complicated than the construction in the previous version
// Time O(N)
// Space O(1)
fn shuffle_bitpacking(mut nums: Vec<i32>, n: i32) -> Vec<i32> {
    let n = usize::try_from(n).expect("n cannot be converted to usize");
    // any value under 10^3 can fit into 10 bits
    let size = 10;
    // mask to retrieve only the lowest 10 bits
    let mask = 0x3FF;
    // Packing
    for i in 0..n {
        nums[i+n] |= nums[i] << size;
    }
    //Unpacking
    for i in 0..n {
        nums[2*i] = nums[i+n] >> 10;
        nums[2*i+1] = nums[i+n] & mask;
    }
    nums
}
// Trying out the same packing strategy but with a=bq+r
// Time O(N)
// Space O(1)
fn shuffle_charlies_grandparents_simplified(mut nums: Vec<i32>, n: i32) -> Vec<i32> {
    let n = usize::try_from(n).expect("n cannot be converted to usize");
    let q = 1001;
    // Packing
    for i in 0..n {
        nums[i+n] += nums[i]*q;
    }
    // Unpacking
    for i in 0..n {
        nums[2*i] = nums[i+n]/q;
        nums[2*i+1] = nums[i+n]%q;
    }
    nums
}

// 1480 Running Sum of 1D Array
// Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).
// Return the running sum of nums.
// Constraints
// 1 <= nums.length <= 1000
// -10^6 <= nums[i] <= 10^6
//
// Time O(N)
// Space O(N)
fn running_sum_brute(nums: Vec<i32>) -> Vec<i32> {
    let mut out = Vec::with_capacity(nums.len());
    nums.iter().fold(0, |acc, num| {
        let current_sum = acc + num;
        out.push(current_sum);
        current_sum
    });
    out
}

fn running_sum_map(nums: Vec<i32>) -> Vec<i32> {
    let mut sum = 0;
    nums
        .into_iter()
        .map(|num| {
                sum += num;
                sum
            })
        .collect::<Vec<i32>>()
}

fn running_sum_scan(nums: Vec<i32>) -> Vec<i32> {
    nums
        .iter()
        .scan(0, |sum, num|{
            *sum += num;
            Some(*sum)
        })
        .collect()
}

// 1929 Concatenation of Array
// Given an integer array nums of length n, you want to create an array ans of length 2n where ans[i] == nums[i] and ans[i + n] == nums[i] for 0 <= i < n (0-indexed).
// Specifically, ans is the concatenation of two nums arrays.
// Return the array ans.
//
// Constraints:
// n == nums.length
// 1 <= n <= 1000
// 1 <= nums[i] <= 1000
//
// Time O(N)
// Space O(N)
fn get_concatenation_brute(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();
    if n == 0 {return nums}
    let mut out = Vec::with_capacity(n*2);
    for i in 0..2*n {
        out.push(nums[i%n]);
    }
    out
}

fn get_concatenation_brute_alt(nums: Vec<i32>) -> Vec<i32> {
    let mut out = vec![0; nums.len() * 2];
    nums.iter().enumerate().for_each(|(i, &x)| {
        out[i] = x;
        out[i + nums.len()] = x;
    });
    out
}

fn get_concatenation_repeat(nums: Vec<i32>) -> Vec<i32> {
    nums.repeat(2)
}

fn get_concatenation_clone_push(nums: Vec<i32>) -> Vec<i32> {
    let mut out = nums.clone();
    for num in nums {
        out.push(num);
    }
    out
}

fn get_concatenation_clone_append(nums: Vec<i32>) -> Vec<i32> {
    let mut out = nums.clone();
    let mut nums_copy = nums.clone();
    out.append(&mut nums_copy); 
    out
}

fn get_concatenation_chain(nums: Vec<i32>) -> Vec<i32> {
    nums.iter().chain(nums.iter()).cloned().collect()
}

fn get_concatenation_chain_copied(nums: Vec<i32>) -> Vec<i32> {
    nums.iter().chain(nums.iter()).copied().collect()
}

fn get_concatenation_chain_map_equivalent(nums: Vec<i32>) -> Vec<i32> {
    nums.iter().chain(nums.iter()).map(|&x| x).collect()
}

fn get_concatenation_cycle(nums: Vec<i32>) -> Vec<i32> {
    nums.iter().cycle().take(nums.len() * 2).cloned().collect()
}

fn get_concatenation_extend(nums: Vec<i32>) -> Vec<i32> {
    let mut out = nums.clone();
    out.extend(nums);
    out
}

fn get_concatenation_concat(nums: Vec<i32>) -> Vec<i32> {
    [nums.clone(), nums].concat()
}

// 2011 Final Value of Variable After Performing Operations
// There is a programming language with only four operations and one variable X:
//     ++X and X++ increments the value of the variable X by 1.
//     --X and X-- decrements the value of the variable X by 1.
// Initially, the value of X is 0.
// Given an array of strings operations containing a list of operations, return the final value of X after performing all the operations.
//
// Constraints:
// 1 <= operations.length <= 100
// operations[i] will be either "++X", "X++", "--X", or "X--".
//
// Time O(N)
// Space O(1)
fn final_value_after_operations_brute(operations: Vec<String>) -> i32 {
    let mut out: i32 = 0;
    for operation in operations {
        match operation.as_str() {
            "++X" | "X++" => out += 1,
            "--X" | "X--" => out -= 1,
            _ => panic!(),
        }
    }
    out
}

// Time O(N)
// Space O(1)
fn final_value_after_operations_functional(operations: Vec<String>) -> i32 {
    operations
        .iter()
        .fold(0, |acc, operation| {
            if operation.contains('+') { acc + 1 } else { acc - 1} 
        })
}

// 2235 Add Two Integers
// Given two integers num1 and num2, return the sum of the two integers.
// Constraints
// -100 <= num1, num2 <= 100
//
// Time O(1)
// Space O(1)
fn sum_brute(num1: i32, num2: i32) -> i32 {
    num1 + num2
}

fn sum_bit_manipulation(num1: i32, num2: i32) -> i32 {
    if num2 == 0 {
        num1
    } else {
        sum_bit_manipulation(num1 ^ num2, (num1 & num2) << 1) 
    }
}

// 1108 Defanging an IP Address
// Given a valid (IPv4) IP address, return a defanged version of that IP address.
// A defanged IP address replaces every period "." with "[.]".
// Constraints:
// The given address is a valid IPv4 address
//
// Time O(N)
// Space O(N)
fn defang_i_paddr_brute(address: String) -> String {
    let num_bracket_chars = 6;
    let mut defang = String::with_capacity(address.len() + num_bracket_chars); 
    address.chars().for_each(|char| {
        match char {
            '.' => defang.push_str("[.]"),
            _ => defang.push(char),
        }
    });
    defang
}

// Time O(N)
// Space O(N)
// Replace calls match_indices on the String, which returns an iterator over the disjoint matches
// of the pattern provided with the String. Each element that the iterator returns is a tuple
// of the index where the pattern was found and the pattern.
fn defang_i_paddr_replace(address: String) -> String {
    address.replace('.', "[.]")
}

// 2469 Convert the Temperature
// You are given a non-negative floating point number rounded to two decimal places celsius, that denotes the temperature in Celsius.
// You should convert Celsius into Kelvin and Fahrenheit and return it as an array ans = [kelvin, fahrenheit].
// Return the array ans. Answers within 10^-5 of the actual answer will be accepted.
// Note that:
//     Kelvin = Celsius + 273.15
//     Fahrenheit = Celsius * 1.80 + 32.00
// Constraints:
// 0 <= celsius <= 1000
//
// Time O(1)
// Space O(1)
fn convert_temperature(celsius: f64) -> Vec<f64> {
    let kelvin = celsius + 273.15;
    let fahrenheit = celsius * 1.80 + 32.00;
    vec![kelvin, fahrenheit]
}

// 1920 Build Array From Permutation
// Given a zero-based permutation nums (0-indexed), build an array ans of the same length where 
// ans[i] = nums[nums[i]] for each 0 <= i < nums.length and return it.
// A zero-based permutation nums is an array of distinct integers from 0 to nums.length - 1 (inclusive).
// 
//Constraints:
// 1 <= nums.length <= 1000
// 0 <= nums[i] < nums.length
// The elements in nums are distinct.
//
// Follow-up: Can you solve it without using an extra space (i.e., O(1) memory)?
fn build_array_brute(nums: Vec<i32>) -> Vec<i32> {
    let length = nums.len();
    let mut ans: Vec<i32> = Vec::with_capacity(length);
    for i in 0..length {
        ans.push(nums[nums[i] as usize]);
    }
    ans
}

// Time O(N)
// Space O(N)
fn build_array_one_liner(nums: Vec<i32>) -> Vec<i32> {
    (0..nums.len()).map(|i| nums[nums[i] as usize]).collect()
}

// Time O(N)
// Space O(1)
//
// Summary:
// Stores two values as one element so you can get O(1) space
// Here instead of creating a new Vector, since we own nums, we're going to overload
// each element in it, in such a way that it stores BOTH the original element as well
// as nums[nums[i]].
//
// In order to acheive this, we're going to use a classic modulo technique (Charlie's Grandparents... sleeping head to toe):
//
// NOTE: (x + y)//z != x//z + y//z take x = 5, y = 5, z = 6 as counterexample
// Take a = qb + r, where b % q != 0, and r < q. We can extract b and r from a:
// b = a // q
//
// NOTE: 
// r < q, means that qb =< qb + r < q(b + 1)
// which means that ((qb + r) // q) is going to be b
//
// To get r out of a:
// r = a % q, since r % q != 0 unless r is 0, due to r < q
//
// So here, the original i32 stored in nums is our r, and nums[nums[i]] is going to be our b
// Since nums stores "distinct integers from 0 to nums.length - 1 inclusive", we can select a q
// that will always be larger than r, by setting it to nums.len()
pub fn build_array_small(mut nums: Vec<i32>) -> Vec<i32> {
    let q = nums.len() as i32;
    // turn the array into a=qb+r form
    (0..q as usize).for_each(|i| {
        // r captures the original value in array before we convert it into
        // a=qb+r below
        let r = nums[i];
        // This is the tricky part:
        // nums[i] might be less than i, so
        // nums[nums[i]] might already be in a=qb+r form (but it might not). If it is, and we 
        // take mod q we should get back the original r (aka the value that was there before which
        // is what we want. that's the permutation bit).
        // if it hasn't been made into a=qb+r, then taking mod won't change anything since
        // r < q
        let b = nums[nums[i] as usize] % q;
        // Now combine b and r in a in a manner that can be reversed.
        nums[i] = q*b + r;
    });
    // now turn all elements in nums (a's) into b's which is what we want to return
    // using integer division in order to "unwrap" the permuted value from the "combo" value
    (0..q as usize).for_each(|i| nums[i] /= q);
    nums

    // MORE CONCISELY:
    // let q = nums.len() as i32;
    // (0..q as usize).for_each(|i| nums[i] += (nums[nums[i] as usize] % q) * q);
    // (0..q as usize).for_each(|i| nums[i] /= q);
    // nums
}

// Time O(N)
// Space O(1)
// Summary:
// The idea here is to leverage constraints in order to once again store two numbers
// in one, but instead of using a=qb+r, we're just going to use an i32 to pack two
// values that can be held in 10 bits each (we'll even have 12 bits to spare).
//
// Since, the constraints tell us that nums.len() is smaller than 1000, we also
// know that any original values stored in nums will be smaller than 1000 (b1111101000).
// That means that all values can be stored in 10 bits. We can store the original
// values in the 10 least significant bits, and then store the permuted one in the next
// 10 bits like so:
// 
// nums[i] = (0000 0000 0000)(1000 1000 11)(01 0011 0100)
// where the lowest 10 bits hold the original values nums[i]
// the next 10 bits hold the new values nums[nums[i]]
// and the next 12 bits are just unused
// 
// We're achieving this setup by simply bitwise ORing each nums[i]
// with nums[nums[i] & mask] << 10
//
// Where the mask is 1023 which looks like this in our i32:  
// 0000 0000 0000 0000 0000 0011 1111 1111
// 
// The reason for the masking is similar to the reason why we take %q in our a=qb+r solution
// nums[i] may or may not already have been modified to hold both the original and the new values
// we care about. If it hasn't been modified, since we know the value fits in 10 bits, ORing
// it with 10 1's won't change it. If on the other hand it has been modified, we extract the
// orginal value via a bitwise AND with our mask. Exactly the same concept as modulo with
// a=qb+r.
// 
// Once we have our new value, we create a combo value by shifting the new value over 10
// bits and ORing the new and the old values together. Very clever.
// 
// After we created all of our combo values,
// In order to unpack everything and return just the desired 'new' values, we just perform
// a right shift by 10 on each element of the combo array, and we return the result. Exact
// same concept as a=qb+r, but leveraging constraints given to use very fast bitwise operations.
fn build_array_small_and_fast(mut nums: Vec<i32>) -> Vec<i32> {
    let mask = 1023;
    let q = nums.len();
    (0..q).for_each(|i| nums[i] |= (nums[nums[i] as usize] & mask) << 10);
    (0..q).for_each(|i| nums[i] >>= 10);
    nums
}



#[cfg(test)]
mod tests {
    use super::*;

    // NUMBER OF GOOD PAIRS
    #[test] 
    fn num_identical_pairs_brute_basic() {
        let nums = vec![1, 2, 3, 1, 1, 3, 1, 1];
        let expected = 11;
        assert_eq!(num_identical_pairs_brute(nums), expected);
    }
    #[test] 
    fn num_identical_pairs_frequency_counter_basic() {
        let nums = vec![1, 2, 3, 1, 1, 3, 1, 1];
        let expected = 11;
        assert_eq!(num_identical_pairs_frequency_counter(nums), expected);
    }
    #[test] 
    fn num_identical_pairs_combinatorics_basic() {
        let nums = vec![1, 2, 3, 1, 1, 3, 1, 1];
        let expected = 11;
        assert_eq!(num_identical_pairs_combinatorics(nums), expected);
    }
    #[test]
    fn num_identical_pairs_combinatorics_refactored_basic() {
        let nums = [1, 2, 3, 1, 1, 3, 1, 1];
        let expected = 11;
        assert_eq!(num_identical_pairs_combinatorics_refactored(&nums), expected);
    }
    #[test]
    fn num_identical_pairs_array_basic() {
        let nums = [1, 2, 3, 1, 1, 3, 1, 1];
        let expected = 11;
        assert_eq!(num_identical_pairs_array(&nums), expected);
    }
    // SHUFFLE THE ARRAY
    #[test]
    fn shuffle_brute_basic() {
        let nums = vec![1, 2, 3, 4, 4, 3, 2, 1];
        let n = 4;
        let expected = vec![1, 4, 2, 3, 3, 2, 4, 1];
        assert_eq!(shuffle_brute(nums, n), expected);
    }
    #[test]
    fn shuffle_charlies_grandparents_basic() {
        let nums = vec![1, 2, 3, 4, 4, 3, 2, 1];
        let n = 4;
        let expected = vec![1, 4, 2, 3, 3, 2, 4, 1];
        assert_eq!(shuffle_charlies_grandparents(nums, n), expected);
    }
    #[test]
    fn shuffle_charlies_grandparents_simplified_basic() {
        let nums = vec![1, 2, 3, 4, 4, 3, 2, 1];
        let n = 4;
        let expected = vec![1, 4, 2, 3, 3, 2, 4, 1];
        assert_eq!(shuffle_charlies_grandparents_simplified(nums, n), expected);
    }
    #[test]
    fn shuffle_bitpacking_basic() {
        let nums = vec![1, 2, 3, 4, 4, 3, 2, 1];
        let n = 4;
        let expected = vec![1, 4, 2, 3, 3, 2, 4, 1];
        assert_eq!(shuffle_bitpacking(nums, n), expected);
    }
    // RUNNING SUM
    #[test]
    fn running_sum_brute_basic() {
        let nums = vec![3, 1, 2, 10, 1];
        let expected = vec![3, 4, 6, 16, 17];
        assert_eq!(running_sum_brute(nums), expected);
    }
    #[test]
    fn running_sum_map_basic() {
        let nums = vec![3, 1, 2, 10, 1];
        let expected = vec![3, 4, 6, 16, 17];
        assert_eq!(running_sum_map(nums), expected);
    }
    #[test]
    fn running_sum_scan_basic() {
        let nums = vec![3, 1, 2, 10, 1];
        let expected = vec![3, 4, 6, 16, 17];
        assert_eq!(running_sum_scan(nums), expected);
    }
    // CONCATENATION OF ARRAY
    #[test]
    fn get_concatenation_brute_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_brute(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_brute_alt_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_brute_alt(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_repeat_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_repeat(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_clone_push_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_clone_push(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_append_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_clone_append(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_chain_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_chain(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_chain_copied_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_chain_copied(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_chain_map_equivalent_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_chain_map_equivalent(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_cycle_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_cycle(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_extend_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_extend(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    #[test]
    fn get_concatenation_concat_basic() {
        let nums = vec![1, 3, 2, 1];
        assert_eq!(get_concatenation_concat(nums), vec![1, 3, 2, 1, 1, 3, 2, 1]);
    }
    // FINAL VALUE AFTER OPERATIONS
    #[test]
    fn final_value_after_operations_brute_basic() {
        assert_eq!(final_value_after_operations_brute(vec!["--X".to_string(), "X++".to_string(), "X++".to_string(), "++X".to_string(), "X--".to_string()]), 1);
    }
    #[test]
    fn final_value_after_operations_functional_basic() {
        assert_eq!(final_value_after_operations_functional(vec!["--X".to_string(), "X++".to_string(), "X++".to_string(), "++X".to_string(), "X--".to_string()]), 1);
    }
    // ADD TWO INTEGERS
    #[test]
    fn sum_brute_basic() {
        assert_eq!(sum_brute(-10, 4), -6);
    }
    #[test]
    fn sum_bit_manipulation_basic() {
        assert_eq!(sum_bit_manipulation(-10, 4), -6);
    }
    // DEFANG IP ADDRESS
    #[test]
    fn defang_i_paddr_brute_basic() {
        let address = "255.100.50.0".to_string();
        assert_eq!(defang_i_paddr_brute(address), "255[.]100[.]50[.]0".to_string());
    }
    #[test]
    fn defang_i_paddr_replace_basic() {
        let address = "255.100.50.0".to_string();
        assert_eq!(defang_i_paddr_replace(address), "255[.]100[.]50[.]0".to_string());
    }
    // CONVERT THE TEMPERATURE
    #[test]
    fn convert_temperature_basic() {
        let celsius: f64 = 36.50;
        assert_eq!(convert_temperature(celsius), vec![309.65, 97.70]);
    }
    // BUILD ARRAY
    #[test]
    fn build_array_brute_basic() {
        let nums = vec![0, 2, 1, 5, 3, 4];
        let solution = vec![0, 1, 2, 4, 5, 3];
        assert_eq!(build_array_brute(nums), solution);
    }
    #[test]
    fn build_array_one_liner_basic() {
        let nums = vec![0, 2, 1, 5, 3, 4];
        let solution = vec![0, 1, 2, 4, 5, 3];
        assert_eq!(build_array_one_liner(nums), solution);
    }
    #[test]
    fn build_array_small_basic() {
        let nums = vec![0, 2, 1, 5, 3, 4];
        let solution = vec![0, 1, 2, 4, 5, 3];
        assert_eq!(build_array_small(nums), solution);
    }
    #[test]
    fn build_array_small_and_fast_basic() {
        let nums = vec![0, 2, 1, 5, 3, 4];
        let solution = vec![0, 1, 2, 4, 5, 3];
        assert_eq!(build_array_small_and_fast(nums), solution);
    }
}
