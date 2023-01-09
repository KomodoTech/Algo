
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
    let mut result: i32 = 0;
    for operation in operations {
        match operation.as_str() {
            "++X" | "X++" => result += 1,
            "--X" | "X--" => result -= 1,
            _ => panic!(),
        }
    }
    result
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
